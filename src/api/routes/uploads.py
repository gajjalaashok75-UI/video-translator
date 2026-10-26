"""
Upload endpoint.

The body is streamed straight to the staging file, one chunk at a time, and the
byte count is checked as it goes. That is the whole reason this does not use
multipart form parsing: a multipart upload is buffered by the framework before
the handler sees it, so a 4 GiB file would be written to a spool file and then
copied to its destination, and a limit could only be applied after the whole body
had already been accepted. Streaming means the ceiling is enforced *during* the
transfer, and a rejected upload has cost at most one chunk of disk.

The client supplies the filename as a query parameter (`?filename=clip.mp4`) and
the raw file as the body. It is sanitised and its extension checked before a
single byte is written, so an obviously wrong upload is refused for free.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, Request

from src.api import media, settings, uploads
from src.api.errors import ApiError, ErrorCode
from src.api.schemas import MediaInfo, UploadResponse
from src.core import jobs as joblib

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/uploads", tags=["uploads"])


def _as_response(upload: uploads.Upload) -> UploadResponse:
    return UploadResponse(
        upload_id=upload.upload_id,
        filename=upload.filename,
        bytes=upload.bytes,
        media=MediaInfo(**upload.media),
        status=upload.status,
        created_at=upload.created_at,
        claimed_by=upload.claimed_by,
    )


@router.post("", response_model=UploadResponse, summary="Upload a source video")
async def create_upload(
    request: Request,
    filename: str = Query(
        description="Original filename, used for the extension and display. "
                    "Sanitised before use; never used as a path."),
) -> UploadResponse:
    """
    Stage a source video for a later render.

    Streams the request body to disk, rejecting anything over
    `settings.MAX_UPLOAD_BYTES` part-way through rather than after the fact, and
    then probes the result with the pipeline's own validation. The returned
    `upload_id` is the only handle needed to start a job; it is single-use.

    A rejected upload leaves nothing behind: the staging directory is removed
    before the error is raised.
    """
    try:
        uploads.validate_extension(filename)
    except uploads.UploadError as exc:
        raise ApiError(400, ErrorCode.INVALID_UPLOAD, str(exc),
                       field="filename") from exc

    # Content-Length is advisory: a client can lie or omit it, so it is used only
    # to reject an obviously oversized upload early, never as the enforcement
    # point.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > settings.MAX_UPLOAD_BYTES:
        raise ApiError(
            413, ErrorCode.UPLOAD_TOO_LARGE,
            f"Upload is larger than the {settings.MAX_UPLOAD_BYTES} byte limit.",
            details={"max_bytes": settings.MAX_UPLOAD_BYTES},
        )

    upload_id = uploads.new_upload_id()
    display_name = uploads.safe_filename(filename)
    try:
        upload = uploads.begin(upload_id, display_name)
    except uploads.UploadError as exc:
        raise ApiError(500, ErrorCode.INTERNAL_ERROR, str(exc)) from exc

    written = 0
    try:
        with open(upload.path, "wb") as handle:
            async for chunk in request.stream():
                if not chunk:
                    continue
                written += len(chunk)
                if written > settings.MAX_UPLOAD_BYTES:
                    raise ApiError(
                        413, ErrorCode.UPLOAD_TOO_LARGE,
                        f"Upload exceeds the {settings.MAX_UPLOAD_BYTES} byte "
                        "limit; nothing was stored.",
                        details={"max_bytes": settings.MAX_UPLOAD_BYTES},
                    )
                handle.write(chunk)
        if written == 0:
            raise ApiError(400, ErrorCode.INVALID_UPLOAD,
                           "The request body was empty.")
    except ApiError:
        uploads.discard(upload)
        raise
    except Exception as exc:  # noqa: BLE001 - never leak a partial file
        uploads.discard(upload)
        logger.warning("Upload %s failed while streaming: %s", upload_id, exc)
        raise ApiError(
            400, ErrorCode.INVALID_UPLOAD,
            "The upload could not be read from the connection.",
        ) from exc

    upload.bytes = written
    # Validation and probing shell out to ffprobe, which is a subprocess with a
    # 30-second timeout. Off the event loop, or a slow probe on one upload would
    # stall every other request for its duration.
    ok, reason = await asyncio.to_thread(media.validate_upload, upload.path)
    if not ok:
        uploads.discard(upload)
        raise ApiError(422, ErrorCode.INVALID_UPLOAD, reason)

    upload.media = await asyncio.to_thread(media.probe_media, upload.path)
    uploads.write_meta(upload)
    logger.info("Staged upload %s (%s, %d bytes)", upload_id, display_name, written)
    return _as_response(upload)


@router.get("/{upload_id}", response_model=UploadResponse, summary="Upload status")
def get_upload(upload_id: str) -> UploadResponse:
    """
    A staged upload's metadata.

    Exists so a client can confirm a staging step succeeded, and see what the
    server made of the file, without keeping the POST response.
    """
    try:
        upload = uploads.load(upload_id)
    except uploads.UploadNotFound as exc:
        raise ApiError(404, ErrorCode.UPLOAD_NOT_FOUND, str(exc)) from exc
    except uploads.UploadError as exc:
        raise ApiError(400, ErrorCode.INVALID_UPLOAD, str(exc)) from exc
    return _as_response(upload)


@router.delete("/{upload_id}", summary="Discard a staged upload")
def delete_upload(upload_id: str) -> dict:
    """
    Remove a staged upload that will not be used.

    Refuses to delete one a job has claimed: the file now belongs to the job's
    durable workspace, and removing it would break that job's ability to resume.
    """
    if not joblib.is_valid_job_id(upload_id):
        raise ApiError(404, ErrorCode.INVALID_UPLOAD,
                       f"Unknown upload: {upload_id!r}")
    try:
        upload = uploads.load(upload_id)
    except uploads.UploadNotFound as exc:
        raise ApiError(404, ErrorCode.UPLOAD_NOT_FOUND, str(exc)) from exc
    except uploads.UploadError as exc:
        raise ApiError(400, ErrorCode.INVALID_UPLOAD, str(exc)) from exc
    if upload.status == uploads.STATUS_CLAIMED:
        raise ApiError(
            409, ErrorCode.UPLOAD_ALREADY_CLAIMED,
            f"Upload {upload_id!r} was claimed by job "
            f"{upload.claimed_by!r}; delete the job instead.",
            details={"claimed_by": upload.claimed_by},
        )
    uploads.discard(upload)
    return {"upload_id": upload_id, "deleted": True}
