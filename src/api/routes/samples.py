"""
Bundled sample endpoints.

Three read-only-ish operations, all safe to call on a machine with no models:

    GET  /api/samples                 list the allow-listed samples
    GET  /api/samples/{id}/media      stream a sample's media
    POST /api/samples/{id}/claim      stage it as an ordinary upload

The claim endpoint is the one that matters for the demo: it produces the same
`UploadResponse` a browser upload produces, so the demo starts a real job
through the unchanged `POST /api/jobs` path. There is no demo-only runner.

Every path is derived from the server-side allow-list in `src/api/samples.py`;
no client string ever becomes a path, and `media` re-checks containment before
serving.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import FileResponse

from src.api import artifacts as artifactlib
from src.api import samples as samplelib
from src.api import uploads
from src.api.errors import ApiError, ErrorCode
from src.api.schemas import (
    MediaInfo,
    SampleInfo,
    SampleListResponse,
    UploadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/samples", tags=["samples"])


@router.get("", response_model=SampleListResponse, summary="List demo samples")
def list_samples() -> SampleListResponse:
    """
    The bundled samples this installation can offer.

    Never a directory listing: the response is built from the curated table, so
    only files someone added on purpose are reachable. A sample whose media is
    missing is still listed, with `available: false` and a reason, because "this
    exists but is not on this machine" is more useful to the demo page than a
    silently short list.
    """
    items = samplelib.list_samples()
    return SampleListResponse(
        samples=[SampleInfo(**item) for item in items],
        total=len(items),
    )


@router.get("/{sample_id}/media", summary="Stream a sample's media")
def sample_media(sample_id: str) -> FileResponse:
    """
    Serve a sample's video so a browser can preview it before starting a demo.

    `FileResponse` supports byte ranges, so the preview seeks without downloading
    the whole clip. The media type comes from the file's extension, and the path
    is re-derived from the allow-list and re-checked for containment on every
    request, so no request can name a file outside the samples directory.
    """
    try:
        path = samplelib.media_path(sample_id)
    except samplelib.SampleNotFound as exc:
        raise ApiError(404, ErrorCode.NOT_FOUND, str(exc)) from exc
    except samplelib.SampleError as exc:
        raise ApiError(400, ErrorCode.INVALID_REQUEST, str(exc)) from exc
    if not path.is_file():
        raise ApiError(
            404, ErrorCode.NOT_FOUND,
            f"Sample {sample_id!r} is not available on this machine.")
    return FileResponse(
        path,
        media_type=artifactlib.media_type_for(path),
        filename=path.name,
    )


@router.post("/{sample_id}/claim", response_model=UploadResponse,
             summary="Stage a sample as an upload")
async def claim_sample(sample_id: str) -> UploadResponse:
    """
    Stage an allow-listed sample as an ordinary upload.

    The sample is *copied* into the upload staging area and returned with the
    same shape as `POST /api/uploads`, so the client starts a demo with the
    normal `POST /api/jobs` call and gets a normal durable job. The bundled
    original is never moved or modified.

    Copying and probing are blocking file/ffprobe work, so they run in a worker
    thread rather than on the event loop.
    """
    try:
        upload = await asyncio.to_thread(samplelib.stage, sample_id)
    except samplelib.SampleNotFound as exc:
        raise ApiError(404, ErrorCode.NOT_FOUND, str(exc)) from exc
    except uploads.UploadError as exc:
        raise ApiError(422, ErrorCode.INVALID_UPLOAD, str(exc)) from exc
    except samplelib.SampleError as exc:
        raise ApiError(400, ErrorCode.INVALID_REQUEST, str(exc)) from exc

    return UploadResponse(
        upload_id=upload.upload_id,
        filename=upload.filename,
        bytes=upload.bytes,
        media=MediaInfo(**upload.media),
        status=upload.status,
        created_at=upload.created_at,
        claimed_by=upload.claimed_by,
    )
