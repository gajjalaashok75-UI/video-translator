"""
Job endpoints: create, inspect, cancel, follow and download.

The ordering in `create_job` is the part worth reading. Everything that can fail
for a *client* reason — an unknown upload, a bad language, an impossible speaker
range — is checked before a job directory exists. Once a directory does exist the
steps are ordered so that a failure leaves the least behind:

    1. validate the request            (nothing on disk yet)
    2. create the job directory        (empty, cheap to remove)
    3. claim the upload into it        (the file moves in)
    4. write the manifest              (the job becomes resumable)
    5. submit the runner               (execution starts)

If step 3 fails the job directory is removed, so a rejected request does not
leave an orphan. After step 4 the job is a real, resumable job even if the
process dies before step 5, which is the correct outcome: the durable record
exists and `resume_job` can continue it.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from src.api import artifacts as artifactlib
from src.api import events, joblogs, jobstate, preflight, recovery, settings, uploads
from src.api import validation
from src.api.errors import ApiError, ErrorCode
from src.api.events import EventType
from src.api.manager import JobAlreadyActive, JobManager, get_manager
from src.api.runner import JobConfig, JobRunner, RunMode
from src.api.schemas import (
    ArtifactDescriptor,
    ArtifactListResponse,
    CancelResponse,
    CleanupLevel,
    DeleteResponse,
    JobCreateRequest,
    JobListResponse,
    JobState,
    JobStatusResponse,
    LogsResponse,
    OutputResponse,
    PreflightResponse,
    ResumeResponse,
)
from src.core import jobs as joblib
from src.core.manifest import Manifest, software_versions
from src.utils import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _manager() -> JobManager:
    return get_manager()


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------
def _build_config(request: JobCreateRequest) -> tuple[JobConfig, str, str]:
    """
    Turn a request into a `JobConfig`, applying any quality preset.

    The preset is resolved here rather than being forwarded to the pipeline. The
    pipeline applies a preset by overriding four arguments before it records the
    run configuration, so forwarding the preset *and* letting the API record the
    config would make the two disagree, and `differs_from` would then refuse to
    resume the very job it had just created. Applying it in one place — here —
    keeps the recorded configuration and the effective one identical.
    """
    source, target = validation.validate_request(request)

    values = request.model_dump()
    values["source_lang"] = source
    values["target_lang"] = target
    values.pop("upload_id", None)
    preset = values.pop("quality_preset", None)
    if preset:
        for key, value in config.preset_config(preset).items():
            if key in JobConfig.__dataclass_fields__:
                values[key] = value
    return JobConfig(**values), source, target


def _run_config(cfg: JobConfig) -> dict:
    """
    The run configuration as the pipeline will record it.

    Mirrors `VideoTranslator._run_config`: the same key set, in the same form.
    The manifest is written with this before the render starts, and the pipeline
    recomputes it and compares, so the two must agree exactly — a mismatch is
    reported as configuration drift and the job refuses to run. Keeping the key
    list in one place here means a new pipeline key that matters for caching has
    to be added in both, which the drift check makes loud rather than silent.
    """
    return {key: getattr(cfg, key) for key in (
        "source_lang", "target_lang", "audio_model_name", "tts_model_name",
        "translation_model_name", "context_model_name",
        "transcription_model_name", "optimize_translation",
        "enable_diarization", "diarization_model", "min_speakers",
        "max_speakers", "enable_time_stretch", "enable_vad",
        "vad_min_silence_duration_ms", "enable_lipsync",
        "lipsync_model_name", "live_portrait_mode",
        "enable_visual_translation", "ocr_model_name", "tts_voice",
        "transcription_beam_size", "tts_enable_cfg",
        "enable_audio_enhancement", "chunk_duration", "subtitle_mode",
    )}


def _claim_upload(upload_id: str, job_root) -> "object":
    """Claim the upload, converting upload errors into HTTP errors."""
    try:
        return uploads.claim(upload_id, job_root)
    except uploads.UploadNotFound as exc:
        raise ApiError(404, ErrorCode.UPLOAD_NOT_FOUND, str(exc)) from exc
    except uploads.UploadAlreadyClaimed as exc:
        raise ApiError(409, ErrorCode.UPLOAD_ALREADY_CLAIMED,
                       str(exc)) from exc
    except uploads.UploadError as exc:
        raise ApiError(400, ErrorCode.INVALID_UPLOAD, str(exc)) from exc


@router.post("/preflight", response_model=PreflightResponse,
             summary="Check a render before starting it")
def preflight_job(request: JobCreateRequest) -> PreflightResponse:
    """
    Report whether a prospective render is ready, and why not if it is not.

    Advisory only: the checks are computed from the same registries
    `POST /api/jobs` validates against, but creating a job re-runs every check
    from scratch. This exists so the UI can explain a disabled Start button
    instead of leaving a user to guess, and so a problem is caught before a job
    directory is created.

    Read-only and cheap for everything but the component probes, which it shares
    with `/api/models`.
    """
    return preflight.build_preflight(request)


@router.post("", response_model=JobStatusResponse, status_code=202,
             summary="Start a render")
async def create_job(request: JobCreateRequest,
                     http: Request) -> JobStatusResponse:
    """
    Create a job from a staged upload and start it.

    Returns 202 with the job's initial state; execution happens on a worker
    thread and its progress is on `events_url`. The upload is claimed exactly
    once, so a second request with the same `upload_id` is a 409 rather than a
    second render of the same file.

    The setup work — creating the job directory, moving the source in, writing
    the manifest — is blocking file I/O, so it runs in a worker thread rather
    than on the event loop. A multi-gigabyte `os.replace` on a busy filesystem
    would otherwise stall every other request, including `/api/health`.
    """
    try:
        cfg, source, target = _build_config(request)
    except validation.ValidationError as exc:
        raise ApiError(422, ErrorCode.INVALID_SETTINGS, exc.message,
                       field=exc.field) from exc

    # Fail fast on a spent upload before creating anything.
    try:
        existing = await asyncio.to_thread(uploads.load, request.upload_id)
    except uploads.UploadNotFound as exc:
        raise ApiError(404, ErrorCode.UPLOAD_NOT_FOUND, str(exc)) from exc
    except uploads.UploadError as exc:
        raise ApiError(400, ErrorCode.INVALID_UPLOAD, str(exc)) from exc
    if existing.status == uploads.STATUS_CLAIMED:
        raise ApiError(
            409, ErrorCode.UPLOAD_ALREADY_CLAIMED,
            f"Upload {request.upload_id!r} was already claimed by job "
            f"{existing.claimed_by!r}.",
            details={"claimed_by": existing.claimed_by},
        )

    loop = asyncio.get_running_loop()
    job_id, source_path = await asyncio.to_thread(
        _prepare_job, request.upload_id, existing.filename, cfg, source, target)

    mgr = _manager()
    runner = JobRunner(
        job_id=job_id,
        source_path=source_path,
        config=cfg,
        loop=loop,
        hf_token=_hf_token(),
        on_finished=mgr.on_runner_finished,
    )
    mgr.submit(runner)

    status = jobstate.build_status(job_id, manager=mgr)
    if status is None:
        # Should be impossible: the manifest was just written. Report it rather
        # than returning a 500 with no context.
        raise ApiError(
            500, ErrorCode.INTERNAL_ERROR,
            f"Job {job_id} was created but could not be read back.")
    return status


def _prepare_job(upload_id: str, filename: str, cfg: JobConfig,
                 source_lang: str, target_lang: str) -> tuple[str, Path]:
    """
    The blocking half of job creation, run in a worker thread.

    Creates the job directory, claims the upload into it and writes the
    manifest. On a failure after the directory exists the directory is removed,
    so a rejected request leaves no orphan that a later listing would show as a
    real job.
    """
    # The job id is prefixed with the source stem so `ls jobs/` is readable.
    prefix = filename.rsplit(".", 1)[0] or "job"
    job = joblib.Job.create(prefix=prefix)
    try:
        source_path = _claim_upload(upload_id, job.root)
    except HTTPException:
        shutil.rmtree(job.root, ignore_errors=True)
        raise

    try:
        manifest = Manifest.create(
            job,
            source_path=source_path,
            source_lang=source_lang,
            target_lang=target_lang,
            run_config=_run_config(cfg),
            software=software_versions(),
        )
        manifest.job_id = job.job_id
        manifest.save()
        job.log(f"job created from upload {upload_id}")
    except Exception:
        # The upload has already moved into the job directory, so the job is
        # removed with it: leaving the directory would present an unusable job
        # and strand the source file inside it.
        shutil.rmtree(job.root, ignore_errors=True)
        raise
    return job.job_id, Path(source_path)


def _hf_token() -> Optional[str]:
    """
    The Hugging Face token from the server environment.

    Read per request rather than captured at import, so rotating the environment
    does not require a restart. Never accepted from the client and never echoed:
    a token in a request body would end up in logs and in a browser's memory.
    """
    import os
    token = (os.environ.get("HF_TOKEN") or "").strip()
    return token or None


# ---------------------------------------------------------------------------
# Listing and inspection
# ---------------------------------------------------------------------------
@router.get("", response_model=JobListResponse, summary="List jobs")
def list_jobs(
    limit: int = Query(default=50, ge=1, le=500,
                       description="Maximum number of jobs to return."),
    state: Optional[JobState] = Query(
        default=None,
        description="Only jobs in this state. Applied after the state is "
                    "computed, so 'interrupted' works."),
    sort: str = Query(default="newest",
                      pattern="^(newest|oldest)$",
                      description="Order by creation time."),
) -> JobListResponse:
    """
    Recent jobs, read from the jobs directory.

    Every job on disk is included, not only those this process is running: a job
    from a previous process is still a job, and its state comes from its
    manifest. `active` and `pending` count only this process's queue.

    `state` filters on the computed state, which is the one a client sees — so
    filtering by `interrupted` finds exactly the jobs whose Resume button would
    work, which is what a recovery screen needs. Filtering after the manifest read
    (rather than by directory name) means a filter can scan more than `limit`
    jobs; the directory listing is bounded separately to keep that from being
    unbounded work.
    """
    # Read more ids than requested when filtering, because the filter can discard
    # most of them. Capped so a filter can never make this scan the whole history.
    scan_limit = limit if state is None else min(500, max(limit * 4, limit))
    ids = joblib.list_jobs()[:scan_limit]
    if sort == "oldest":
        ids = list(reversed(ids))
    mgr = _manager()
    statuses = jobstate.build_statuses(ids, manager=mgr)
    if state is not None:
        statuses = [s for s in statuses if s.state == state][:limit]
    return JobListResponse(
        jobs=statuses,
        total=len(statuses),
        active=mgr.active_count,
        pending=mgr.pending_count,
    )


@router.get("/{job_id}", response_model=JobStatusResponse, summary="Job status")
def get_job(job_id: str) -> JobStatusResponse:
    """
    One job's state.

    Durable fields come from the manifest and are correct after a restart; live
    fields are present only when this process is running the job. A job whose
    manifest says `RUNNING` with no runner in this process is reported
    `interrupted`, because the process that was writing it is gone — and that is
    the state `resume.can_resume` is true for.
    """
    if not joblib.is_valid_job_id(job_id):
        raise ApiError(404, ErrorCode.INVALID_JOB_ID,
                       f"{job_id!r} is not a valid job id.")
    status = jobstate.build_status(job_id)
    if status is None:
        raise ApiError(404, ErrorCode.JOB_NOT_FOUND,
                       f"No job {job_id!r} on disk.")
    return status


@router.post("/{job_id}/resume", response_model=ResumeResponse,
             status_code=202, summary="Resume an interrupted job")
async def resume_job_route(job_id: str) -> ResumeResponse:
    """
    Continue a job that was interrupted, failed or cancelled.

    This does not rebuild the job's configuration. `resume_job` on the pipeline
    replays the configuration recorded in the manifest and lets each stage's
    fingerprint decide whether its work is still valid, so completed stages are
    reused rather than redone. The API's job is to check that a resume is *safe*
    (see `recovery.evaluate`), not to decide what the pipeline should redo.

    The resume goes through the same queue as a new job, so it obeys
    `max_concurrent` and cannot run alongside a render that is already using the
    GPU. Eligibility and duplicate-protection are both checked here and again,
    race-safely, inside the manager: two simultaneous requests cannot both start a
    runner for one job.
    """
    if not joblib.is_valid_job_id(job_id):
        raise ApiError(404, ErrorCode.INVALID_JOB_ID,
                       f"{job_id!r} is not a valid job id.")

    mgr = _manager()
    decision = recovery.evaluate(job_id, manager=mgr)
    if decision is None:
        raise ApiError(404, ErrorCode.JOB_NOT_FOUND,
                       f"No job {job_id!r} on disk.")
    if not decision.can_resume:
        # A job this process already owns is a different situation from one that
        # is simply finished, and a client acts on them differently (wait, versus
        # offer nothing), so they get different codes.
        code = (ErrorCode.JOB_ALREADY_ACTIVE
                if decision.reason in ("already_running", "already_queued")
                else ErrorCode.JOB_NOT_RESUMABLE)
        raise ApiError(409, code, decision.message,
                       details={"reason": decision.reason})

    manifest = jobstate.manifest_for(job_id)
    if manifest is None:
        raise ApiError(404, ErrorCode.JOB_NOT_FOUND,
                       f"No job {job_id!r} on disk.")
    source_path = Path(str((manifest.source or {}).get("path")))

    loop = asyncio.get_running_loop()
    runner = JobRunner(
        job_id=job_id,
        source_path=source_path,
        config=None,
        loop=loop,
        hf_token=_hf_token(),
        on_finished=mgr.on_runner_finished,
        mode=RunMode.RESUME,
    )
    try:
        mgr.submit_resume(job_id, runner)
    except JobAlreadyActive as exc:
        raise ApiError(
            409, ErrorCode.JOB_ALREADY_ACTIVE,
            f"Job {job_id!r} is already active in this process.",
            details={"state": exc.state},
        ) from exc

    status = jobstate.build_status(job_id, manager=mgr)
    state = status.state if status is not None else JobState.QUEUED
    return ResumeResponse(
        job_id=job_id,
        state=state,
        run_mode=RunMode.RESUME,
        message="Resume queued; completed stages will be reused.",
        queue_position=mgr.queue_position(job_id),
    )


@router.post("/{job_id}/cancel", response_model=CancelResponse,
             summary="Cancel a job")
def cancel_job(job_id: str) -> CancelResponse:
    """
    Ask a job to stop.

    A queued job is cancelled outright and never runs. A running job is flagged
    and stops at the pipeline's next checkpoint — between stages, not inside a
    model call — so cancellation is not instant and the response says
    `cancelling` to make that visible. Cancelling a job that has already
    finished is a no-op that reports its final state, not an error.
    """
    if not joblib.is_valid_job_id(job_id):
        raise ApiError(404, ErrorCode.INVALID_JOB_ID,
                       f"{job_id!r} is not a valid job id.")
    mgr = _manager()
    state = mgr.cancel(job_id)
    if state is None:
        # Not tracked here. It may still exist on disk from another process, in
        # which case there is nothing running to cancel.
        status = jobstate.build_status(job_id, manager=mgr)
        if status is None:
            raise ApiError(404, ErrorCode.JOB_NOT_FOUND,
                           f"No job {job_id!r} on disk.")
        raise ApiError(
            409, ErrorCode.JOB_NOT_RESUMABLE,
            (f"Job {job_id!r} is not running in this process (state "
             f"{status.state.value}); there is nothing to cancel."),
            details={"state": status.state.value},
        )
    mapped = {
        "queued": JobState.QUEUED,
        "cancelling": JobState.RUNNING,
        "running": JobState.RUNNING,
        "completed": JobState.COMPLETED,
        "failed": JobState.FAILED,
        "cancelled": JobState.CANCELLED,
    }.get(state, JobState.UNKNOWN)
    message = {
        "cancelling": "Cancellation requested; the job will stop at the next "
                      "checkpoint.",
        "cancelled": "The job was cancelled.",
        "completed": "The job had already completed.",
        "failed": "The job had already failed.",
    }.get(state, f"Job state: {state}")
    return CancelResponse(job_id=job_id, state=mapped, message=message)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
@router.get("/{job_id}/output", response_model=OutputResponse,
            summary="Finished video metadata")
def get_output(job_id: str) -> OutputResponse:
    """
    Where a finished video is, as a URL rather than a path.

    The absolute path on the server is deliberately not published: it is not
    useful to a client and it is a filesystem disclosure. The file itself is
    served by `download_output`, which supports byte ranges so a browser can seek
    without downloading the whole file.
    """
    if not joblib.is_valid_job_id(job_id):
        raise ApiError(404, ErrorCode.INVALID_JOB_ID,
                       f"{job_id!r} is not a valid job id.")
    status = jobstate.build_status(job_id)
    if status is None:
        raise ApiError(404, ErrorCode.JOB_NOT_FOUND,
                       f"No job {job_id!r} on disk.")
    info = status.output
    if not info.available:
        return OutputResponse(job_id=job_id, available=False)
    return OutputResponse(
        job_id=job_id,
        available=True,
        filename=info.filename,
        download_url=info.download_url,
        media_url=info.media_url,
        media_type=info.media_type,
        size_bytes=info.size_bytes,
    )


@router.get("/{job_id}/output/download", summary="Download the finished video")
def download_output(job_id: str) -> FileResponse:
    """
    Stream the finished video, supporting HTTP byte ranges.

    `FileResponse` handles `Range` itself, which is what makes seeking work in a
    browser's video element, and it streams from disk rather than reading the file
    into memory. The media type is taken from the file's extension: the real
    output of a render is a `.webm`, and labelling it `video/mp4` — as an earlier
    version did — is why a browser would refuse to play it.

    The path is re-derived from the manifest and re-checked for containment on
    every request (see `jobstate.output_path`), so a manifest that names a file
    outside the job cannot be used to read it.
    """
    if not joblib.is_valid_job_id(job_id):
        raise ApiError(404, ErrorCode.INVALID_JOB_ID,
                       f"{job_id!r} is not a valid job id.")
    path = jobstate.output_path(job_id)
    if path is None:
        raise ApiError(
            404, ErrorCode.OUTPUT_UNAVAILABLE,
            f"Job {job_id!r} has no finished video available.")
    return FileResponse(
        path,
        media_type=artifactlib.media_type_for(path),
        filename=path.name,
    )


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------
@router.get("/{job_id}/logs", response_model=LogsResponse,
            summary="A job's durable log")
def get_logs(
    job_id: str,
    tail: int = Query(default=joblogs.DEFAULT_TAIL, ge=1,
                      le=joblogs.MAX_TAIL,
                      description="Return the last N lines."),
    offset: Optional[int] = Query(
        default=None, ge=0,
        description="Start at this line (0-based). Use with `limit` to page."),
    limit: Optional[int] = Query(
        default=None, ge=1, le=joblogs.MAX_TAIL,
        description="Maximum lines to return when using `offset`."),
) -> LogsResponse:
    """
    The job's own log file, bounded and filtered.

    This is the durable `jobs/<id>/logs/job.log`, not the in-memory event buffer:
    it survives a restart, which is exactly when a client wants to read it. The
    default is a tail rather than the whole file, and the response says whether
    more lines exist.

    Lines are filtered of secrets and absolute filesystem paths before they are
    returned (see `src.api.redact`). The file on disk keeps its full detail; only
    the response is filtered, so debugging on the server is not degraded.
    """
    if not joblib.is_valid_job_id(job_id):
        raise ApiError(404, ErrorCode.INVALID_JOB_ID,
                       f"{job_id!r} is not a valid job id.")
    if jobstate.build_status(job_id) is None:
        raise ApiError(404, ErrorCode.JOB_NOT_FOUND,
                       f"No job {job_id!r} on disk.")
    result = joblogs.read(job_id, tail=tail, offset=offset, limit=limit)
    if result is None:
        # The job exists but has written no log yet.
        return LogsResponse(job_id=job_id, lines=[], returned_lines=0,
                            total_lines=0, has_more=False)
    return LogsResponse(**result)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
@router.get("/{job_id}/artifacts", response_model=ArtifactListResponse,
            summary="List a job's artifacts")
def list_artifacts(job_id: str) -> ArtifactListResponse:
    """
    The allow-listed artifacts for a job, present or not.

    Deliberately not a directory listing: the job directory contains separation
    stems, per-segment TTS audio, model cache and speaker reference recordings,
    none of which is an API surface. Only the entries in `src.api.artifacts.SPECS`
    are ever described, and an entry that this render did not produce is returned
    with `available: false` so a UI can show it disabled rather than guess.
    """
    if not joblib.is_valid_job_id(job_id):
        raise ApiError(404, ErrorCode.INVALID_JOB_ID,
                       f"{job_id!r} is not a valid job id.")
    if jobstate.build_status(job_id) is None:
        raise ApiError(404, ErrorCode.JOB_NOT_FOUND,
                       f"No job {job_id!r} on disk.")
    descriptors = artifactlib.list_for(job_id)
    return ArtifactListResponse(
        job_id=job_id,
        artifacts=[ArtifactDescriptor(**d) for d in descriptors],
        count=len(descriptors),
        available_count=sum(1 for d in descriptors if d["available"]),
    )


@router.get("/{job_id}/artifacts/{artifact_name}",
            summary="Download one artifact")
def download_artifact(job_id: str, artifact_name: str) -> FileResponse:
    """
    Serve one allow-listed artifact by name.

    `artifact_name` is looked up in the allow-list; it is never joined to a path.
    An unknown name is a 404 and a withheld name (a cache directory, speaker
    recordings) is a 404 with a reason, so the two are distinguishable. The
    resolved file is confirmed to be inside the job root before it is served.
    """
    if not joblib.is_valid_job_id(job_id):
        raise ApiError(404, ErrorCode.INVALID_JOB_ID,
                       f"{job_id!r} is not a valid job id.")
    if jobstate.build_status(job_id) is None:
        raise ApiError(404, ErrorCode.JOB_NOT_FOUND,
                       f"No job {job_id!r} on disk.")
    if artifact_name in artifactlib.WITHHELD:
        raise ApiError(404, ErrorCode.ARTIFACT_UNAVAILABLE,
                       artifactlib.WITHHELD[artifact_name],
                       details={"artifact": artifact_name, "withheld": True})
    path = artifactlib.resolve(job_id, artifact_name)
    if path is None:
        known = artifact_name in artifactlib.SPECS_BY_NAME
        raise ApiError(
            404, ErrorCode.ARTIFACT_UNAVAILABLE,
            (f"Job {job_id!r} has no artifact {artifact_name!r} on disk."
             if known else
             f"{artifact_name!r} is not an artifact this API publishes."),
            details={"artifact": artifact_name, "known": known})
    return FileResponse(
        path,
        media_type=artifactlib.media_type_for(path),
        filename=path.name,
    )


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------
@router.delete("/{job_id}", response_model=DeleteResponse,
               summary="Delete a job")
def delete_job(
    job_id: str,
    level: CleanupLevel = Query(
        default=CleanupLevel.EVERYTHING,
        description="How much to remove. KEEP_ALL is a no-op, KEEP_RESUMABLE "
                    "and FINAL_ONLY mirror the pipeline's own levels."),
) -> DeleteResponse:
    """
    Delete a finished job, or clean its intermediates.

    An active job is refused with 409 rather than cancelled implicitly: cancelling
    a render takes effect at the pipeline's next checkpoint and may take minutes,
    so silently combining "cancel" with "delete" would make a DELETE either block
    or lie about having completed. A client that wants a running job gone cancels
    it first, then deletes it once it is terminal — which is a decision the client
    can make and the server cannot make for it.

    `level` maps to the pipeline's own `Job.clean` semantics:

      * `EVERYTHING` — remove the whole job directory. The job is gone.
      * `KEEP_RESUMABLE` — remove the bulky intermediates that can be regenerated
        (audio, separation, subtitles, lipsync, cache), keep everything a resume
        needs. The job is still resumable and still listed.
      * `FINAL_ONLY` — keep only the final output. The job is no longer resumable.
      * `KEEP_ALL` — remove nothing; a no-op that reports what would be kept.

    Every removal is containment-checked (`Job.clean` calls `assert_within` per
    target), so a job id or a tampered subdirectory cannot cause a delete outside
    the job root.
    """
    if not joblib.is_valid_job_id(job_id):
        raise ApiError(404, ErrorCode.INVALID_JOB_ID,
                       f"{job_id!r} is not a valid job id.")

    mgr = _manager()
    if mgr.owns_active(job_id):
        status = jobstate.build_status(job_id, manager=mgr)
        state = status.state.value if status is not None else "active"
        raise ApiError(
            409, ErrorCode.JOB_ACTIVE,
            (f"Job {job_id!r} is {state} in this process. Cancel it and wait "
             "for it to stop before deleting it."),
            details={"state": state},
        )

    if jobstate.build_status(job_id, manager=mgr) is None:
        raise ApiError(404, ErrorCode.JOB_NOT_FOUND,
                       f"No job {job_id!r} on disk.")

    try:
        job = joblib.Job.load(job_id)
    except joblib.JobError as exc:
        raise ApiError(404, ErrorCode.JOB_NOT_FOUND,
                       f"No job {job_id!r} on disk.") from exc

    if level == CleanupLevel.EVERYTHING:
        return _delete_everything(job, job_id)

    # The core levels. `active=False` is passed explicitly: the manager check
    # above is the API's guard, and the core guard is the second line of defence.
    try:
        removed = job.clean(level=level.value, active=False)
    except joblib.JobError as exc:
        raise ApiError(400, ErrorCode.INVALID_SETTINGS, str(exc)) from exc
    return DeleteResponse(
        job_id=job_id,
        deleted=False,
        level=level,
        removed=list(removed),
        message=(f"Removed {len(removed)} intermediate "
                 f"{'directory' if len(removed) == 1 else 'directories'}; the "
                 "job itself is kept." if removed
                 else "Nothing was removed; this level keeps everything."),
    )


def _delete_everything(job, job_id: str) -> DeleteResponse:
    """
    Remove the whole job directory, safely.

    The containment check comes first: the path about to be removed is confirmed
    to be a directory inside the jobs root and named exactly as the validated job
    id. That is what stops a symlinked job directory, or a bug in id handling,
    from turning a DELETE into an arbitrary `rmtree`.
    """
    root = Path(config.JOBS_DIR).resolve()
    target = job.root.resolve()
    if target.name != job_id or target.parent != root:
        raise ApiError(
            400, ErrorCode.INVALID_JOB_ID,
            "Refusing to delete: the job directory is not where it should be.",
        )
    try:
        shutil.rmtree(target)
    except OSError as exc:
        logger.warning("Could not delete job %s: %s", job_id, exc)
        raise ApiError(
            500, ErrorCode.INTERNAL_ERROR,
            "The job could not be deleted. The failure has been logged.",
        ) from exc
    logger.info("Deleted job %s", job_id)
    return DeleteResponse(
        job_id=job_id,
        deleted=True,
        level=CleanupLevel.EVERYTHING,
        removed=["<job directory>"],
        message="The job and its files were deleted.",
    )


# ---------------------------------------------------------------------------
# Event stream
# ---------------------------------------------------------------------------
def _sse_headers() -> dict:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # Tell nginx not to buffer, otherwise events arrive in bursts and a
        # progress bar jumps instead of moving.
        "X-Accel-Buffering": "no",
    }


@router.get("/{job_id}/events", summary="Server-sent event stream")
async def job_events(job_id: str, http: Request) -> StreamingResponse:
    """
    Follow a job as Server-Sent Events.

    The first frame is a `snapshot` of the durable state, so a client that
    connects (or reconnects) immediately has something to render without a second
    request. Then recent events are replayed from the buffer, and then live ones
    are streamed until the job reaches a terminal state.

    `Last-Event-ID` is honoured: a browser's `EventSource` reconnects
    automatically and sends the id of the last frame it saw, so a dropped
    connection resumes where it left off. If that position has already rotated
    out of the bounded buffer, a `resync` event is sent first and the client is
    expected to re-read the job — being told there is a gap is better than
    silently missing events.

    A job from a previous process has no buffer and no runner. The snapshot is
    still sent (its state is on disk), followed by a `resync` explaining that
    live events are unavailable, and the stream closes. Nothing here ever blocks
    the event loop: the worker publishes through `call_soon_threadsafe`.
    """
    if not joblib.is_valid_job_id(job_id):
        raise HTTPException(status_code=404, detail=f"No job {job_id!r} on disk.")
    status = jobstate.build_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"No job {job_id!r} on disk.")

    mgr = _manager()
    runner = mgr.get(job_id)
    if runner is None:
        return StreamingResponse(
            _snapshot_only(status),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )

    if runner.events.subscriber_count >= settings.MAX_SUBSCRIBERS_PER_JOB:
        raise HTTPException(
            status_code=503,
            detail=("Too many subscribers are following this job; retry "
                    "shortly."),
        )

    last_event_id = _last_event_id(http)
    # Subscribe *before* replaying, so an event published between the replay and
    # the first queue read is not lost. A duplicate is possible and harmless;
    # the client keys on event_id.
    token, queue = runner.events.subscribe()
    return StreamingResponse(
        _stream(job_id, runner, token, queue, last_event_id, status),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


def _last_event_id(http: Request) -> int:
    """
    The client's last seen event id.

    `Last-Event-ID` is the standard header; `last_event_id` is accepted as a
    query parameter too, because `EventSource` cannot set headers and a
    non-browser client may prefer the query string.
    """
    raw = http.headers.get("last-event-id")
    if raw is None:
        raw = http.query_params.get("last_event_id")
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


async def _snapshot_only(status: JobStatusResponse) -> AsyncIterator[str]:
    """The frames for a job this process cannot stream: state, then why."""
    frame = events.Event(
        event_id=0, job_id=status.job_id, type=EventType.SNAPSHOT.value,
        at=_now(), payload=status.model_dump(mode="json"),
    )
    # An id of 0 is not emitted as `id:`, so the client's replay position is not
    # advanced by a frame that is not part of the event sequence.
    yield _frame_without_id(frame)
    resync = events.Event(
        event_id=0, job_id=status.job_id, type=EventType.RESYNC.value,
        at=_now(),
        payload={
            "reason": "no_live_runner",
            "message": ("This job is not running in the current server process, "
                        "so no live events are available. Its state above is read "
                        "from the durable manifest."),
        },
    )
    yield _frame_without_id(resync)


def _now() -> float:
    import time
    return time.time()


def _frame_without_id(event: events.Event) -> str:
    """An SSE frame with no `id:` line, for frames outside the event sequence."""
    import json
    data = json.dumps(event.to_dict(), ensure_ascii=False)
    return f"event: {event.type}\ndata: {data}\n\n"


def _terminal_event(event_type: str) -> bool:
    return event_type in (EventType.RESULT.value, EventType.FAILED.value,
                          EventType.CANCELLED.value)


async def _stream(job_id: str, runner: JobRunner, token: int,
                  queue: asyncio.Queue, last_event_id: int,
                  status: JobStatusResponse) -> AsyncIterator[str]:
    """
    The live stream body.

    Subscribed before this coroutine starts, so replay and live delivery are
    continuous. The subscriber is always unregistered in `finally`, so a
    disconnected client cannot leak a queue into the buffer.
    """
    try:
        yield _frame_without_id(events.Event(
            event_id=0, job_id=job_id, type=EventType.SNAPSHOT.value,
            at=_now(), payload=status.model_dump(mode="json"),
        ))

        replayed, needs_resync = runner.events.replay_after(last_event_id)
        if needs_resync:
            yield _frame_without_id(events.Event(
                event_id=0, job_id=job_id, type=EventType.RESYNC.value,
                at=_now(),
                payload={
                    "reason": "replay_window_exceeded",
                    "message": ("Events between your last position and the oldest "
                                "buffered event were dropped; re-read the job "
                                "for its current state."),
                },
            ))
        terminal_seen = False
        for event in replayed:
            yield event.to_sse()
            if _terminal_event(event.type):
                terminal_seen = True
        if terminal_seen:
            return

        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(), timeout=settings.HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                # Keep the connection alive across a long, silent model stage.
                yield events.heartbeat_frame()
                # If the runner finished while we were waiting, close cleanly
                # rather than waiting for a terminal event that was already sent
                # before this subscriber existed.
                if runner.is_finished:
                    return
                continue
            yield event.to_sse()
            if _terminal_event(event.type):
                return
    except asyncio.CancelledError:
        # The client went away. Normal, not an error.
        raise
    finally:
        runner.events.unsubscribe(token)
