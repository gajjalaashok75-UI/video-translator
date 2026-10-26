"""
The read model: a job's durable state and live state merged into one response.

Two sources of truth are combined here, and keeping them straight is the whole
job of this module.

  * **Durable** — the `Job` directory and its `Manifest`, written by the
    pipeline. This survives a restart, so it is what a status request reads.
  * **Live** — the `JobRunner` in this process, if there is one. Queue position,
    the current fraction and the cancel state live here and are meaningless
    after a restart.

A job with a durable record but no live runner is the normal case for anything
finished before this process started, and is reported from the manifest alone.
A job with no manifest at all is a 404. A job whose manifest says `RUNNING` but
which has no runner is reported `interrupted`: the process that was running it is
gone, and saying `running` would be a lie the client would wait on forever.

Nothing here writes. Reads go through `Manifest.load`, which is the same code the
pipeline resumes from, so the status endpoint cannot disagree with the pipeline
about what has been done.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.api import artifacts, recovery, summary
from src.api.manager import JobManager, get_manager
from src.api.schemas import (
    JobState,
    JobStatusResponse,
    OutputInfo,
    ResumeInfo,
    RunMode,
    SegmentCounts,
    StageStatus,
)
from src.core import jobs as joblib
from src.core.manifest import Manifest
from src.utils import config

logger = logging.getLogger(__name__)

# Manifest status -> API state, for jobs with no live runner. `RUNNING` is
# handled separately because it means "interrupted" once the process that wrote
# it is gone.
_TERMINAL_BY_MANIFEST = {
    "COMPLETE": JobState.COMPLETED,
    "FAILED": JobState.FAILED,
    "CANCELLED": JobState.CANCELLED,
}

# Coarse fraction for a stage, used when there is no live progress to report.
# These mirror the `yield ("progress", ...)` values the pipeline emits, so a
# client that reconnects sees roughly the same bar it saw before disconnecting.
_STAGE_FRACTIONS = {
    "extraction": 0.10,
    "separation": 0.20,
    "transcription": 0.30,
    "diarization": 0.30,
    "references": 0.35,
    "translation": 0.40,
    "tts": 0.50,
    "sync": 0.70,
    "subtitles": 0.75,
    "lipsync": 0.90,
    "final": 0.95,
}


def _manifest_state(manifest: Manifest) -> JobState:
    status = str(manifest.status or "").upper()
    if status in _TERMINAL_BY_MANIFEST:
        return _TERMINAL_BY_MANIFEST[status]
    if status in ("RUNNING", "PENDING"):
        # The manifest is written by a process; if no runner in *this* process
        # owns the job, that process is gone and the render is not progressing.
        return JobState.INTERRUPTED
    return JobState.UNKNOWN


def _stage_fraction(manifest: Manifest) -> float:
    """
    The furthest stage that has started, as a coarse fraction.

    Used only when there is no live progress to report — a job this process is
    not running, typically one interrupted by a restart. It never overstates: a
    stage's value is credited only once that stage has actually begun.
    """
    fraction = 0.0
    for name in config.JOB_STAGES:
        record = manifest.stage(name)
        if record.status in ("RUNNING", "COMPLETE", "PARTIAL"):
            fraction = max(fraction, _STAGE_FRACTIONS.get(name, 0.0))
    return fraction


def _stages(manifest: Manifest) -> list[StageStatus]:
    out = []
    # Iterated in the pipeline's own stage order rather than the manifest's dict
    # order, so the response reads top-to-bottom in the order stages run.
    for name in config.JOB_STAGES:
        record = manifest.stage(name)
        out.append(StageStatus(
            name=name,
            status=record.status,
            error=record.error,
            attempts=int(getattr(record, "attempts", 0) or 0),
        ))
    return out


def _output_available(manifest: Manifest) -> bool:
    final = (manifest.artifacts or {}).get("final_output")
    return bool(final) and Path(str(final)).exists()


def _run_mode(runner) -> Optional[RunMode]:
    if runner is None:
        return None
    try:
        return RunMode(runner.status_dict().get("run_mode") or "new")
    except ValueError:
        return None


def output_info(job_id: str, manifest: Manifest) -> OutputInfo:
    """
    The output descriptor, built from the manifest and a real stat.

    The URL is returned whether or not the file is there — the client needs to
    know the address to use once the render finishes — but `available` is honest
    about whether it exists right now, and the URLs are omitted when there is
    nothing to serve.
    """
    final = (manifest.artifacts or {}).get("final_output")
    if not final:
        return OutputInfo()
    path = Path(str(final))
    if not path.is_file():
        return OutputInfo()
    # Containment: a manifest is a file on disk and is not a trust boundary, so
    # the recorded path is confirmed to be inside the job before it is offered.
    try:
        path.resolve().relative_to(joblib.Job.load(job_id).root.resolve())
    except (ValueError, OSError, joblib.JobError):
        logger.warning("Job %s: recorded output is outside the job; hiding it",
                       job_id)
        return OutputInfo()
    try:
        size = path.stat().st_size
    except OSError:
        return OutputInfo()
    url = f"/api/jobs/{job_id}/output/download"
    return OutputInfo(
        available=True,
        filename=path.name,
        size_bytes=size,
        media_type=artifacts.media_type_for(path),
        media_url=url,
        download_url=url,
    )


def build_status(job_id: str,
                 manager: Optional[JobManager] = None) -> Optional[JobStatusResponse]:
    """
    A job's status, or None when there is no job with that id on disk.

    Reads the manifest from disk every time. That is a small JSON file and the
    call is not on a hot path, and reading it means the response always agrees
    with what a resume would see.
    """
    if not joblib.is_valid_job_id(job_id):
        return None
    try:
        job = joblib.Job.load(job_id)
    except joblib.JobError:
        return None
    manifest = Manifest.load(job)
    if manifest is None:
        return None

    manager = manager or get_manager()
    runner = manager.get(job_id)

    state = _manifest_state(manifest)
    live_progress = None
    live_description = None
    queue_position = None
    started_at = None
    finished_at = None
    error = None
    fraction = _stage_fraction(manifest)

    if runner is not None:
        live = runner.status_dict()
        runner_state = live["state"]
        state = {
            "queued": JobState.QUEUED,
            "running": JobState.RUNNING,
            "completed": JobState.COMPLETED,
            "failed": JobState.FAILED,
            "cancelled": JobState.CANCELLED,
        }.get(runner_state, state)
        live_progress = live["live_progress"]
        live_description = live["live_description"]
        started_at = live["started_at"]
        finished_at = live["finished_at"]
        if state == JobState.QUEUED:
            queue_position = manager.queue_position(job_id)
        if runner_state == "failed":
            error = runner.error
    else:
        queue_position = manager.queue_position(job_id)

    if live_progress is not None:
        fraction = live_progress
    if state == JobState.COMPLETED:
        fraction = 1.0

    if error is None and manifest.errors:
        # Surface the most recent durable error summary, but only its message:
        # a manifest error record is already truncated by the manifest, and it is
        # the pipeline's own text, not a traceback.
        last = manifest.errors[-1]
        if isinstance(last, dict) and last.get("error"):
            error = str(last["error"])

    # The resume decision is made here, once, from the state this builder has
    # already computed, so every response that mentions resume agrees.
    decision = recovery.evaluate(job_id, manager=manager, manifest=manifest,
                                 state=state)
    if decision is None:
        # Should be impossible: the manifest was just read.
        decision_info = ResumeInfo(
            can_resume=False, reason="unknown_state",
            message="This job's state could not be determined.")
    else:
        decision_info = ResumeInfo(can_resume=decision.can_resume,
                                   reason=decision.reason,
                                   message=decision.message)

    source = dict(manifest.source or {})
    cfg = dict(manifest.config or {})
    return JobStatusResponse(
        job_id=job_id,
        state=state,
        status=str(manifest.status or "UNKNOWN"),
        run_mode=_run_mode(runner),
        source=source,
        source_name=source.get("name"),
        source_language=source.get("language") or cfg.get("source_lang"),
        target_language=(source.get("target_language")
                         or cfg.get("target_lang")),
        quality_preset=cfg.get("quality_preset"),
        config=cfg,
        stages=_stages(manifest),
        segments=SegmentCounts(**manifest.segment_counts()),
        progress_fraction=round(min(1.0, max(0.0, fraction)), 4),
        live_progress=live_progress,
        live_description=live_description,
        queue_position=queue_position,
        created_at=getattr(manifest, "created_at", None),
        updated_at=getattr(manifest, "updated_at", None),
        started_at=started_at,
        finished_at=finished_at,
        output_available=_output_available(manifest),
        output=output_info(job_id, manifest),
        summary=summary.build_summary(manifest, state.value),
        resume=decision_info,
        error=error,
        events_url=f"/api/jobs/{job_id}/events",
        logs_url=f"/api/jobs/{job_id}/logs",
        artifacts_url=f"/api/jobs/{job_id}/artifacts",
        errors=[e for e in (manifest.errors or []) if isinstance(e, dict)],
    )


def build_statuses(job_ids: list[str],
                   manager: Optional[JobManager] = None) -> list[JobStatusResponse]:
    """Build statuses for several ids, skipping any that are unreadable."""
    manager = manager or get_manager()
    out = []
    for job_id in job_ids:
        try:
            status = build_status(job_id, manager=manager)
        except Exception:  # noqa: BLE001 - one bad job must not fail the list
            logger.warning("Could not read status for job %s", job_id,
                           exc_info=True)
            continue
        if status is not None:
            out.append(status)
    return out


def manifest_for(job_id: str) -> Optional[Manifest]:
    """
    The manifest for a job, or None when the job or its manifest is unreadable.

    Exposed so a route that needs the manifest (resume, for the source path) reads
    it through the same validated path the status builder uses, rather than
    loading a `Job` itself and repeating the validation.
    """
    if not joblib.is_valid_job_id(job_id):
        return None
    try:
        job = joblib.Job.load(job_id)
    except joblib.JobError:
        return None
    return Manifest.load(job)


def output_path(job_id: str) -> Optional[Path]:
    """
    The finished video for a job, if it exists and is contained by the job.

    The path is read from the manifest, which the pipeline wrote, but it is
    re-checked against the job directory before it is served: a manifest is a
    file on disk, and a file on disk is not a trust boundary. Anything that
    resolves outside the job is treated as absent.
    """
    if not joblib.is_valid_job_id(job_id):
        return None
    try:
        job = joblib.Job.load(job_id)
    except joblib.JobError:
        return None
    manifest = Manifest.load(job)
    if manifest is None:
        return None
    final = (manifest.artifacts or {}).get("final_output")
    if not final:
        return None
    path = Path(str(final))
    if not path.is_file():
        return None
    try:
        path.resolve().relative_to(job.root.resolve())
    except ValueError:
        # The recorded output lives outside the job. Refuse rather than serve
        # an arbitrary file the manifest happens to name.
        logger.warning("Job %s: recorded output %s is outside the job; ignoring",
                       job_id, path)
        return None
    return path
