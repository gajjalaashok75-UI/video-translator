"""
Resume eligibility: the one place that decides whether a job can be resumed.

The decision is made here and nowhere else, and every response that mentions
resume (`GET /api/jobs`, `GET /api/jobs/{id}`) carries it, so a client never has
to reconstruct the rules from a job's state. That matters because the rules are
not obvious — `completed` is normally final but is resumable when its output is
missing, and `interrupted` looks like a failure but is the *expected* state after
a restart and is exactly what resume is for.

The rules follow the pipeline's own semantics rather than inventing new ones.
`VideoTranslator.resume_job` already decides what it will and will not continue:

  * a `COMPLETE` job whose output still exists is a no-op — it re-yields the
    existing file and runs nothing;
  * a `COMPLETE` job whose output is *gone* re-runs the missing stages, and the
    per-stage fingerprints make every surviving stage a no-op;
  * anything else (RUNNING, FAILED, CANCELLED) continues, skipping the stages
    whose fingerprints are still valid.

So "resumable" means "`resume_job` would make progress", and the only things
this module refuses are the states where it would either do nothing (a complete
job with its output) or corrupt an in-flight render (a job this process is
already running).

The API deliberately does not try to be cleverer than that. It does not inspect
which stage failed, and it does not decide that a particular failure is
unrecoverable: the fingerprints on disk are the authority on what still needs
doing, and second-guessing them is how a resume silently redoes hours of work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.api.manager import JobManager, get_manager
from src.api.schemas import JobState
from src.core import jobs as joblib
from src.core.manifest import Manifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResumeDecision:
    """
    Whether a job can be resumed, and why.

    `reason` is a short machine-readable token (`"interrupted"`, `"already_running"`,
    `"missing_source"`) so a client can branch on it; `message` is the sentence to
    show a person. Both are always populated, including when `can_resume` is true,
    so the UI can explain *why* the button is enabled.
    """

    can_resume: bool
    reason: str
    message: str


def _source_is_present(manifest: Manifest) -> bool:
    """
    Whether the job's source video still exists.

    `resume_job` refuses a job whose source is gone — the pipeline cannot
    re-derive the parts of the render that never happened without it — so the
    status endpoint reports the same thing rather than offering a Resume button
    that would fail.
    """
    source = (manifest.source or {}).get("path")
    if not source:
        return False
    return Path(str(source)).exists()


def evaluate(job_id: str, manager: Optional[JobManager] = None,
             manifest: Optional[Manifest] = None,
             state: Optional[JobState] = None) -> Optional[ResumeDecision]:
    """
    The resume decision for a job, or None when no such job exists.

    `manifest` and `state` may be supplied by a caller that has already read
    them (the status builder has), so a status response does not read the
    manifest twice.
    """
    manager = manager or get_manager()

    if manifest is None:
        if not joblib.is_valid_job_id(job_id):
            return None
        try:
            job = joblib.Job.load(job_id)
        except joblib.JobError:
            return None
        manifest = Manifest.load(job)
        if manifest is None:
            return None

    # A job this process is running must not be resumed: a second runner would
    # build a second pipeline over the same job directory, and the two would
    # overwrite each other's stage state and fight over the same files.
    runner = manager.get(job_id)
    if runner is not None and not runner.is_finished:
        runner_state = runner.status_dict()["state"]
        if runner_state == "queued":
            return ResumeDecision(
                False, "already_queued",
                "This job is already queued in this process.")
        return ResumeDecision(
            False, "already_running",
            "This job is already running in this process.")

    if state is None:
        state = _state_from_manifest(manifest)

    if state == JobState.RUNNING:
        # The manifest says RUNNING but no runner owns it, which the read model
        # would normally have already turned into `interrupted`. Reaching here
        # means the caller passed a state directly; treat it as interrupted,
        # which is what it actually is.
        state = JobState.INTERRUPTED

    if not _source_is_present(manifest):
        return ResumeDecision(
            False, "missing_source",
            "The source video for this job is no longer on disk, so the render "
            "cannot be continued.")

    if state == JobState.COMPLETED:
        # Only resumable when the output is gone: `resume_job` re-runs the
        # missing stages, and the fingerprints skip everything that survives.
        if _output_present(manifest):
            return ResumeDecision(
                False, "complete",
                "This job is already complete.")
        return ResumeDecision(
            True, "output_missing",
            "This job is marked complete but its video is missing; resuming "
            "will regenerate it from the stages that survive.")

    if state in (JobState.FAILED, JobState.CANCELLED, JobState.INTERRUPTED):
        # All three are "stopped, not broken". The fingerprints decide what is
        # redone, which is the pipeline's call, not the API's.
        reason = {
            JobState.FAILED: "failed",
            JobState.CANCELLED: "cancelled",
            JobState.INTERRUPTED: "interrupted",
        }[state]
        return ResumeDecision(
            True, reason,
            "This job can be resumed; completed stages will be reused.")

    if state == JobState.QUEUED:
        return ResumeDecision(
            False, "queued", "This job is already queued in this process.")

    return ResumeDecision(
        False, "unknown_state",
        "This job's state is not recognised, so it cannot be resumed safely.")


def _state_from_manifest(manifest: Manifest) -> JobState:
    """Manifest status -> the state a resume decision is based on."""
    status = str(manifest.status or "").upper()
    if status == "COMPLETE":
        return JobState.COMPLETED
    if status == "FAILED":
        return JobState.FAILED
    if status == "CANCELLED":
        return JobState.CANCELLED
    if status in ("RUNNING", "PENDING"):
        return JobState.INTERRUPTED
    return JobState.UNKNOWN


def _output_present(manifest: Manifest) -> bool:
    final = (manifest.artifacts or {}).get("final_output")
    return bool(final) and Path(str(final)).exists()
