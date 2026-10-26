"""
Process-local job scheduling.

`JobManager` knows one thing: which jobs are running, which are waiting, and how
many may run at once. It is deliberately *not* a persistence layer. The durable
record of a render is the `Job` directory and its `Manifest`, written by the
pipeline; this object holds only the volatile state that cannot survive a
restart and does not need to — a queue position and a worker thread. After a
restart the queue is gone and every job's state comes from its manifest, which
is the correct outcome: a render that was interrupted is resumable from disk,
not from a process that no longer exists.

The concurrency limit exists because a render loads Whisper, Demucs and
optionally a cloning TTS model into one process. Two at once would contend for
the same GPU memory and the same model cache, so the default is one. It is
configurable for a machine with room to spare, but it is not a user-facing
setting: it describes the host, not the request.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Optional

from src.api import settings
from src.api.events import EventType
from src.api.runner import JobRunner

logger = logging.getLogger(__name__)


class JobAlreadyActive(Exception):
    """
    A resume was asked for while this process already owns the job.

    Raised from inside the manager's critical section, so it is a race-safe
    answer rather than a check that a second request could slip past.
    """

    def __init__(self, job_id: str, state: str):
        super().__init__(f"Job {job_id!r} is already active (state {state}).")
        self.job_id = job_id
        self.state = state


class JobManager:
    """Owns the run queue and the runners that are executing."""

    def __init__(self, max_concurrent: Optional[int] = None,
                 max_retained: Optional[int] = None):
        self.max_concurrent = max(1, int(max_concurrent
                                         or settings.MAX_CONCURRENT_JOBS))
        self.max_retained = max(1, int(max_retained
                                       or settings.MAX_RETAINED_RUNNERS))
        self._lock = threading.RLock()
        self._pending: list[JobRunner] = []
        self._active: dict[str, JobRunner] = {}
        # Finished runners are kept briefly so a client that connects just after
        # a render ends still gets the terminal event and a replay buffer, then
        # evicted oldest-first. Eviction drops only the in-memory runner; the
        # job directory and manifest are untouched.
        self._finished: "OrderedDict[str, JobRunner]" = OrderedDict()

    # -- submission --------------------------------------------------------
    def submit(self, runner: JobRunner) -> None:
        """
        Accept a runner and start it if a slot is free.

        The `queued` event is published here, so a subscriber that connects
        before the job starts still sees it.
        """
        with self._lock:
            self._pending.append(runner)
        runner.events.publish(EventType.QUEUED, {
            "message": "Queued for processing.",
            "queue_position": self.queue_position(runner.job_id),
        })
        self._pump()

    def _pump(self) -> None:
        """Start pending jobs while slots remain."""
        starting: list[JobRunner] = []
        with self._lock:
            while self._pending and len(self._active) < self.max_concurrent:
                runner = self._pending.pop(0)
                # A job cancelled while queued is already terminal; skip it
                # rather than occupying a slot.
                if runner.is_finished:
                    continue
                self._active[runner.job_id] = runner
                starting.append(runner)
        for runner in starting:
            runner.start()

    # -- resuming ----------------------------------------------------------
    def submit_resume(self, job_id: str, runner: JobRunner) -> None:
        """
        Accept a resume runner, refusing if the job is already owned.

        The duplicate check and the insertion happen under one hold of the lock,
        which is the whole point: a check that releases the lock before the
        insert lets two simultaneous `POST /resume` requests both pass the check
        and both start a runner, and two runners over one job directory would
        overwrite each other's stage state and race on the same files.

        Returns normally when the job was already owned, with the existing runner
        left untouched and the new one never started — the caller reports the
        conflict, so a rapid double-click is a 409 rather than a second render.
        """
        with self._lock:
            existing = self._active.get(job_id) or self._finished.get(job_id)
            if existing is None:
                for pending in self._pending:
                    if pending.job_id == job_id:
                        existing = pending
                        break
            if existing is not None and not existing.is_finished:
                raise JobAlreadyActive(job_id, existing.status_dict()["state"])
            # A finished runner may still be retained for late subscribers. The
            # resume supersedes it, so drop it rather than leaving two runners
            # for one id where `get()` could return the stale one.
            self._finished.pop(job_id, None)
            self._pending.append(runner)
        runner.events.publish(EventType.QUEUED, {
            "message": "Queued for resume.",
            "run_mode": runner.mode.value,
            "queue_position": self.queue_position(job_id),
        })
        self._pump()

    def owns_active(self, job_id: str) -> bool:
        """Whether a runner for `job_id` is queued or running in this process."""
        with self._lock:
            if job_id in self._active:
                return True
            return any(r.job_id == job_id for r in self._pending)

    # -- completion --------------------------------------------------------
    def on_runner_finished(self, job_id: str) -> None:
        """
        Called by a runner when its worker ends.

        Frees the slot, retains the runner for late subscribers, and starts the
        next queued job. Must not raise: it runs on the worker thread, at the end
        of a render, and an exception here would lose the "start the next job"
        step and stall the queue.
        """
        try:
            with self._lock:
                runner = self._active.pop(job_id, None)
                if runner is not None:
                    self._finished[job_id] = runner
                    self._finished.move_to_end(job_id)
                    while len(self._finished) > self.max_retained:
                        self._finished.popitem(last=False)
            self._pump()
        except Exception:  # noqa: BLE001 - never lose the queue to a bookkeeping bug
            logger.exception("JobManager: error finalising %s", job_id)

    def _forget_pending(self, job_id: str) -> None:
        with self._lock:
            self._pending = [r for r in self._pending if r.job_id != job_id]

    # -- lookup ------------------------------------------------------------
    def get(self, job_id: str) -> Optional[JobRunner]:
        """The runner for `job_id` if it is queued, running or recently finished."""
        with self._lock:
            runner = self._active.get(job_id)
            if runner is not None:
                return runner
            for pending in self._pending:
                if pending.job_id == job_id:
                    return pending
            return self._finished.get(job_id)

    def queue_position(self, job_id: str) -> Optional[int]:
        """1-based position in the pending queue, or None when not waiting."""
        with self._lock:
            for index, runner in enumerate(self._pending):
                if runner.job_id == job_id:
                    return index + 1
            return None

    def is_active(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._active

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def active_ids(self) -> list[str]:
        with self._lock:
            return list(self._active)

    def pending_ids(self) -> list[str]:
        with self._lock:
            return [r.job_id for r in self._pending]

    # -- cancellation ------------------------------------------------------
    def cancel(self, job_id: str) -> Optional[str]:
        """
        Cancel a job if this manager owns it.

        Returns the resulting state, or None when the job is not tracked here
        (it may still exist on disk from a previous process). A queued job is
        removed from the queue and never runs; a running job is asked to stop at
        its next checkpoint.
        """
        runner = self.get(job_id)
        if runner is None:
            return None
        state = runner.request_cancel()
        if not runner.is_finished:
            # Still shutting down at a checkpoint; it will leave `_active` when
            # its worker returns.
            return state
        self._forget_pending(job_id)
        if self.is_active(job_id):
            # A queued job cancelled before starting is already terminal; make
            # sure it does not linger in the active map.
            self.on_runner_finished(job_id)
        return state

    # -- shutdown ----------------------------------------------------------
    def shutdown(self, timeout: float = 5.0) -> None:
        """
        Ask every tracked job to stop and wait briefly.

        Used by tests and by a clean server shutdown. A render in a model call
        cannot be interrupted instantly, so the timeout is a wait, not a
        guarantee; the worker threads are daemons and the process can exit
        without them.
        """
        with self._lock:
            runners = list(self._active.values()) + list(self._pending)
        for runner in runners:
            try:
                runner.request_cancel()
            except Exception:  # noqa: BLE001
                pass
        for runner in runners:
            runner.join(timeout)


# The application-wide manager. FastAPI runs one process per worker, so a
# process-local manager is the right scope: two uvicorn workers would each
# enforce the limit independently, which is why the concurrency limit is
# documented as per-process.
_manager: Optional[JobManager] = None
_manager_lock = threading.Lock()


def get_manager() -> JobManager:
    """The process-wide `JobManager`, created on first use."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager


def reset_manager(manager: Optional[JobManager] = None) -> None:
    """Replace the process-wide manager. For tests that need a clean queue."""
    global _manager
    with _manager_lock:
        _manager = manager
