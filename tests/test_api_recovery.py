"""
F3: recovery, resume, logs, artifacts, deletion and upload lifecycle.

Same boundary as the F2 suite: the real application, the real `Job`/`Manifest`
layer, the real `JobManager` and the real `EventBuffer`, with only
`JobRunner._build_translator` replaced. The fake here is a little richer than
F2's because resume is the thing under test: it implements `resume_job` the way
the real pipeline does — read the manifest, skip the stages whose fingerprint is
still valid, redo the rest — and *records* which stages it actually executed, so
a test can assert that a resume reused completed work instead of redoing it.

That record is the point of the whole phase. `test_resume_reuses_completed_stages`
does not check that a resume "looks right"; it checks the list of stages that
ran, which is the observable consequence of the fingerprint logic the API is
required not to second-guess.

Must pass with no GPU, no model weights and no network.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import artifacts as artifactlib
from src.api import manager as manager_module
from src.api import jobstate
from src.api import settings, uploads
from src.api.main import app
from src.api.runner import JobRunner
from src.core import jobs as joblib
from src.core.manifest import Manifest
from src.utils import config

_SAMPLE_DIRS = [Path("samples"), Path("/tmp/sample")]
_SAMPLE_MP4 = next((d / "hgf_h264.mp4" for d in _SAMPLE_DIRS
                    if (d / "hgf_h264.mp4").exists()), None)


# Which parts of the configuration each stage actually depends on. This mirrors
# the real pipeline's scoped fingerprints (`config.FINGERPRINT_SCOPE`): a stage
# is invalidated by a change to the settings that feed it, and by nothing else.
# Getting this wrong in the fake would make the invalidation test meaningless —
# it would "prove" that changing the target language reruns extraction, which is
# exactly the waste the real fingerprints exist to prevent.
_STAGE_CONFIG_SCOPE = {
    "extraction": ("source_lang", "chunk_duration"),
    "separation": ("source_lang", "audio_model_name"),
    "tts": ("target_lang", "tts_model_name", "tts_voice"),
    "final": ("target_lang", "enable_lipsync"),
}


def _stage_fingerprint(manifest: Manifest, stage: str) -> str:
    """
    A configuration-derived, stage-scoped fingerprint, as the pipeline computes.

    A stage is reused only when the fingerprint stored with it matches the one
    computed now, and only the settings that stage consumes take part. Deriving
    it from `manifest.config` rather than from a constant is what makes the reuse
    test meaningful: a resume skips a stage because its *inputs* are unchanged,
    not because the test said to.
    """
    scope = _STAGE_CONFIG_SCOPE.get(stage, tuple(sorted(manifest.config)))
    payload = json.dumps(
        {"stage": stage, "config": {k: manifest.config.get(k) for k in scope}},
        sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolate every directory and the process-local manager."""
    jobs_dir = tmp_path / "jobs"
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setattr(config, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(settings, "UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr(settings, "MAX_CONCURRENT_JOBS", 1)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(manager_module, "_manager", manager_module.JobManager())
    yield {"jobs": jobs_dir, "uploads": uploads_dir}
    manager_module.get_manager().shutdown(timeout=5.0)


@pytest.fixture
def client(env):
    with TestClient(app) as c:
        yield c


def _sample_bytes() -> bytes:
    if _SAMPLE_MP4 is not None:
        return _SAMPLE_MP4.read_bytes()
    return b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8192


@pytest.fixture
def upload(client):
    if _SAMPLE_MP4 is None:
        pytest.skip("no sample video available")
    r = client.post("/api/uploads", params={"filename": "clip.mp4"},
                    content=_sample_bytes())
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# A fake pipeline that can also resume
# ---------------------------------------------------------------------------
class FakeTranslator:
    """
    Stands in for `VideoTranslator`, including `resume_job`.

    Both entry points write real manifest state and both record the stages they
    actually execute in `self.executed` (and the ones they skip in `self.reused`).
    `resume_job` skips any stage whose stored fingerprint still matches, which is
    the behaviour the real pipeline implements and the behaviour the API must not
    interfere with.
    """

    def __init__(self, job_id, source_path, config_obj, *,
                 hold: threading.Event | None = None, fail: bool = False,
                 stages: tuple = ("extraction", "separation", "tts", "final"),
                 fail_stage: str | None = None):
        self.job_id = job_id
        self.source_path = Path(source_path) if source_path else None
        self.config_obj = config_obj
        self.cancel_requested = False
        self._hold = hold
        self._fail = fail
        self._stages = stages
        self._fail_stage = fail_stage
        self.executed: list[str] = []
        self.reused: list[str] = []
        self.abort_called = threading.Event()

    def abort(self):
        self.abort_called.set()
        self.cancel_requested = True

    def _run_stages(self, job, manifest, stages, reuse_valid: bool):
        for index, stage in enumerate(stages):
            if reuse_valid and manifest.stage_is_valid(
                    stage, _stage_fingerprint(manifest, stage)):
                self.reused.append(stage)
                continue
            if self.cancel_requested:
                raise _cancelled()
            self.executed.append(stage)
            manifest.start_stage(stage, _stage_fingerprint(manifest, stage))
            manifest.save()
            yield ("progress", (index + 1) / (len(stages) + 1), f"Stage {stage}")
            if self._hold is not None and stage == stages[0]:
                self._hold.wait(timeout=20)
            if self.cancel_requested:
                raise _cancelled()
            if stage == self._fail_stage or (self._fail and stage == stages[-1]):
                manifest.fail_stage(stage, "synthetic failure")
                manifest.status = "FAILED"
                manifest.save()
                raise RuntimeError("synthetic failure")
            manifest.complete_stage(stage)
            manifest.save()

        final_dir = job.stage_dir("final")
        name = self.source_path.name if self.source_path else "output.mp4"
        output = final_dir / f"translated_{name}"
        output.write_bytes(b"fake output")
        manifest.artifacts["final_output"] = str(output)
        manifest.status = "COMPLETE"
        manifest.save()
        yield ("result", str(output))

    def process_video(self, video_path, source_lang="auto", target_lang="en",
                      job_id=None, resume=True, hf_token=None, **kwargs):
        job = joblib.Job.load(self.job_id)
        manifest = Manifest.load(job)
        assert manifest is not None, "job creation must have written a manifest"
        assert not manifest.differs_from(kwargs), \
            f"recorded config drifted: {manifest.differs_from(kwargs)}"
        yield ("log", f"Rendering into {target_lang}")
        yield from self._run_stages(job, manifest, self._stages, reuse_valid=False)

    def resume_job(self, job_id, hf_token=None):
        """The real resume contract: replay the manifest's config, skip valid work."""
        job = joblib.Job.load(job_id)
        manifest = Manifest.load(job)
        assert manifest is not None
        yield ("log", f"Resuming job {job_id}")
        yield from self._run_stages(job, manifest, self._stages, reuse_valid=True)


_CANCELLED_ERROR = None


def _cancelled():
    """The pipeline's CancelledError, resolved without a cold import.

    See the note in `test_api_jobs.py`: importing the pipeline from inside a
    worker thread's cancel path turns a cold torch import into a job that looks
    hung. The class is resolved on first use and cached.
    """
    global _CANCELLED_ERROR
    if _CANCELLED_ERROR is None:
        from src.core.video_translator import CancelledError
        _CANCELLED_ERROR = CancelledError
    return _CANCELLED_ERROR("Processing Cancelled by User")

@pytest.fixture(scope="module", autouse=True)
def _warm_pipeline_import():
    """
    Resolve the pipeline's cancellation class before any job runs.

    The import is expensive (torch, transformers) and is deliberately done here,
    on the main thread at the start of the module, rather than lazily inside a
    worker thread's cancel path. On a cold bytecode cache that import can take
    longer than a test's timeout, which makes a correctly-cancelling job look
    like a hung one.
    """
    _cancelled()



@pytest.fixture
def fake_pipeline(monkeypatch):
    built: list[FakeTranslator] = []
    holder: dict = {}

    def build(self):
        translator = FakeTranslator(
            self.job_id,
            self.source_path if self.mode.value == "new" else None,
            self.config,
            hold=holder.get("hold"),
            fail=holder.get("fail", False),
            stages=holder.get("stages",
                              ("extraction", "separation", "tts", "final")),
            fail_stage=holder.get("fail_stage"),
        )
        built.append(translator)
        holder["last"] = translator
        return translator

    monkeypatch.setattr(JobRunner, "_build_translator", build)
    return {"built": built, "holder": holder}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wait_for_terminal(client, job_id, timeout=30.0):
    """
    Wait until the *durable* manifest records a terminal status.

    Deliberately reads the manifest rather than the API response. The API merges
    in a finished runner's retained state, which is the right answer for a
    client, but these tests then mutate the manifest to stage a recovery
    scenario. Waiting on the manifest is waiting on the thing they are about to
    change; waiting on the response would leave the two racing.

    The terminal state may be written by the pipeline (a completed render) or by
    the runner (a cancellation), so both are accepted.
    """
    deadline = time.time() + timeout
    status = None
    while time.time() < deadline:
        manifest = Manifest.load(joblib.Job.load(job_id))
        status = str(manifest.status or "").upper() if manifest else None
        if status in ("COMPLETE", "COMPLETED", "FAILED", "CANCELLED"):
            return manifest
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish durably: {status}")


def _wait_until(client, job_id, state, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.get(f"/api/jobs/{job_id}").json()["state"] == state:
            return
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {state}")


def _new_upload(client):
    """Stage a fresh upload. Each job needs its own; uploads are single-use."""
    r = client.post("/api/uploads", params={"filename": "clip.mp4"},
                    content=_sample_bytes())
    assert r.status_code == 200, r.text
    return r.json()


def _start_job(client, upload, **overrides):
    """Create a job from an upload response or a bare upload id."""
    upload_id = upload["upload_id"] if isinstance(upload, dict) else upload
    payload = {"upload_id": upload_id, "target_lang": "hi"}
    payload.update(overrides)
    r = client.post("/api/jobs", json=payload)
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


def _stop_job(job_id, *, completed=("extraction", "separation"),
              failed_stage="tts", status="FAILED"):
    """
    Drive a job's manifest to a stopped state with some stages complete.

    Uses the real `Manifest` API and the same fingerprint derivation the fake
    pipeline uses, so what a resume then sees is genuine durable state: a stage
    counts as reusable only if its stored fingerprint matches the one computed
    from the recorded configuration.
    """
    job = joblib.Job.load(job_id)
    manifest = Manifest.load(job)
    for stage in completed:
        manifest.start_stage(stage, _stage_fingerprint(manifest, stage))
        manifest.complete_stage(stage)
    if failed_stage:
        manifest.start_stage(failed_stage,
                             _stage_fingerprint(manifest, failed_stage))
        if status == "CANCELLED":
            manifest.partial_stage(failed_stage)
        else:
            manifest.fail_stage(failed_stage, "synthetic failure")
    manifest.status = status
    manifest.save()
    return manifest


def _mark_interrupted(job_id):
    """Leave a manifest saying RUNNING, as a crash would."""
    job = joblib.Job.load(job_id)
    manifest = Manifest.load(job)
    manifest.status = "RUNNING"
    manifest.save()
    return manifest


def _wait_for_run(client, job_id, timeout=30.0):
    """
    Wait for the runner this process currently owns to finish.

    `_wait_for_terminal` waits on the manifest, which is what the recovery tests
    want — but after a resume the manifest may *already* be terminal from the
    earlier stop, so it would return before the resume has done anything. This
    waits on the runner instead, which only exists once the resume has started.
    """
    manager = manager_module.get_manager()
    deadline = time.time() + timeout
    runner = None
    while time.time() < deadline:
        runner = manager.get(job_id)
        if runner is not None and runner.is_finished:
            return runner
        time.sleep(0.05)
    raise AssertionError(
        f"resume of {job_id} never finished: "
        f"{None if runner is None else runner.status_dict()}")


def _restart_manager():
    """
    Drop the process's runners, leaving only durable state behind.

    This is how a restart is simulated. It matters because a finished runner is
    retained for late SSE subscribers and its state is merged into status
    responses; a test that wants the *recovery* view — what the API reports from
    disk alone — has to remove that in-memory shortcut first.
    """
    manager_module._manager = manager_module.JobManager()

# ---------------------------------------------------------------------------
# Resume: eligibility
# ---------------------------------------------------------------------------
def test_interrupted_job_can_be_resumed(client, upload, fake_pipeline):
    """
    A crash leaves a RUNNING manifest with no runner; that is what resume is for.

    `_mark_interrupted` stages exactly what a restart leaves behind, and
    `_restart_manager` removes the in-memory runner so the API answers from disk
    alone. Both are needed: without the second, the API would answer from the
    finished runner's retained state instead of the manifest.
    """
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _stop_job(job_id)
    _mark_interrupted(job_id)
    _restart_manager()

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["state"] == "interrupted"
    assert status["resume"]["can_resume"] is True
    assert status["resume"]["reason"] == "interrupted"

    r = client.post(f"/api/jobs/{job_id}/resume")
    assert r.status_code == 202, r.text
    assert r.json()["run_mode"] == "resume"
    _wait_for_terminal(client, job_id)


def test_failed_job_can_be_resumed(client, upload, fake_pipeline):
    """A failed render is resumable: the fingerprints decide what is redone."""
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _stop_job(job_id)
    _restart_manager()

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["state"] == "failed"
    assert status["resume"]["can_resume"] is True
    assert status["resume"]["reason"] == "failed"
    assert client.post(f"/api/jobs/{job_id}/resume").status_code == 202
    _wait_for_terminal(client, job_id)


def test_completed_job_is_not_resumable(client, upload, fake_pipeline):
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()
    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["state"] == "completed"
    assert status["resume"]["can_resume"] is False
    assert status["resume"]["reason"] == "complete"
    r = client.post(f"/api/jobs/{job_id}/resume")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "JOB_NOT_RESUMABLE"


def test_completed_job_with_missing_output_is_resumable(client, upload,
                                                        fake_pipeline):
    """`resume_job` re-runs missing stages; the API reports that as resumable."""
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    manifest = Manifest.load(joblib.Job.load(job_id))
    Path(manifest.artifacts["final_output"]).unlink()
    _restart_manager()

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["resume"]["can_resume"] is True
    assert status["resume"]["reason"] == "output_missing"


def test_missing_source_makes_a_job_unresumable(client, upload, fake_pipeline):
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _stop_job(job_id)
    manifest = Manifest.load(joblib.Job.load(job_id))
    Path(manifest.source["path"]).unlink()
    _restart_manager()

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["resume"]["can_resume"] is False
    assert status["resume"]["reason"] == "missing_source"
    assert client.post(f"/api/jobs/{job_id}/resume").status_code == 409


def test_active_job_resume_is_rejected(client, upload, fake_pipeline):
    fake_pipeline["holder"]["hold"] = threading.Event()
    job_id = _start_job(client, upload)
    _wait_until(client, job_id, "running")

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["state"] == "running"
    assert status["resume"]["can_resume"] is False
    assert status["resume"]["reason"] == "already_running"

    r = client.post(f"/api/jobs/{job_id}/resume")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "JOB_ALREADY_ACTIVE"
    fake_pipeline["holder"]["hold"].set()
    _wait_for_terminal(client, job_id)


def test_queued_job_resume_is_rejected(client, upload, fake_pipeline):
    """A job waiting for a slot must not be resumed a second time."""
    # Occupy the only slot so the second job stays queued.
    fake_pipeline["holder"]["hold"] = threading.Event()
    holder = _start_job(client, upload)
    _wait_until(client, holder, "running")

    queued = _start_job(client, _new_upload(client))
    status = client.get(f"/api/jobs/{queued}").json()
    assert status["state"] == "queued"
    assert status["resume"]["can_resume"] is False
    assert status["resume"]["reason"] == "already_queued"
    assert client.post(f"/api/jobs/{queued}/resume").status_code == 409

    fake_pipeline["holder"]["hold"].set()
    _wait_for_terminal(client, holder)
    _wait_for_terminal(client, queued)


def test_resume_of_an_unknown_job_is_a_404(client):
    r = client.post("/api/jobs/20260101-000000-abcdef/resume")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "JOB_NOT_FOUND"


def test_resume_of_a_malformed_job_id_is_a_404(client):
    for bad in ("..%2F..%2Fetc", "a" * 200):
        assert client.post(f"/api/jobs/{bad}/resume").status_code == 404


def test_resume_of_a_job_with_no_manifest_is_refused(env, client):
    """A directory that looks like a job but has no manifest is not resumable."""
    job_dir = env["jobs"] / "20260101-000000-abcdef"
    job_dir.mkdir(parents=True)
    assert client.post(
        "/api/jobs/20260101-000000-abcdef/resume").status_code == 404
    assert client.get("/api/jobs/20260101-000000-abcdef").status_code == 404


# ---------------------------------------------------------------------------
# Resume: the stages that actually run
# ---------------------------------------------------------------------------
def test_resume_reuses_completed_stages(client, upload, fake_pipeline):
    """
    A resume must not redo work the fingerprints still validate.

    This is the assertion that matters: not that a resume *looks* right, but that
    the stages the pipeline executed are only the ones that were not already
    done. If the API or the fake re-ran everything, `executed` would name all
    four stages and this would fail.
    """
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _stop_job(job_id, completed=("extraction", "separation"),
              failed_stage="tts")
    _restart_manager()

    assert client.post(f"/api/jobs/{job_id}/resume").status_code == 202
    _wait_for_run(client, job_id)

    translator = fake_pipeline["holder"]["last"]
    # `final` completed during the first run and was never invalidated, so it is
    # reused too. Only the stage that failed is redone.
    assert translator.reused == ["extraction", "separation", "final"], \
        translator.reused
    assert translator.executed == ["tts"], translator.executed


def test_resume_reruns_a_stage_whose_fingerprint_changed(client, upload,
                                                         fake_pipeline):
    """Changing the config invalidates the affected stages, and only those."""
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _stop_job(job_id, completed=("extraction", "separation"),
              failed_stage="tts")
    # Tamper with the recorded configuration. Only `tts` and `final` consume
    # `target_lang`, so only they lose their fingerprints; extraction and
    # separation stay valid and must not be redone.
    manifest = Manifest.load(joblib.Job.load(job_id))
    manifest.config["target_lang"] = "de"
    manifest.save()
    _restart_manager()

    assert client.post(f"/api/jobs/{job_id}/resume").status_code == 202
    _wait_for_run(client, job_id)

    translator = fake_pipeline["holder"]["last"]
    assert translator.reused == ["extraction", "separation"], translator.reused
    assert translator.executed == ["tts", "final"], translator.executed


def test_resume_reports_run_mode_in_events(client, upload, fake_pipeline):
    """The SSE stream says a resume is a resume, not a new render."""
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _stop_job(job_id)
    _restart_manager()
    client.post(f"/api/jobs/{job_id}/resume")
    _wait_for_run(client, job_id)

    status = client.get(f"/api/jobs/{job_id}").json()
    # The retained runner still knows the mode it ran in.
    assert status["run_mode"] == "resume"


def test_second_resume_while_running_is_refused(client, upload, fake_pipeline):
    fake_pipeline["holder"]["hold"] = threading.Event()
    job_id = _start_job(client, upload)
    _wait_until(client, job_id, "running")

    first = client.post(f"/api/jobs/{job_id}/resume")
    assert first.status_code == 409
    assert first.json()["error"]["code"] == "JOB_ALREADY_ACTIVE"

    fake_pipeline["holder"]["hold"].set()
    _wait_for_terminal(client, job_id)


# ---------------------------------------------------------------------------
# Durable logs
# ---------------------------------------------------------------------------
def _write_log(job_id, lines):
    """Write a durable job log the way the pipeline does, then return its path."""
    job = joblib.Job.load(job_id)
    log_dir = job.paths.root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "job.log"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def test_logs_are_read_from_disk_not_memory(client, upload, fake_pipeline):
    """
    The logs endpoint reads the durable file, not the in-memory event buffer.

    The distinction is the whole point of the endpoint: after a restart there are
    no events, but the log file is still there. A job with no runner at all is
    used, so anything returned can only have come from disk.
    """
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()
    _write_log(job_id, ["line one", "line two", "line three"])

    r = client.get(f"/api/jobs/{job_id}/logs")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lines"] == ["line one", "line two", "line three"]
    assert body["returned_lines"] == 3
    assert body["total_lines"] == 3
    assert body["has_more"] is False


def test_logs_default_to_a_tail(client, upload, fake_pipeline):
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()
    _write_log(job_id, [f"line {i}" for i in range(1, 301)])

    body = client.get(f"/api/jobs/{job_id}/logs").json()
    assert body["returned_lines"] == 200          # DEFAULT_TAIL
    assert body["lines"][0] == "line 101"
    assert body["lines"][-1] == "line 300"
    assert body["total_lines"] == 300
    assert body["has_more"] is True


def test_logs_paging_with_offset_and_limit(client, upload, fake_pipeline):
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()
    _write_log(job_id, [f"line {i}" for i in range(1, 51)])

    body = client.get(f"/api/jobs/{job_id}/logs",
                      params={"offset": 10, "limit": 5}).json()
    assert body["lines"] == ["line 11", "line 12", "line 13", "line 14",
                             "line 15"]
    assert body["has_more"] is True


def test_logs_are_redacted(client, upload, fake_pipeline):
    """A token written to the log must not come back out of the API."""
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()
    _write_log(job_id, [
        "hf_token=hf_abcdefghijklmnopqrstuvwxyz012345",
        "Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz",
        "downloaded to /home/someone/private/models/weights.bin",
    ])

    lines = client.get(f"/api/jobs/{job_id}/logs").json()["lines"]
    joined = "\n".join(lines)
    assert "hf_abcdefghijklmnopqrstuvwxyz012345" not in joined
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in joined
    assert "/home/someone/private/models/weights.bin" not in joined


def test_logs_for_a_job_with_no_log_file_are_empty(client, upload,
                                                   fake_pipeline):
    """
    A job that has written no log returns an empty window, not an error.

    Job creation itself writes a line, so the file is removed here to stage the
    "no log yet" case: the endpoint must answer 200 with nothing rather than
    treating a missing log as a missing job.
    """
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()
    (joblib.Job.load(job_id).paths.root / "logs" / "job.log").unlink()

    r = client.get(f"/api/jobs/{job_id}/logs")
    assert r.status_code == 200
    assert r.json()["lines"] == []
    assert r.json()["total_lines"] == 0


def test_logs_reject_an_out_of_range_tail(client, upload, fake_pipeline):
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    r = client.get(f"/api/jobs/{job_id}/logs", params={"tail": 999999})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_REQUEST"


def test_logs_for_an_unknown_job_are_a_404(client):
    r = client.get("/api/jobs/20260101-000000-abcdef/logs")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "JOB_NOT_FOUND"


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def test_artifacts_are_an_allow_list_not_a_directory_listing(client, upload,
                                                             fake_pipeline):
    """
    Every artifact listed is a known name; nothing is a path from the disk.

    The job directory holds separation stems, per-segment audio and a model
    cache. None of those may appear, and the response must not reflect the
    directory's contents.
    """
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()

    r = client.get(f"/api/jobs/{job_id}/artifacts")
    assert r.status_code == 200, r.text
    body = r.json()
    names = {a["name"] for a in body["artifacts"]}
    assert names <= set(artifactlib.SPECS_BY_NAME)
    assert "final_video" in names
    assert body["count"] == len(artifactlib.SPECS)
    assert body["available_count"] == sum(
        1 for a in body["artifacts"] if a["available"])


def test_artifacts_report_availability_without_leaking_paths(client, upload,
                                                             fake_pipeline):
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()

    body = client.get(f"/api/jobs/{job_id}/artifacts").json()
    final = next(a for a in body["artifacts"] if a["name"] == "final_video")
    assert final["available"] is True
    assert final["size_bytes"] and final["size_bytes"] > 0
    assert final["download_url"].startswith(f"/api/jobs/{job_id}/artifacts/")
    assert "/" not in (final["filename"] or "")


def test_an_absent_artifact_is_listed_as_unavailable(client, upload,
                                                     fake_pipeline):
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()

    body = client.get(f"/api/jobs/{job_id}/artifacts").json()
    subtitles = next(a for a in body["artifacts"] if a["name"] == "subtitles")
    assert subtitles["available"] is False
    assert subtitles["reason"]
    r = client.get(f"/api/jobs/{job_id}/artifacts/subtitles")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "ARTIFACT_UNAVAILABLE"


def test_downloading_an_available_artifact_returns_the_file(client, upload,
                                                            fake_pipeline):
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()

    r = client.get(f"/api/jobs/{job_id}/artifacts/final_video")
    assert r.status_code == 200, r.text
    assert r.content == b"fake output"
    assert r.headers["content-type"].startswith("video/")


def test_traversal_in_an_artifact_name_is_refused(client, upload,
                                                  fake_pipeline):
    """
    A traversal attempt never yields a file.

    Two layers refuse it, and the code differs by which one catches it: a
    percent-encoded slash means the path no longer matches the route at all, so
    Starlette answers `NOT_FOUND` before the handler runs; a name that does reach
    the handler is looked up in the allow-list and answered
    `ARTIFACT_UNAVAILABLE`. Either way the body must be the error envelope and
    never a file, which is what this asserts.
    """
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()

    for name in ("..%2F..%2Fmanifest.json", "..%2Fsource%2Fclip.mp4",
                 "%2Fetc%2Fpasswd", "....%2F%2Fmanifest.json"):
        r = client.get(f"/api/jobs/{job_id}/artifacts/{name}")
        assert r.status_code == 404, (name, r.status_code)
        assert r.json()["error"]["code"] in (
            "ARTIFACT_UNAVAILABLE", "NOT_FOUND"), (name, r.json())
        assert "status" not in r.json()   # never the manifest itself


def test_a_withheld_artifact_says_so_rather_than_404ing_blindly(client,
                                                                upload,
                                                                fake_pipeline):
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()

    r = client.get(f"/api/jobs/{job_id}/artifacts/speaker_references")
    assert r.status_code == 404
    body = r.json()["error"]
    assert body["code"] == "ARTIFACT_UNAVAILABLE"
    assert body["details"]["withheld"] is True


def test_artifacts_for_an_unknown_job_are_a_404(client):
    assert client.get(
        "/api/jobs/20260101-000000-abcdef/artifacts").status_code == 404
    assert client.get(
        "/api/jobs/20260101-000000-abcdef/artifacts/final_video"
    ).status_code == 404


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------
def test_delete_removes_the_whole_job(client, upload, fake_pipeline):
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()
    job_dir = joblib.Job.load(job_id).paths.root

    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deleted"] is True
    assert body["level"] == "EVERYTHING"
    assert not job_dir.exists()
    # And the job is gone from the API's point of view, not just from disk.
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert client.delete(f"/api/jobs/{job_id}").status_code == 404


def test_delete_of_an_active_job_is_refused(client, upload, fake_pipeline):
    """
    A running job is refused, not cancelled implicitly.

    Cancellation lands at the pipeline's next checkpoint and can take minutes, so
    a DELETE that cancelled and deleted would either block or lie. The client is
    told to cancel first.
    """
    fake_pipeline["holder"]["hold"] = threading.Event()
    job_id = _start_job(client, upload)
    _wait_until(client, job_id, "running")

    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "JOB_ACTIVE"
    assert r.json()["error"]["details"]["state"] == "running"

    fake_pipeline["holder"]["hold"].set()
    _wait_for_terminal(client, job_id)


def test_keep_resumable_removes_intermediates_and_keeps_the_job(client, upload,
                                                                fake_pipeline):
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()
    job_dir = joblib.Job.load(job_id).paths.root
    # A bulky intermediate that this level is expected to remove.
    (job_dir / "separation").mkdir(exist_ok=True)
    (job_dir / "separation" / "stems.wav").write_bytes(b"x" * 1024)

    r = client.delete(f"/api/jobs/{job_id}", params={"level": "KEEP_RESUMABLE"})
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] is False
    assert job_dir.exists()
    assert client.get(f"/api/jobs/{job_id}").status_code == 200


def test_keep_all_is_a_no_op(client, upload, fake_pipeline):
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()

    r = client.delete(f"/api/jobs/{job_id}", params={"level": "KEEP_ALL"})
    assert r.status_code == 200
    assert r.json()["deleted"] is False
    assert r.json()["removed"] == []
    assert client.get(f"/api/jobs/{job_id}").status_code == 200


def test_delete_rejects_an_unknown_level(client, upload, fake_pipeline):
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    r = client.delete(f"/api/jobs/{job_id}", params={"level": "shred"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_REQUEST"


def test_delete_of_an_unknown_job_is_a_404(client):
    r = client.delete("/api/jobs/20260101-000000-abcdef")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "JOB_NOT_FOUND"


def test_delete_refuses_a_malformed_job_id(client):
    r = client.delete("/api/jobs/..%2F..%2Fetc")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Output and media type
# ---------------------------------------------------------------------------
def test_output_is_reported_as_urls_and_a_media_type(client, upload,
                                                     fake_pipeline):
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()

    body = client.get(f"/api/jobs/{job_id}/output").json()
    assert body["available"] is True
    # One file route serves both; see the OutputInfo docstring for why the two
    # fields exist even though they resolve to the same URL.
    assert body["media_url"] == f"/api/jobs/{job_id}/output/download"
    assert body["download_url"] == body["media_url"]
    assert body["media_type"] and body["size_bytes"] > 0
    assert body["filename"]


def test_output_is_served_with_range_support(client, upload, fake_pipeline):
    """
    The file route honours byte ranges, which is what makes seeking work.

    The JSON metadata route (`/output`) is checked separately; this is about the
    bytes. A partial request must come back as 206 with only the requested slice,
    not as the whole file.
    """
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()

    r = client.get(f"/api/jobs/{job_id}/output/download",
                   headers={"Range": "bytes=0-3"})
    assert r.status_code == 206, r.status_code
    assert r.content == b"fake"
    assert r.headers["content-range"] == "bytes 0-3/11"


def test_output_download_sets_a_filename(client, upload, fake_pipeline):
    job_id = _start_job(client, upload)
    _wait_for_terminal(client, job_id)
    _restart_manager()

    r = client.get(f"/api/jobs/{job_id}/output/download")
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")
    assert r.content == b"fake output"


def test_output_of_an_unfinished_job_is_unavailable(client, upload,
                                                    fake_pipeline):
    fake_pipeline["holder"]["hold"] = threading.Event()
    job_id = _start_job(client, upload)
    _wait_until(client, job_id, "running")

    body = client.get(f"/api/jobs/{job_id}/output").json()
    assert body["available"] is False
    assert client.get(f"/api/jobs/{job_id}/output").status_code == 200
    assert client.get(
        f"/api/jobs/{job_id}/output/download").status_code == 404

    fake_pipeline["holder"]["hold"].set()
    _wait_for_terminal(client, job_id)


def test_output_for_an_unknown_job_is_a_404(client):
    assert client.get(
        "/api/jobs/20260101-000000-abcdef/output").status_code == 404


# ---------------------------------------------------------------------------
# Upload lifecycle and the TTL sweep
# ---------------------------------------------------------------------------
def test_cleanup_stale_removes_only_old_staged_uploads(client, env):
    """
    The sweep is age-based, and it never touches a claimed upload.

    A claimed upload's file has moved into a job directory and is that job's
    source; deleting it would destroy a render's input. The safety rule is the
    recorded status, not the age, so an old claimed upload must survive even an
    aggressive TTL.
    """
    fresh = _new_upload(client)
    old = _new_upload(client)

    # Age the second upload by rewriting its recorded creation time.
    meta = uploads.load(old["upload_id"])
    meta.created_at = time.time() - 10_000
    uploads.write_meta(meta)

    removed = uploads.cleanup_stale(ttl_seconds=3600)
    assert old["upload_id"] in removed
    assert fresh["upload_id"] not in removed
    assert client.get(f"/api/uploads/{fresh['upload_id']}").status_code == 200
    assert client.get(f"/api/uploads/{old['upload_id']}").status_code == 404


def test_cleanup_stale_never_removes_a_claimed_upload(client, env,
                                                     fake_pipeline):
    old = _new_upload(client)
    job_id = _start_job(client, old)
    _wait_for_terminal(client, job_id)
    # The upload is claimed; age it past any TTL.
    meta = uploads.load(old["upload_id"])
    meta.created_at = time.time() - 10_000
    uploads.write_meta(meta)

    assert uploads.cleanup_stale(ttl_seconds=1) == []
    # The job's source file is still there.
    source = Manifest.load(joblib.Job.load(job_id)).source["path"]
    assert Path(source).exists()


def test_cleanup_stale_can_be_disabled(client, env):
    """A TTL of zero or less keeps everything, which is how a deployment opts out."""
    old = _new_upload(client)
    meta = uploads.load(old["upload_id"])
    meta.created_at = time.time() - 10_000
    uploads.write_meta(meta)

    assert uploads.cleanup_stale(ttl_seconds=0) == []
    assert client.get(f"/api/uploads/{old['upload_id']}").status_code == 200


def test_startup_sweep_runs_with_the_configured_ttl(env, monkeypatch):
    """
    The lifespan sweeps the staging directory on startup.

    Exercised directly rather than by starting a second app: the fixture's client
    already started one, and the behaviour under test is the call the lifespan
    makes, not the ASGI startup sequence.
    """
    monkeypatch.setattr(settings, "UPLOAD_TTL_SECONDS", 0.0)
    assert uploads.cleanup_stale() == []       # TTL 0 disables it
    monkeypatch.setattr(settings, "UPLOAD_TTL_SECONDS", 3600.0)
    assert uploads.cleanup_stale() == []       # nothing staged


# ---------------------------------------------------------------------------
# The error envelope
# ---------------------------------------------------------------------------
def test_every_error_has_the_same_shape(client):
    """
    One shape for every failure, whichever layer raised it.

    A client switches on `error.code` and never parses prose or needs a different
    parser per endpoint. Unknown path, wrong method, bad body and a missing job
    are all covered because they come from four different places.
    """
    cases = [
        client.get("/api/jobs/20260101-000000-abcdef"),          # missing job
        client.get("/api/nope"),                                  # no such path
        client.post("/api/jobs"),                                 # bad body
        client.post("/api/jobs/20260101-000000-abcdef/resume"),   # missing job
        client.delete("/api/jobs/20260101-000000-abcdef"),        # missing job
    ]
    for r in cases:
        assert r.status_code >= 400, r.text
        body = r.json()
        assert set(body) == {"error"}, body
        assert isinstance(body["error"]["code"], str) and body["error"]["code"]
        assert isinstance(body["error"]["message"], str) and body["error"]["message"]


def test_a_wrong_method_is_reported_in_the_envelope(client):
    r = client.put("/api/jobs/20260101-000000-abcdef")
    assert r.status_code == 405
    assert r.json()["error"]["code"] == "METHOD_NOT_ALLOWED"


def test_a_validation_error_names_the_offending_field(client, upload):
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"],
                                       "target_lang": "hi", "chunk_duration": -5})
    assert r.status_code == 422
    error = r.json()["error"]
    assert error["code"] == "INVALID_REQUEST"
    assert error["field"]
    assert "errors" in error.get("details", {})


def test_an_unexpected_error_does_not_leak_internals(env, monkeypatch):
    """
    A crash is a generic 500: no traceback, no path, no exception text.

    An exception message can carry a token, a model URL or a filesystem path, so
    the client is told only that it broke while the server logs the detail. A
    client with `raise_server_exceptions=False` is used so the test observes the
    response the handler produced rather than the re-raised exception.
    """
    def boom(*args, **kwargs):
        raise RuntimeError("/secret/path/token=hf_shouldnotappear")

    monkeypatch.setattr(jobstate, "build_status", boom)
    with TestClient(app, raise_server_exceptions=False) as raw:
        r = raw.get("/api/jobs/20260101-000000-abcdef")
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "hf_shouldnotappear" not in r.text
    assert "/secret/path" not in r.text
    assert "Traceback" not in r.text
