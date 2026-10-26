"""
F2: uploads, job execution, cancellation and the event stream.

These tests run the real FastAPI application through `TestClient`, the real
`Job`/`Manifest` durable layer, the real `JobManager` and the real `EventBuffer`.
The one thing they substitute is the *pipeline itself*: `JobRunner._build_translator`
returns a fake translator that yields the same event tuples the real generator
does and writes real durable state through the real `Manifest` API.

That boundary is chosen deliberately. Everything under test in F2 is the layer
*around* the pipeline — scheduling, isolation, event mapping, cancellation,
durability — and every one of those paths is exercised for real. Running the
genuine pipeline would require downloading Whisper and Demucs weights and minutes
of CPU time per test, which is neither available offline nor a unit test. The
fake is a stand-in for "some component that yields log/progress/result tuples",
which is the contract the pipeline publishes and the only part F2 depends on.
`test_pipeline_contract` below pins that contract against the real signature, so
the fake cannot drift from the thing it replaces.

The suite must pass with no GPU, no model weights and no network.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import manager as manager_module
from src.api import settings, uploads
from src.api.main import app
from src.api.runner import JobConfig, JobRunner
from src.core import jobs as joblib
from src.core.manifest import Manifest
from src.utils import config

# A real, small H.264 MP4 if one is available; otherwise a synthetic file that
# ffprobe can still read. Upload tests are skipped only when neither exists.
_SAMPLE_DIRS = [Path("samples"), Path("/tmp/sample")]
_SAMPLE_MP4 = next((d / "hgf_h264.mp4" for d in _SAMPLE_DIRS
                    if (d / "hgf_h264.mp4").exists()), None)


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
    # Do not leave worker threads running into the next test.
    manager_module.get_manager().shutdown(timeout=5.0)


@pytest.fixture
def client(env):
    with TestClient(app) as c:
        yield c


def _sample_bytes() -> bytes:
    if _SAMPLE_MP4 is not None:
        return _SAMPLE_MP4.read_bytes()
    # Minimal valid-ish MP4 header; only used when no real sample exists.
    return b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8192


@pytest.fixture
def upload(client):
    """Stage one upload and return its response body."""
    if _SAMPLE_MP4 is None:
        pytest.skip("no sample video available")
    r = client.post("/api/uploads", params={"filename": "clip.mp4"},
                    content=_sample_bytes())
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# The pipeline contract the fake stands in for
# ---------------------------------------------------------------------------
def test_pipeline_contract():
    """
    Pin the real pipeline's interface against what F2 assumes.

    If `process_video` stops accepting `job_id`/`resume`, or the set of
    configuration keys it caches changes, F2's job creation would record a
    configuration the pipeline disagrees with and every job would refuse to run
    as "drifted". This test fails loudly at that point instead.
    """
    import inspect

    from src.core.video_translator import VideoTranslator

    params = set(inspect.signature(VideoTranslator.process_video).parameters)
    for required in ("job_id", "resume", "hf_token", "quality_preset",
                     "video_path"):
        assert required in params, f"process_video no longer accepts {required!r}"

    # Every field the API sends must be one the pipeline accepts.
    sent = set(JobConfig().to_pipeline_kwargs())
    assert sent <= params, f"API sends unknown arguments: {sorted(sent - params)}"

    # And the API's recorded run configuration must match the pipeline's own key
    # set exactly, or the drift check in process_video will reject every job.
    from src.api.routes.jobs import _run_config
    all_values = {k: getattr(JobConfig(), k) for k in JobConfig.__dataclass_fields__}
    pipeline_keys = set(VideoTranslator._run_config(all_values))
    assert set(_run_config(JobConfig())) == pipeline_keys


def test_pipeline_is_not_imported_by_the_api():
    """
    F2 must not make `/api/health` load torch.

    Job execution imports the pipeline lazily, inside the worker thread, so the
    API process still starts and answers without the ML stack.
    """
    import subprocess
    import sys
    code = (
        "from fastapi.testclient import TestClient\n"
        "from src.api.main import app\n"
        "c = TestClient(app)\n"
        "assert c.get('/api/health').status_code == 200\n"
        "assert c.get('/api/jobs').status_code == 200\n"
        "import sys\n"
        "assert 'src.core.video_translator' not in sys.modules, 'pipeline imported'\n"
        "print('CLEAN')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parent.parent), timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "CLEAN" in proc.stdout


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------
def test_upload_stages_and_probes_a_real_video(client):
    if _SAMPLE_MP4 is None:
        pytest.skip("no sample video available")
    r = client.post("/api/uploads", params={"filename": "my movie.mp4"},
                    content=_sample_bytes())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["upload_id"].startswith("up-")
    assert body["status"] == "staged"
    assert body["bytes"] > 1024
    # The probe reports real stream detail, which is the point of doing it.
    assert body["media"]["has_video"] is True
    assert body["media"]["duration"] > 0
    assert body["media"]["width"] and body["media"]["height"]
    assert body["claimed_by"] is None


def test_upload_rejects_a_disallowed_extension(client):
    r = client.post("/api/uploads", params={"filename": "notes.txt"},
                    content=b"hello")
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["error"]["message"]


def test_upload_rejects_a_file_with_no_extension(client):
    r = client.post("/api/uploads", params={"filename": "video"},
                    content=b"hello")
    assert r.status_code == 400


def test_upload_rejects_a_non_video_body(client):
    """A text file renamed `.mp4` must be refused, not turned into a job."""
    r = client.post("/api/uploads", params={"filename": "fake.mp4"},
                    content=b"this is definitely not a video" * 100)
    assert r.status_code == 422
    assert "not a readable video" in r.json()["error"]["message"] or \
           "video stream" in r.json()["error"]["message"]


def test_upload_rejects_an_empty_body(client):
    r = client.post("/api/uploads", params={"filename": "empty.mp4"},
                    content=b"")
    assert r.status_code == 400


def test_upload_leaves_nothing_behind_when_rejected(env, client):
    """A rejected upload must not leave a staging directory or partial file."""
    before = {p.name for p in env["uploads"].iterdir() if p.name.startswith("up-")}
    r = client.post("/api/uploads", params={"filename": "fake.mp4"},
                    content=b"not a video" * 1000)
    assert r.status_code == 422
    after = {p.name for p in env["uploads"].iterdir() if p.name.startswith("up-")}
    assert before == after == set()


def test_upload_enforces_the_size_limit(env, client, monkeypatch):
    """
    The ceiling is applied while streaming, not after the body is buffered.

    Proved by the fact that a body larger than the limit is rejected with 413 and
    no file survives — the handler never wrote the whole thing to a kept file.
    """
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 4096)
    r = client.post("/api/uploads", params={"filename": "big.mp4"},
                    content=b"\x00" * 20000)
    assert r.status_code == 413
    assert "limit" in r.json()["error"]["message"]
    leftovers = [p for p in env["uploads"].iterdir() if p.name.startswith("up-")]
    assert leftovers == []


def test_upload_sanitises_a_traversal_filename(client):
    """
    A traversal attempt in the filename is neutralised.

    The name is never used as a path: it is reduced to one component and the
    staging directory is derived from a server-generated id.
    """
    if _SAMPLE_MP4 is None:
        pytest.skip("no sample video available")
    r = client.post("/api/uploads",
                    params={"filename": "../../etc/passwd.mp4"},
                    content=_sample_bytes())
    assert r.status_code == 200, r.text
    body = r.json()
    assert "/" not in body["filename"] and ".." not in body["filename"]
    # And nothing was written outside the staging root.
    assert not (Path(settings.UPLOADS_DIR).parent / "etc").exists()


def test_upload_can_be_read_and_deleted(client, upload):
    got = client.get(f"/api/uploads/{upload['upload_id']}")
    assert got.status_code == 200
    assert got.json()["upload_id"] == upload["upload_id"]

    deleted = client.delete(f"/api/uploads/{upload['upload_id']}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    assert client.get(f"/api/uploads/{upload['upload_id']}").status_code == 404


def test_unknown_upload_is_a_404(client):
    assert client.get("/api/uploads/up-doesnotexist").status_code == 404
    # A malformed id is a 404 too, never a traversal.
    assert client.get("/api/uploads/..%2F..%2Fetc").status_code in (404, 400)


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------
def test_create_job_rejects_an_unknown_language(client, upload):
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"],
                                       "target_lang": "klingon"})
    assert r.status_code == 422
    assert r.json()["error"]["field"] == "target_lang"


def test_create_job_rejects_auto_as_a_target(client, upload):
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"],
                                       "target_lang": "auto"})
    assert r.status_code == 422
    assert r.json()["error"]["field"] == "target_lang"


def test_create_job_rejects_an_impossible_speaker_range(client, upload):
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"],
                                       "min_speakers": 8, "max_speakers": 2})
    assert r.status_code == 422
    assert r.json()["error"]["field"] == "min_speakers"


def test_create_job_rejects_an_unknown_option_value(client, upload):
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"],
                                       "tts_model_name": "not-a-backend"})
    assert r.status_code == 422
    assert r.json()["error"]["field"] == "tts_model_name"


def test_create_job_rejects_an_incompatible_tts_language_pair(client, upload):
    """
    The TTS/language rule the pipeline itself enforces is applied up front.

    IndicF5 speaks only the Indic set, so asking it for Spanish must fail before
    a job exists rather than inside a model call.
    """
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"],
                                       "tts_model_name": "indicf5",
                                       "target_lang": "es"})
    assert r.status_code == 422
    assert r.json()["error"]["field"] == "tts_model_name"


def test_create_job_rejects_an_unknown_preset(client, upload):
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"],
                                       "quality_preset": "ULTRA"})
    assert r.status_code == 422
    assert r.json()["error"]["field"] == "quality_preset"


def test_create_job_rejects_an_unknown_subtitle_mode(client, upload):
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"],
                                       "subtitle_mode": "bogus"})
    assert r.status_code == 422
    assert r.json()["error"]["field"] == "subtitle_mode"
    assert "bogus" in r.json()["error"]["message"]
    assert "none, external, embedded, burned" in r.json()["error"]["message"]


def test_create_job_accepts_every_known_subtitle_mode(client, fake_pipeline):
    """Every advertised mode must be accepted; an upload serves one job."""
    for mode in ("none", "external", "embedded", "burned"):
        up = client.post("/api/uploads", params={"filename": "clip.mp4"},
                         content=_sample_bytes())
        assert up.status_code == 200, up.text
        r = client.post("/api/jobs", json={"upload_id": up.json()["upload_id"],
                                           "subtitle_mode": mode})
        assert r.status_code == 202, (mode, r.text)


def test_a_rejected_subtitle_mode_creates_no_job_directory(env, client, upload):
    before = set(joblib.list_jobs())
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"],
                                       "subtitle_mode": "bogus"})
    assert r.status_code == 422
    assert set(joblib.list_jobs()) == before


def test_create_job_rejects_an_unknown_upload(client):
    r = client.post("/api/jobs", json={"upload_id": "up-nope"})
    assert r.status_code == 404


def test_validation_creates_no_job_directory(env, client, upload):
    """A rejected request must not leave a job directory behind."""
    before = set(joblib.list_jobs())
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"],
                                       "target_lang": "klingon"})
    assert r.status_code == 422
    assert set(joblib.list_jobs()) == before


def test_normalises_display_names_to_codes(client, upload, fake_pipeline):
    """
    "Hindi" and "hi" must produce the same job.

    The UI sends display names; the API records codes. Without normalisation the
    manifest would record the display name, the pipeline would compute its
    fingerprint from the code, and the drift check would refuse to resume the
    job it just created.
    """
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"],
                                       "target_lang": "Hindi",
                                       "source_lang": "Auto Detect"})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["config"]["target_lang"] == "hi"
    assert body["config"]["source_lang"] == "auto"
    _wait_for_terminal(client, body["job_id"])


# ---------------------------------------------------------------------------
# A fake pipeline
# ---------------------------------------------------------------------------
class FakeTranslator:
    """
    A stand-in for `VideoTranslator` that publishes the same event tuples.

    Writes real durable state through the real `Manifest`, so the status and
    listing endpoints are exercised against genuine manifest code rather than a
    fixture's idea of it. `hold` lets a test park the render mid-flight to
    observe `running`, cancellation and SSE behaviour.
    """

    def __init__(self, job_id, source_path, config_obj, *,
                 hold: threading.Event | None = None, fail: bool = False,
                 stages: tuple = ("extraction", "separation", "final")):
        self.job_id = job_id
        self.source_path = Path(source_path)
        self.config_obj = config_obj
        self.cancel_requested = False
        self._hold = hold
        self._fail = fail
        self._stages = stages
        self.abort_called = threading.Event()

    def abort(self):
        self.abort_called.set()
        self.cancel_requested = True

    def process_video(self, video_path, source_lang="auto", target_lang="en",
                      job_id=None, resume=True, hf_token=None, **kwargs):
        job = joblib.Job.load(self.job_id)
        manifest = Manifest.load(job)
        assert manifest is not None, "job creation must have written a manifest"
        # Prove the API recorded a configuration the pipeline agrees with: this
        # is exactly the comparison process_video performs.
        assert not manifest.differs_from(kwargs), \
            f"recorded config drifted: {manifest.differs_from(kwargs)}"

        yield ("log", f"Rendering into {target_lang}")
        for index, stage in enumerate(self._stages):
            if self.cancel_requested:
                raise _cancelled()
            manifest.start_stage(stage)
            manifest.save()
            yield ("progress", (index + 1) / (len(self._stages) + 1),
                   f"Stage {stage}")
            if self._hold is not None and stage == self._stages[0]:
                # Park here so a test can observe a running job.
                self._hold.wait(timeout=20)
            if self.cancel_requested:
                raise _cancelled()
            if self._fail and stage == self._stages[-1]:
                manifest.fail_stage(stage, "synthetic failure")
                manifest.status = "FAILED"
                manifest.save()
                raise RuntimeError("synthetic failure")
            manifest.complete_stage(stage)
            manifest.save()

        final_dir = job.stage_dir("final")
        output = final_dir / f"translated_{self.source_path.name}"
        output.write_bytes(b"fake output")
        manifest.artifacts["final_output"] = str(output)
        manifest.status = "COMPLETE"
        manifest.save()
        yield ("result", str(output))


# Resolved once at import. The pipeline is imported here deliberately: these
# tests already import it (see `test_pipeline_contract`), and doing the import
# now rather than inside `_cancelled` keeps a multi-second cold import of torch
# and transformers from happening *inside a job's worker thread* at the moment it
# cancels — which made a cancelling test look like a hung job on a cold cache.
_CANCELLED_ERROR = None


def _cancelled():
    """The pipeline's CancelledError, resolved without a cold import."""
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
    """
    Replace the pipeline construction with the fake, and record the instances.

    Patched on `JobRunner._build_translator`, which is the single point where F2
    touches the pipeline. Everything above it — manager, runner, events, routes —
    is the real implementation.
    """
    built: list[FakeTranslator] = []
    holder: dict = {}

    def build(self):
        translator = FakeTranslator(
            self.job_id, self.source_path, self.config,
            hold=holder.get("hold"), fail=holder.get("fail", False),
            stages=holder.get("stages", ("extraction", "separation", "final")),
        )
        built.append(translator)
        holder["last"] = translator
        return translator

    monkeypatch.setattr(JobRunner, "_build_translator", build)
    return {"built": built, "holder": holder}


def _wait_for_terminal(client, job_id, timeout=30.0):
    """Poll a job until it reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["state"] in ("completed", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: "
                         f"{client.get(f'/api/jobs/{job_id}').json()}")


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------
def test_create_job_runs_to_completion(client, upload, fake_pipeline):
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"],
                                       "target_lang": "hi"})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    # The upload is claimed exactly once, so a second job cannot reuse it.
    assert client.get(f"/api/uploads/{upload['upload_id']}").json()["claimed_by"] \
        == job_id

    body = _wait_for_terminal(client, job_id)
    assert body["state"] == "completed"
    assert body["status"] == "COMPLETE"
    assert body["output_available"] is True
    assert body["progress_fraction"] == 1.0
    assert all(s["status"] == "COMPLETE" for s in body["stages"]
               if s["name"] in ("extraction", "separation", "final"))

    out = client.get(f"/api/jobs/{job_id}/output").json()
    assert out["available"] is True
    assert out["download_url"] == f"/api/jobs/{job_id}/output/download"
    assert out["size_bytes"] == len(b"fake output")
    # No absolute server path is ever published.
    assert "path" not in out

    download = client.get(f"/api/jobs/{job_id}/output/download")
    assert download.status_code == 200
    assert download.content == b"fake output"


def test_job_source_is_moved_into_the_job(client, upload, fake_pipeline, env):
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
    job_id = r.json()["job_id"]
    source = env["jobs"] / job_id / "source" / "clip.mp4"
    assert source.is_file(), "the claimed upload must live in the job"
    _wait_for_terminal(client, job_id)


def test_claiming_an_upload_twice_is_refused(client, upload, fake_pipeline):
    first = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
    assert first.status_code == 202
    _wait_for_terminal(client, first.json()["job_id"])

    second = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
    assert second.status_code == 409
    assert "already claimed" in second.json()["error"]["message"]


def test_a_claimed_upload_cannot_be_deleted(client, upload, fake_pipeline):
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
    _wait_for_terminal(client, r.json()["job_id"])
    deleted = client.delete(f"/api/uploads/{upload['upload_id']}")
    assert deleted.status_code == 409


def test_failed_job_reports_failure_durably(client, upload, fake_pipeline):
    fake_pipeline["holder"]["fail"] = True
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
    job_id = r.json()["job_id"]
    body = _wait_for_terminal(client, job_id)
    assert body["state"] == "failed"
    assert body["status"] == "FAILED"
    assert body["error"]
    # The durable error record survives, so a client that connects later sees it.
    assert body["errors"]
    assert body["output_available"] is False


def test_job_list_reports_durable_jobs(client, upload, fake_pipeline):
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
    job_id = r.json()["job_id"]
    _wait_for_terminal(client, job_id)

    listing = client.get("/api/jobs").json()
    assert listing["total"] >= 1
    assert any(j["job_id"] == job_id for j in listing["jobs"])
    assert listing["active"] == 0


def test_unknown_job_is_a_404(client):
    assert client.get("/api/jobs/no-such-job").status_code == 404
    assert client.get("/api/jobs/no-such-job/output").status_code == 404
    assert client.post("/api/jobs/no-such-job/cancel").status_code == 404


def _seed_durable_job(jobs_dir, job_id, *, created_at):
    """Write a minimal but real manifest for a job id, as a finished render."""
    job = joblib.Job(job_id, jobs_dir=jobs_dir)
    job.paths.ensure()
    manifest = Manifest.create(
        job, source_path="/tmp/seed.mp4", source_lang="en", target_lang="es",
        run_config={},
    )
    manifest.created_at = created_at
    manifest.updated_at = created_at
    manifest.status = "COMPLETE"
    manifest.save()
    return job_id


def test_newest_job_survives_the_list_limit(client, upload, fake_pipeline, env):
    """
    A job that finished just now must be in `/api/jobs` even when its source
    filename sorts below the truncation boundary.

    This is the real F5 failure: with lexicographic id ordering, 60 jobs whose
    stems sort after the new job's stem pushed the new job past the limit and it
    disappeared from the list. Timestamp ordering must keep it first.
    """
    jobs_dir = env["jobs"]
    # Older jobs whose source stems sort *after* the new job's stem ("zzz"),
    # so id-string ordering would rank them all above it.
    for i in range(60):
        _seed_durable_job(jobs_dir, f"zzz_video-20261020-{i:06d}-{i:08x}",
                          created_at=1_700_000_000 + i)

    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
    new_job = r.json()["job_id"]
    _wait_for_terminal(client, new_job)

    listing = client.get("/api/jobs").json()
    assert listing["total"] == 50, "listing must still honour its limit"
    assert listing["jobs"][0]["job_id"] == new_job, \
        "the freshly created job must be first, not truncated away"
    assert any(j["job_id"] == new_job for j in listing["jobs"])


def test_list_order_is_by_creation_time_not_source_name(client, env):
    """Two seeded jobs: the newer one must lead regardless of its name."""
    jobs_dir = env["jobs"]
    older = _seed_durable_job(jobs_dir, "zzz_video-20261020-000000-aaaaaaaa",
                              created_at=1_700_000_000)
    newer = _seed_durable_job(jobs_dir, "aaa_video-20261022-000000-bbbbbbbb",
                              created_at=1_700_100_000)
    listing = client.get("/api/jobs").json()
    ids = [j["job_id"] for j in listing["jobs"]]
    assert ids.index(newer) < ids.index(older)


def test_each_job_gets_its_own_pipeline_instance(client, upload, fake_pipeline,
                                                env):
    """
    Two jobs must never share a translator.

    The legacy UI reuses one instance; F2 must not, or a second render would
    overwrite the first's session, model and cancel flag.
    """
    first = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
    _wait_for_terminal(client, first.json()["job_id"])

    second_upload = client.post("/api/uploads", params={"filename": "b.mp4"},
                                content=_sample_bytes()).json()
    second = client.post("/api/jobs", json={"upload_id": second_upload["upload_id"]})
    _wait_for_terminal(client, second.json()["job_id"])

    assert len(fake_pipeline["built"]) == 2
    assert fake_pipeline["built"][0] is not fake_pipeline["built"][1]


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------
def test_cancelling_a_running_job_aborts_its_own_pipeline(client, upload,
                                                          fake_pipeline):
    """
    Cancellation reaches the right instance and ends the job as `cancelled`.

    The fake parks in its first stage, so the job is genuinely running when the
    cancel arrives, and it honours `cancel_requested` the way the real pipeline
    does at its checkpoints.
    """
    fake_pipeline["holder"]["hold"] = threading.Event()
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
    job_id = r.json()["job_id"]

    # Wait until it is actually running before cancelling.
    deadline = time.time() + 10
    while time.time() < deadline:
        if client.get(f"/api/jobs/{job_id}").json()["state"] == "running":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("job never started")

    cancel = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["state"] in ("running", "cancelled")
    assert fake_pipeline["holder"]["last"].abort_called.wait(timeout=5), \
        "abort() was not called on this job's own translator"

    # Release the parked stage so the pipeline reaches its next checkpoint.
    fake_pipeline["holder"]["hold"].set()
    body = _wait_for_terminal(client, job_id)
    assert body["state"] == "cancelled"


def test_cancelling_a_queued_job_never_starts_it(env, upload, fake_pipeline,
                                                 monkeypatch):
    """
    A job cancelled while queued must not run at all.

    The queue is held full by a first job parked in its first stage, so the
    second is still pending when it is cancelled.
    """
    fake_pipeline["holder"]["hold"] = threading.Event()
    first = client_first = None
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        first = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
        job_id_1 = first.json()["job_id"]
        deadline = time.time() + 10
        while time.time() < deadline:
            if client.get(f"/api/jobs/{job_id_1}").json()["state"] == "running":
                break
            time.sleep(0.05)

        second_upload = client.post("/api/uploads", params={"filename": "c.mp4"},
                                    content=_sample_bytes()).json()
        second = client.post("/api/jobs",
                             json={"upload_id": second_upload["upload_id"]})
        job_id_2 = second.json()["job_id"]
        assert client.get(f"/api/jobs/{job_id_2}").json()["state"] == "queued"

        cancel = client.post(f"/api/jobs/{job_id_2}/cancel")
        assert cancel.status_code == 200
        assert cancel.json()["state"] == "cancelled"
        assert client.get(f"/api/jobs/{job_id_2}").json()["state"] == "cancelled"

        # The queued job's translator was never built.
        assert len(fake_pipeline["built"]) == 1
        fake_pipeline["holder"]["hold"].set()
        _wait_for_terminal(client, job_id_1)


def test_cancelling_a_finished_job_is_a_no_op(client, upload, fake_pipeline):
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
    job_id = r.json()["job_id"]
    _wait_for_terminal(client, job_id)
    cancel = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["state"] == "completed"


# ---------------------------------------------------------------------------
# Event stream
# ---------------------------------------------------------------------------
def _read_sse(response, stop_when, max_frames=200, timeout=30.0):
    """Read SSE frames until `stop_when(frame)` is true."""
    frames = []
    deadline = time.time() + timeout
    for line in response.iter_lines():
        if time.time() > deadline:
            break
        if not line:
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data: "):
            frame = json.loads(line[6:])
            frames.append(frame)
            if stop_when(frame):
                break
        if len(frames) >= max_frames:
            break
    return frames


def test_events_stream_delivers_a_snapshot_then_completion(client, upload,
                                                           fake_pipeline):
    """
    A client that follows a job gets a snapshot immediately and the result at the
    end, without polling.
    """
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
    job_id = r.json()["job_id"]

    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = _read_sse(
            response,
            stop_when=lambda f: f["type"] in ("result", "failed", "cancelled"))

    types = [f["type"] for f in frames]
    assert types[0] == "snapshot", types
    assert "result" in types, types
    # The snapshot carries the durable state a client needs to render at once.
    assert frames[0]["payload"]["job_id"] == job_id
    assert "stages" in frames[0]["payload"]


def test_events_include_typed_progress(client, upload, fake_pipeline):
    """Progress must be structured, not only a log sentence."""
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
    job_id = r.json()["job_id"]
    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        frames = _read_sse(
            response,
            stop_when=lambda f: f["type"] in ("result", "failed", "cancelled"))
    progress = [f for f in frames if f["type"] == "progress"]
    assert progress, [f["type"] for f in frames]
    for frame in progress:
        assert 0.0 <= frame["payload"]["fraction"] <= 1.0
        assert "description" in frame["payload"]


def test_events_replay_from_last_event_id(client, upload, fake_pipeline):
    """
    `Last-Event-ID` resumes a stream without losing events.

    A client that saw the first two events and reconnects must receive the rest,
    not start over and not miss the middle.
    """
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
    job_id = r.json()["job_id"]
    _wait_for_terminal(client, job_id)

    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        all_frames = _read_sse(
            response,
            stop_when=lambda f: f["type"] in ("result", "failed", "cancelled"))
    ids = [f["event_id"] for f in all_frames if f["event_id"]]
    assert len(ids) >= 3, all_frames

    with client.stream("GET", f"/api/jobs/{job_id}/events",
                       headers={"Last-Event-ID": str(ids[1])}) as response:
        replayed = _read_sse(
            response,
            stop_when=lambda f: f["type"] in ("result", "failed", "cancelled"))
    replayed_ids = [f["event_id"] for f in replayed if f["event_id"]]
    # Everything after the position the client had seen, in order, no gaps.
    assert replayed_ids == [i for i in ids if i > ids[1]]


def test_events_stream_for_a_job_this_process_is_not_running(client, upload,
                                                             fake_pipeline):
    """
    A job from a previous process still yields a snapshot and an explanation.

    Without this a client following an old job would hang on a stream that will
    never produce an event.
    """
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
    job_id = r.json()["job_id"]
    _wait_for_terminal(client, job_id)

    # Simulate a restart: forget the runner entirely.
    manager_module.reset_manager(manager_module.JobManager())

    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        assert response.status_code == 200
        frames = _read_sse(response, stop_when=lambda f: f["type"] == "resync")
    assert frames[0]["type"] == "snapshot"
    assert frames[-1]["type"] == "resync"
    assert frames[-1]["payload"]["reason"] == "no_live_runner"
    # And the state is still readable from disk.
    assert client.get(f"/api/jobs/{job_id}").json()["state"] == "completed"


def test_interrupted_job_is_not_reported_as_running(client, upload,
                                                    fake_pipeline):
    """
    A manifest left `RUNNING` by a dead process reports `interrupted`.

    Reporting `running` would leave a client waiting forever for progress that
    cannot come.
    """
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
    job_id = r.json()["job_id"]
    _wait_for_terminal(client, job_id)

    job = joblib.Job.load(job_id)
    manifest = Manifest.load(job)
    manifest.status = "RUNNING"
    manifest.save()

    manager_module.reset_manager(manager_module.JobManager())
    body = client.get(f"/api/jobs/{job_id}").json()
    assert body["state"] == "interrupted"
    assert body["status"] == "RUNNING"
    # A coarse fraction is still reported, derived from completed stages.
    assert 0.0 < body["progress_fraction"] <= 1.0


def test_events_buffer_is_bounded(env):
    """
    The replay buffer must not grow without limit.

    A long render emits a log line per stage; an unbounded history is exactly the
    leak the phase brief forbids.
    """
    import asyncio

    from src.api.events import EventBuffer, EventType

    loop = asyncio.new_event_loop()
    try:
        buffer = EventBuffer("job-x", loop, maxlen=10)
        for index in range(100):
            buffer.publish(EventType.LOG, {"message": f"line {index}"})
        replayed, needs_resync = buffer.replay_after(0)
        assert len(replayed) == 10
        assert needs_resync is True, "an out-of-window position must be flagged"
        assert replayed[-1].payload["message"] == "line 99"
    finally:
        loop.close()


def test_events_buffer_reports_no_resync_for_a_contiguous_position(env):
    import asyncio

    from src.api.events import EventBuffer, EventType

    loop = asyncio.new_event_loop()
    try:
        buffer = EventBuffer("job-y", loop, maxlen=10)
        for index in range(5):
            buffer.publish(EventType.LOG, {"message": f"line {index}"})
        replayed, needs_resync = buffer.replay_after(2)
        assert needs_resync is False
        assert [e.event_id for e in replayed] == [3, 4, 5]
    finally:
        loop.close()


def test_job_config_applies_a_quality_preset(client, upload, fake_pipeline):
    """
    A preset is resolved by the API, not forwarded.

    Forwarding it would make the recorded configuration disagree with the
    effective one, and the drift check would then refuse to resume the job.
    """
    r = client.post("/api/jobs", json={"upload_id": upload["upload_id"],
                                       "quality_preset": "HIGH"})
    assert r.status_code == 202, r.text
    body = r.json()
    # The manifest records the preset's effect, not the preset's name.
    assert "quality_preset" not in body["config"]
    assert body["config"]["enable_lipsync"] == \
        config.preset_config("HIGH")["enable_lipsync"]
    _wait_for_terminal(client, body["job_id"])


def test_upload_meta_is_written_so_a_restart_can_still_read_it(client, upload):
    """
    Staged uploads survive a restart: their metadata is on disk, not in memory.
    """
    directory = uploads.staging_dir(upload["upload_id"])
    meta = json.loads((directory / "upload.json").read_text())
    assert meta["filename"] == "clip.mp4"
    assert meta["status"] == "staged"


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------
def test_shutdown_asks_running_jobs_to_stop(env, upload, fake_pipeline):
    """
    A clean shutdown signals the jobs this process owns.

    Without it the worker threads are simply abandoned by the exiting process,
    and a render can be killed with its manifest still claiming `RUNNING` —
    which a later status read would report as `interrupted` rather than the
    cancellation it actually was.
    """
    from fastapi.testclient import TestClient

    fake_pipeline["holder"]["hold"] = threading.Event()
    with TestClient(app) as client:
        r = client.post("/api/jobs", json={"upload_id": upload["upload_id"]})
        job_id = r.json()["job_id"]
        deadline = time.time() + 10
        while time.time() < deadline:
            if client.get(f"/api/jobs/{job_id}").json()["state"] == "running":
                break
            time.sleep(0.05)
        assert fake_pipeline["holder"]["last"].cancel_requested is False

    # Leaving the `with` block ran the lifespan shutdown.
    assert fake_pipeline["holder"]["last"].cancel_requested is True, \
        "shutdown did not signal the running job"

    # Release the parked stage so the daemon thread can finish its checkpoint.
    fake_pipeline["holder"]["hold"].set()
    manager_module.get_manager().shutdown(timeout=5.0)


def test_shutdown_is_safe_with_nothing_running(env):
    """A server with no jobs must shut down without touching anything."""
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
    assert manager_module.get_manager().active_count == 0
