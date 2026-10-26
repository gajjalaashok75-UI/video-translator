"""
F6: the bundled sample endpoints.

These exercise the real FastAPI application through `TestClient`, the real
allow-list in `src/api/samples.py`, the real upload staging layer and the real
`POST /api/jobs` claim path. Nothing about the demo is a second implementation:
the tests assert that a staged sample becomes an ordinary upload and an ordinary
durable job.

The security-relevant cases get the most attention — an unknown id, a traversal
attempt, and "no absolute path is ever published" — because the whole reason the
allow-list exists is to keep `samples/` from being a filesystem disclosure.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import manager as manager_module
from src.api import samples as samplelib
from src.api import settings, uploads
from src.api.main import app
from src.api.runner import JobConfig, JobRunner
from src.utils import config

_SAMPLE = Path("samples/hgf_scene.webm")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolate the jobs/uploads directories and the process-local manager."""
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


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------
def test_list_samples_returns_the_allowlist(client):
    r = client.get("/api/samples")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == len(body["samples"]) >= 1
    ids = [s["id"] for s in body["samples"]]
    assert "hgf_scene" in ids


def test_sample_metadata_is_complete(client):
    body = client.get("/api/samples").json()
    sample = next(s for s in body["samples"] if s["id"] == "hgf_scene")
    assert sample["title"]
    assert sample["description"]
    assert sample["source_language"] == "en"
    assert sample["target_language"] == "hi"
    assert sample["duration_seconds"] == pytest.approx(96.7, abs=0.5)
    assert sample["credit"]
    assert sample["media_type"] == "video/webm"


def test_sample_availability_matches_the_filesystem(client):
    body = client.get("/api/samples").json()
    sample = next(s for s in body["samples"] if s["id"] == "hgf_scene")
    # The repository ships this file; if it is present it must be available.
    assert sample["available"] is (_SAMPLE.exists())
    if not sample["available"]:
        assert sample["reason"]


def test_unavailable_sample_reports_a_reason_and_is_not_listed_as_available(
    client, monkeypatch,
):
    """A checkout without the media must say so, not advertise a broken demo."""
    monkeypatch.setattr(samplelib, "samples_root",
                        lambda: Path("/nonexistent/samples"))
    body = client.get("/api/samples").json()
    sample = next(s for s in body["samples"] if s["id"] == "hgf_scene")
    assert sample["available"] is False
    assert sample["reason"]
    assert sample["size_bytes"] is None


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _SAMPLE.exists(), reason="sample media not present")
def test_sample_media_is_served_with_the_right_mime(client):
    r = client.get("/api/samples/hgf_scene/media")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("video/webm")
    assert len(r.content) == _SAMPLE.stat().st_size


def test_unknown_sample_media_is_404(client):
    assert client.get("/api/samples/nope/media").status_code == 404


def test_traversal_attempt_is_rejected(client):
    """A path-like id must not escape the samples directory."""
    for attempt in ("..", "../../etc/passwd", "....//....//etc/passwd"):
        r = client.get(f"/api/samples/{attempt}/media")
        assert r.status_code in (400, 404), (attempt, r.status_code)


def test_listing_never_publishes_an_absolute_path(client):
    """The response is metadata only; no server path leaks into it."""
    body = client.get("/api/samples").json()
    text = str(body)
    assert "/workspace" not in text
    for sample in body["samples"]:
        assert not sample["filename"].startswith("/")
        assert ".." not in sample["filename"]


def test_media_path_rejects_a_table_entry_outside_the_root(monkeypatch):
    """
    Defence in depth: even a mis-edited allow-list cannot serve outside root.

    The filename comes from the table, but containment is still checked, so a
    table entry like `../secret.mp4` fails loudly here.
    """
    monkeypatch.setattr(
        samplelib, "_BY_ID",
        {"evil": samplelib.Sample(
            id="evil", filename="../secret.mp4", title="x", description="x",
            source_language="en", target_language="hi", duration_seconds=1.0,
            credit="x")},
    )
    with pytest.raises(samplelib.SampleError):
        samplelib.media_path("evil")


# ---------------------------------------------------------------------------
# Claim / staging into the ordinary upload architecture
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _SAMPLE.exists(), reason="sample media not present")
def test_claim_stages_a_normal_upload(client, env):
    r = client.post("/api/samples/hgf_scene/claim")
    assert r.status_code == 200, r.text
    body = r.json()
    # Same shape as POST /api/uploads.
    assert body["upload_id"].startswith("up-")
    assert body["filename"] == "hgf_scene.webm"
    assert body["bytes"] == _SAMPLE.stat().st_size
    assert body["status"] == "staged"
    assert body["claimed_by"] is None
    assert body["media"]["has_video"] is True

    # It is a real staged upload, readable through the uploads endpoint.
    again = client.get(f"/api/uploads/{body['upload_id']}")
    assert again.status_code == 200
    assert again.json()["upload_id"] == body["upload_id"]


@pytest.mark.skipif(not _SAMPLE.exists(), reason="sample media not present")
def test_claim_does_not_modify_the_bundled_original(client):
    before = _SAMPLE.stat().st_size
    client.post("/api/samples/hgf_scene/claim")
    assert _SAMPLE.exists()
    assert _SAMPLE.stat().st_size == before


def test_unknown_sample_claim_is_404(client):
    assert client.post("/api/samples/nope/claim").status_code == 404


def test_claim_creates_a_normal_durable_job(client, env, monkeypatch):
    """
    A staged sample starts a job through the unchanged POST /api/jobs path.

    The pipeline itself is the one substituted component (as in
    `test_api_jobs.py`): `_build_translator` returns a fake that yields the real
    event tuples and writes durable state through the real `Manifest`. Every
    layer around it — validation, claim, manifest, manager — is real, which is
    what proves there is no demo-only runner.
    """
    if not _SAMPLE.exists():
        pytest.skip("sample media not present")

    from tests.test_api_jobs import FakeTranslator  # noqa: PLC0415

    monkeypatch.setattr(JobRunner, "_build_translator", lambda self: FakeTranslator())

    claimed = client.post("/api/samples/hgf_scene/claim").json()
    r = client.post("/api/jobs", json={
        "upload_id": claimed["upload_id"],
        "source_lang": "en",
        "target_lang": "hi",
        "tts_model_name": "edge",
    })
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["job_id"]
    assert job["source_language"] == "en"
    assert job["target_language"] == "hi"

    # The upload was claimed by that job, exactly once.
    status = client.get(f"/api/uploads/{claimed['upload_id']}").json()
    assert status["status"] == "claimed"
    assert status["claimed_by"] == job["job_id"]


def test_claim_is_single_use_like_a_normal_upload(client, env, monkeypatch):
    """A second job cannot reuse the same staged sample."""
    if not _SAMPLE.exists():
        pytest.skip("sample media not present")

    from tests.test_api_jobs import FakeTranslator  # noqa: PLC0415

    monkeypatch.setattr(JobRunner, "_build_translator", lambda self: FakeTranslator())

    claimed = client.post("/api/samples/hgf_scene/claim").json()
    body = {
        "upload_id": claimed["upload_id"],
        "source_lang": "en",
        "target_lang": "hi",
        "tts_model_name": "edge",
    }
    assert client.post("/api/jobs", json=body).status_code == 202
    second = client.post("/api/jobs", json=body)
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "UPLOAD_ALREADY_CLAIMED"


def test_stage_file_rejects_a_non_video(tmp_path, monkeypatch):
    """Staging holds a sample to the same standard as an upload."""
    monkeypatch.setattr(settings, "UPLOADS_DIR", tmp_path / "uploads")
    junk = tmp_path / "notavideo.webm"
    junk.write_bytes(b"not a video" * 100)
    with pytest.raises(uploads.UploadError):
        uploads.stage_file(junk, "notavideo.webm")
