"""
F7: production static serving, the one-command launcher, and the runtime-dir
policy.

These exercise the real pieces end to end: a real `create_app()` with a real
build directory on disk, a real `TestClient` (so the real ASGI stack, the real
error handlers and the real routing run), and the real `run.py` preflight and
argument parser. Nothing here stubs the thing it is checking.

The cases that matter most are the ones where a static-serving shortcut would
break the API:

* an unmatched `/api/...` path must stay a JSON 404, never the React shell;
* a wrong method must stay Starlette's 405 with the API envelope;
* a traversal-shaped path must not be answered with the shell (or a file);
* an absent build must leave the API fully working.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import run as launcher
from src.api import frontend, settings
from src.api.main import create_app


# ---------------------------------------------------------------------------
# A real build directory
# ---------------------------------------------------------------------------
@pytest.fixture
def build(tmp_path: Path) -> Path:
    """A minimal but real Vite-shaped build: index.html plus hashed assets."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><body><div id=root></div></body></html>",
        encoding="utf-8")
    (dist / "assets" / "index-abc123.css").write_text("body{}", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    return dist


def _client(monkeypatch, dist: Path) -> TestClient:
    """A real app serving `dist`, with the frontend build swapped in."""
    monkeypatch.setattr(settings, "FRONTEND_DIST", dist)
    monkeypatch.setattr(settings, "SERVE_FRONTEND", True)
    return TestClient(create_app())


@pytest.fixture
def client(monkeypatch, build):
    with _client(monkeypatch, build) as c:
        yield c


# ---------------------------------------------------------------------------
# Serving the build
# ---------------------------------------------------------------------------
def test_root_serves_the_shell(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "id=root" in r.text


def test_a_real_asset_is_served_as_itself(client):
    r = client.get("/assets/index-abc123.css")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/css")
    assert r.text == "body{}"


def test_a_root_level_build_file_is_served(client):
    r = client.get("/favicon.svg")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg+xml")


@pytest.mark.parametrize("path", [
    "/jobs", "/jobs/20260101-000000-abcdef",
    "/jobs/20260101-000000-abcdef/result", "/dub", "/demo",
    "/dashboard", "/languages",
])
def test_a_client_route_returns_the_shell_for_a_deep_link(client, path):
    """A refresh or a shared link must reach the router, not a 404."""
    r = client.get(path)
    assert r.status_code == 200, path
    assert r.headers["content-type"].startswith("text/html")
    assert "id=root" in r.text


# ---------------------------------------------------------------------------
# The API is never shadowed
# ---------------------------------------------------------------------------
def test_api_health_still_works(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")


def test_an_unknown_api_path_is_a_json_404_not_the_shell(client):
    r = client.get("/api/nope")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["error"]["code"] == "NOT_FOUND"


def test_a_wrong_method_keeps_the_envelope(client):
    r = client.put("/api/jobs/20260101-000000-abcdef")
    assert r.status_code == 405
    assert r.json()["error"]["code"] == "METHOD_NOT_ALLOWED"


def test_a_traversal_shaped_api_path_is_not_the_shell(client):
    r = client.post("/api/jobs/..%2F..%2Fetc/resume")
    assert r.status_code in (400, 404), r.status_code
    assert r.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize("path", ["/docs", "/openapi.json", "/redoc"])
def test_documentation_routes_are_untouched(client, path):
    r = client.get(path)
    assert r.status_code == 200, path


def test_an_unknown_non_client_path_is_a_404_not_the_shell(client):
    """A path the frontend does not own must not look like a success."""
    for path in ("/nope", "/etc/passwd/media", "/assets/missing.css"):
        r = client.get(path)
        assert r.status_code == 404, path
        assert not r.headers.get("content-type", "").startswith("text/html"), path


def test_a_traversal_path_never_discloses_a_file(client):
    r = client.get("/../etc/passwd")
    assert r.status_code == 404
    assert "root:" not in r.text


# ---------------------------------------------------------------------------
# No build present
# ---------------------------------------------------------------------------
def test_absent_build_still_serves_the_api(monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with _client(monkeypatch, empty) as c:
        assert c.get("/api/health").status_code == 200
        assert c.get("/api/nope").json()["error"]["code"] == "NOT_FOUND"
        assert c.get("/docs").status_code == 200

        root = c.get("/")
        assert root.status_code == 404
        assert root.headers["content-type"].startswith("text/html")
        assert "npm run build" in root.text


def test_serve_frontend_off_is_api_only(monkeypatch, build):
    monkeypatch.setattr(settings, "FRONTEND_DIST", build)
    monkeypatch.setattr(settings, "SERVE_FRONTEND", False)
    with TestClient(create_app()) as c:
        assert c.get("/api/health").status_code == 200
        assert c.get("/").status_code == 404


# ---------------------------------------------------------------------------
# The fallback's own logic, without a server in the way
# ---------------------------------------------------------------------------
class TestFallbackPredicates:

    def test_protected_namespaces(self):
        for path in ("/api", "/api/jobs", "/docs", "/redoc", "/openapi.json"):
            assert frontend._is_protected(path), path

    def test_public_paths_are_not_protected(self):
        for path in ("/", "/jobs", "/jobs/1", "/apiary", "/docsomething"):
            assert not frontend._is_protected(path), path

    def test_client_routes_match_their_subpaths(self):
        for path in ("/", "/jobs", "/jobs/1", "/jobs/1/result", "/demo",
                     "/dub", "/dashboard", "/languages"):
            assert frontend._is_client_route(path), path

    def test_unknown_paths_are_not_client_routes(self):
        for path in ("/nope", "/jobsomething", "/etc/passwd", "/assets/x.css"):
            assert not frontend._is_client_route(path), path

    def test_safe_file_rejects_escapes(self, build):
        for attempt in ("../index.html", "../../etc/passwd", "/etc/passwd"):
            assert frontend._safe_file(build, attempt) is None, attempt

    def test_safe_file_accepts_a_real_file(self, build):
        resolved = frontend._safe_file(build, "/assets/index-abc123.css")
        assert resolved is not None and resolved.is_file()


# ---------------------------------------------------------------------------
# The launcher
# ---------------------------------------------------------------------------
class TestLauncher:

    def test_default_host_and_port(self, monkeypatch):
        monkeypatch.delenv("API_HOST", raising=False)
        monkeypatch.delenv("API_PORT", raising=False)
        args = launcher.parse_args([])
        assert args.host == "127.0.0.1"
        assert args.port == 8000

    def test_environment_overrides_the_defaults(self, monkeypatch):
        monkeypatch.setenv("API_HOST", "0.0.0.0")
        monkeypatch.setenv("API_PORT", "9099")
        args = launcher.parse_args([])
        assert args.host == "0.0.0.0"
        assert args.port == 9099

    def test_an_explicit_flag_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("API_PORT", "9099")
        args = launcher.parse_args(["--port", "1234"])
        assert args.port == 1234

    def test_check_flag_is_recognised(self):
        assert launcher.parse_args(["--check"]).check is True

    def test_display_url_never_shows_a_wildcard_bind(self):
        assert launcher._url("0.0.0.0", 8000) == "http://127.0.0.1:8000"
        assert launcher._url("127.0.0.1", 8000) == "http://127.0.0.1:8000"

    def test_build_presence_matches_the_server_check(self, build, monkeypatch):
        monkeypatch.setenv("API_FRONTEND_DIST", str(build))
        assert launcher.frontend_build_present() is True
        monkeypatch.setenv("API_FRONTEND_DIST", str(build / "missing"))
        assert launcher.frontend_build_present() is False

    def test_preflight_does_not_require_a_build_by_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("API_FRONTEND_DIST", str(tmp_path / "none"))
        # Python and the web dependencies are satisfied in this environment, so
        # a missing build must not be fatal.
        assert launcher.preflight(require_build=False) == []

    def test_preflight_requires_a_build_when_asked(self, monkeypatch, tmp_path):
        monkeypatch.setenv("API_FRONTEND_DIST", str(tmp_path / "none"))
        problems = launcher.preflight(require_build=True)
        assert any("frontend build" in p for p in problems)

    def test_python_check_accepts_a_supported_version(self):
        assert launcher.check_python((3, 11, 0)) == []

    def test_python_check_rejects_too_old_and_too_new(self):
        assert launcher.check_python((3, 8, 0))
        assert launcher.check_python((3, 13, 0))


# ---------------------------------------------------------------------------
# Runtime-directory policy
# ---------------------------------------------------------------------------
class TestRuntimeDirectories:

    def test_a_job_owns_its_scratch_space(self, tmp_path, monkeypatch):
        from src.core import jobs as joblib
        monkeypatch.setattr(joblib.config, "JOBS_DIR", tmp_path / "jobs")
        job = joblib.Job.create(source_path=str(tmp_path / "movie.mp4"))
        scratch = job.scratch_dir()
        assert scratch.is_dir()
        # Contained by the job, so two jobs can never share a working file.
        scratch.resolve().relative_to(job.root.resolve())

    def test_scratch_is_dropped_by_a_resumable_clean(self, tmp_path, monkeypatch):
        from src.core import jobs as joblib
        monkeypatch.setattr(joblib.config, "JOBS_DIR", tmp_path / "jobs")
        job = joblib.Job.create(source_path=str(tmp_path / "movie.mp4"))
        (job.scratch_dir() / "working.wav").write_bytes(b"x" * 16)
        job.clean(level=joblib.config.CLEANUP_KEEP_RESUMABLE)
        assert not (job.scratch_dir() / "working.wav").exists()
