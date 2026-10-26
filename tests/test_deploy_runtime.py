"""
A9: the shared deployment runtime used by the Colab and Kaggle notebooks.

The module's contract is narrow and mostly pure, so almost all of it is tested
directly: platform detection from the marker each platform really sets, the
runtime layout, the environment variables `configure_runtime` sets, the URL
rules, and the hardware-to-preset mapping.

`start_server` is exercised for real on an ephemeral port: a real uvicorn server
in a real thread, probed over real HTTP, then stopped. The point of that test is
the lifecycle the notebooks depend on — start, answer `/api/health`, stop, and
stop again without error — because a leaked server thread is what breaks a
re-run of a notebook cell.

Nothing here mocks the thing it is checking.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from src.deploy import runtime


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
class TestDetectPlatform:

    def test_local_when_no_marker_is_present(self, monkeypatch, tmp_path):
        monkeypatch.delenv("COLAB_GPU", raising=False)
        monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
        monkeypatch.setattr(os.path, "isdir", lambda p: False)
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        assert runtime.detect_platform() == "local"

    def test_colab_from_the_gpu_variable(self, monkeypatch):
        monkeypatch.setenv("COLAB_GPU", "1")
        monkeypatch.setattr(os.path, "isdir", lambda p: p == "/content")
        assert runtime.detect_platform() == "colab"

    def test_kaggle_from_its_directory(self, monkeypatch):
        monkeypatch.delenv("COLAB_GPU", raising=False)
        monkeypatch.setattr(os.path, "isdir", lambda p: p == "/kaggle")
        assert runtime.detect_platform() == "kaggle"

    def test_docker_from_the_dockerenv_file(self, monkeypatch):
        monkeypatch.delenv("COLAB_GPU", raising=False)
        monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
        monkeypatch.setattr(os.path, "isdir", lambda p: False)
        monkeypatch.setattr(os.path, "exists", lambda p: p == "/.dockerenv")
        assert runtime.detect_platform() == "docker"


# ---------------------------------------------------------------------------
# Runtime root and layout
# ---------------------------------------------------------------------------
class TestRuntimeLayout:

    def test_the_root_is_platform_specific(self, monkeypatch):
        monkeypatch.delenv("RUNTIME_DIR", raising=False)
        assert runtime.default_runtime_root("colab") == Path("/content/video-translator-runtime")
        assert runtime.default_runtime_root("kaggle") == Path("/kaggle/working/video-translator-runtime")

    def test_an_explicit_runtime_dir_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "pinned"))
        assert runtime.default_runtime_root("colab") == tmp_path / "pinned"

    def test_runtime_dirs_creates_nothing(self, tmp_path):
        root = tmp_path / "not-created"
        dirs = runtime.runtime_dirs(root)
        assert set(dirs) == {"temp", "output", "jobs", "uploads"}
        assert not root.exists()  # pure: it only computes paths

    def test_configure_runtime_creates_every_directory_and_sets_the_environment(
            self, monkeypatch, tmp_path):
        for var in ("TEMP_DIR", "OUTPUT_DIR", "JOBS_DIR", "API_UPLOADS_DIR"):
            monkeypatch.delenv(var, raising=False)
        root = tmp_path / "runtime"

        dirs = runtime.configure_runtime(root)

        for path in dirs.values():
            assert path.is_dir()
        # The exact variables config.py reads, so a later import picks them up.
        assert os.environ["TEMP_DIR"] == str(dirs["temp"])
        assert os.environ["OUTPUT_DIR"] == str(dirs["output"])
        assert os.environ["JOBS_DIR"] == str(dirs["jobs"])
        assert os.environ["API_UPLOADS_DIR"] == str(dirs["uploads"])

    def test_configure_runtime_is_idempotent(self, monkeypatch, tmp_path):
        for var in ("TEMP_DIR", "OUTPUT_DIR", "JOBS_DIR", "API_UPLOADS_DIR"):
            monkeypatch.delenv(var, raising=False)
        root = tmp_path / "runtime"
        first = runtime.configure_runtime(root)
        second = runtime.configure_runtime(root)
        assert first == second
        assert os.environ["JOBS_DIR"] == str(second["jobs"])


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
class TestPublicUrl:

    def test_a_wildcard_bind_is_shown_as_loopback(self, monkeypatch):
        monkeypatch.delenv("PUBLIC_HOST", raising=False)
        assert runtime.public_url(8000, "0.0.0.0") == "http://127.0.0.1:8000"
        assert runtime.public_url(8000, "::") == "http://127.0.0.1:8000"

    def test_a_public_host_override_wins(self, monkeypatch):
        monkeypatch.setenv("PUBLIC_HOST", "dub.example.dev")
        monkeypatch.delenv("PUBLIC_SCHEME", raising=False)
        assert runtime.public_url(8000, "0.0.0.0") == "https://dub.example.dev"

    def test_the_scheme_is_configurable(self, monkeypatch):
        monkeypatch.setenv("PUBLIC_HOST", "dub.example.dev")
        monkeypatch.setenv("PUBLIC_SCHEME", "http")
        assert runtime.public_url(8000) == "http://dub.example.dev"

    def test_a_specific_host_is_kept(self, monkeypatch):
        monkeypatch.delenv("PUBLIC_HOST", raising=False)
        assert runtime.public_url(9000, "192.168.1.5") == "http://192.168.1.5:9000"

    def test_notebook_url_falls_back_without_colab(self, monkeypatch):
        monkeypatch.delenv("PUBLIC_HOST", raising=False)
        monkeypatch.setattr(runtime, "detect_platform", lambda: "kaggle")
        assert runtime.notebook_public_url(8000) == "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# Hardware report
# ---------------------------------------------------------------------------
class TestHardwareReport:

    def test_it_always_reports_a_recommended_preset(self):
        report = runtime.hardware_report()
        assert report["recommended_preset"] in ("FAST", "BALANCED", "HIGH")
        assert report["device"] in ("cpu", "cuda")
        assert report["platform"]

    def test_no_cuda_recommends_fast_and_says_why(self, monkeypatch):
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        report = runtime.hardware_report()
        assert report["cuda_available"] is False
        assert report["recommended_preset"] == "FAST"
        assert any("CPU" in note for note in report["notes"])

    def _with_fake_gpu(self, monkeypatch, vram_gb: float):
        torch = pytest.importorskip("torch")

        class _Props:
            name = "Fake GPU"
            total_memory = int(vram_gb * (1024 ** 3))

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: _Props())
        return runtime.hardware_report()

    def test_a_large_gpu_recommends_high(self, monkeypatch):
        report = self._with_fake_gpu(monkeypatch, 24)
        assert report["recommended_preset"] == "HIGH"
        assert report["vram_gb"] == 24.0

    def test_a_midrange_gpu_recommends_balanced(self, monkeypatch):
        assert self._with_fake_gpu(monkeypatch, 10)["recommended_preset"] == "BALANCED"

    def test_a_small_gpu_recommends_fast_and_warns(self, monkeypatch):
        report = self._with_fake_gpu(monkeypatch, 6)
        assert report["recommended_preset"] == "FAST"
        assert any("VRAM" in note for note in report["notes"])

    def test_a_missing_torch_is_reported_not_raised(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def _fail(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail)
        report = runtime.hardware_report()
        assert report["cuda_available"] is False
        assert any("torch" in note for note in report["notes"])


# ---------------------------------------------------------------------------
# FFmpeg provisioning
# ---------------------------------------------------------------------------
class TestEnsureFfmpeg:

    def test_it_reports_true_when_ffmpeg_is_already_present(self, monkeypatch):
        monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        assert runtime.ensure_ffmpeg() is True

    def test_it_does_not_install_on_a_plain_local_host(self, monkeypatch):
        monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
        monkeypatch.setattr(runtime, "detect_platform", lambda: "local")
        called = []
        monkeypatch.setattr(runtime.subprocess, "run",
                            lambda *a, **k: called.append(a) or _completed(0))
        assert runtime.ensure_ffmpeg() is False
        assert called == []  # no apt-get on a workstation

    def test_a_failed_install_is_false_not_an_exception(self, monkeypatch):
        monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
        monkeypatch.setattr(runtime, "detect_platform", lambda: "colab")
        monkeypatch.setattr(runtime.subprocess, "run",
                            lambda *a, **k: _completed(1))
        assert runtime.ensure_ffmpeg() is False

    def test_a_successful_install_is_detected(self, monkeypatch):
        state = {"installed": False}

        def _which(name):
            return "/usr/bin/ffmpeg" if state["installed"] else None

        def _run(cmd, **kwargs):
            if cmd[:2] == ["apt-get", "install"]:
                state["installed"] = True
            return _completed(0)

        monkeypatch.setattr(runtime.shutil, "which", _which)
        monkeypatch.setattr(runtime, "detect_platform", lambda: "colab")
        monkeypatch.setattr(runtime.subprocess, "run", _run)
        assert runtime.ensure_ffmpeg() is True


def _completed(returncode: int):
    import subprocess
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout="", stderr="")


# ---------------------------------------------------------------------------
# The server lifecycle the notebooks depend on
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def runtime_root(tmp_path, monkeypatch):
    """A throwaway runtime tree, with the env vars restored afterwards."""
    saved = {k: os.environ.get(k) for k in
             ("TEMP_DIR", "OUTPUT_DIR", "JOBS_DIR", "API_UPLOADS_DIR")}
    runtime.configure_runtime(tmp_path / "runtime")
    yield tmp_path / "runtime"
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class TestStartServer:

    def test_it_starts_answers_health_and_stops_cleanly(self, runtime_root):
        import json
        import urllib.request

        port = _free_port()
        handle = runtime.start_server(host="127.0.0.1", port=port,
                                      ready_timeout=120)
        try:
            assert handle.is_running()
            assert handle.url == f"http://127.0.0.1:{port}"
            with urllib.request.urlopen(handle.url + "/api/health", timeout=10) as r:
                body = json.load(r)
            assert body["status"] == "ok"
        finally:
            handle.stop()

        assert not handle.is_running()
        # Stopping twice must not raise: a re-run cell calls it again.
        handle.stop()

    def test_the_handle_is_a_context_manager(self, runtime_root):
        port = _free_port()
        with runtime.start_server(host="127.0.0.1", port=port,
                                  ready_timeout=120) as handle:
            assert handle.is_running()
        assert not handle.is_running()
