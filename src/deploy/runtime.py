"""
Shared runtime for hosted deployments (Colab, Kaggle, ephemeral VMs).

The notebooks are deliberately thin: everything they need to do that is not
"show the user a form" lives here, so the Colab and Kaggle variants cannot drift
apart and the behaviour is unit-testable without a notebook or a GPU.

Three rules shape this module:

* **No heavy imports at module import time.** `import torch` is not here; the
  pipeline is imported lazily inside `start_server` after `bootstrap()`. A caller
  can `import src.deploy.runtime` in a bare interpreter and still get the pure
  helpers (platform detection, runtime-root choice, URL building).

* **`configure_runtime` must run before `src.utils.config` is imported.**
  `config` reads `TEMP_DIR`/`OUTPUT_DIR`/`JOBS_DIR` from the environment at
  import time and creates them. Redirecting the runtime tree afterwards would be
  too late, so the notebook calls this first, before anything imports the app.

* **One process, one worker.** `start_server` runs uvicorn in a thread of the
  *current* process, matching `run.py`: the pipeline holds multi-gigabyte models
  and an in-process job registry, so a second worker would double the memory and
  split the registry. A notebook cannot afford a subprocess tree either, since
  the kernel must stay responsive.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

__all__ = [
    "ServerHandle",
    "configure_runtime",
    "default_runtime_root",
    "detect_platform",
    "ensure_ffmpeg",
    "hardware_report",
    "is_notebook",
    "notebook_public_url",
    "public_url",
    "runtime_dirs",
    "start_server",
    "wait_until_ready",
]

# Where each platform is happy to write. Colab and Kaggle both mount a
# persistent, writable volume; writing into the checkout is not durable there
# and on Kaggle the checkout is read-only for the working directory.
_PLATFORM_ROOTS = {
    "colab": "/content/video-translator-runtime",
    "kaggle": "/kaggle/working/video-translator-runtime",
}

# The directory names under the runtime root. Kept as a tuple so the layout is
# declared in one place and `runtime_dirs` and `configure_runtime` cannot
# disagree about what exists.
_SUBDIRS = ("temp", "output", "jobs", "uploads")


def detect_platform() -> str:
    """
    Which hosted environment this is: `colab`, `kaggle`, `docker`, or `local`.

    Detection is by the marker each platform actually sets, not by guessing from
    the hostname: Colab sets `COLAB_*` environment variables, Kaggle ships a
    `/kaggle` directory, and Docker leaves `/.dockerenv`.
    """
    if os.environ.get("COLAB_GPU") is not None or os.path.isdir("/content"):
        return "colab"
    if os.path.isdir("/kaggle") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if os.path.exists("/.dockerenv"):
        return "docker"
    return "local"


def is_notebook() -> bool:
    """
    Whether we are running inside an IPython kernel.

    Checked via the shell class rather than `__file__` sniffing, which is wrong
    for a notebook launched from a path without a name.
    """
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
    except Exception:  # noqa: BLE001 - IPython is optional
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def default_runtime_root(platform: str | None = None) -> Path:
    """
    The directory the runtime tree should live under on this platform.

    An explicit `RUNTIME_DIR` environment variable always wins, so a user can pin
    the location (for example onto a mounted Drive) without editing a notebook.
    """
    override = os.environ.get("RUNTIME_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    platform = platform or detect_platform()
    return Path(_PLATFORM_ROOTS.get(platform, "/tmp/video-translator-runtime"))


def runtime_dirs(root: str | Path | None = None) -> dict[str, Path]:
    """
    The runtime layout as absolute paths, without creating anything.

    Pure and side-effect-free so a caller can show the user where things will go
    before committing to it.
    """
    base = Path(root) if root is not None else default_runtime_root()
    return {name: base / name for name in _SUBDIRS}


def configure_runtime(root: str | Path | None = None) -> dict[str, Path]:
    """
    Point the application's runtime directories at a writable tree and create it.

    Sets `TEMP_DIR`, `OUTPUT_DIR`, `JOBS_DIR` and `API_UPLOADS_DIR` in the
    environment, then creates each directory. **Call this before importing
    `src.utils.config`** (or anything that imports it): config snapshots those
    variables at import time.

    Returns the layout it created. Idempotent: calling twice with the same root
    is a no-op beyond re-setting the same variables.
    """
    dirs = runtime_dirs(root)
    base = next(iter(dirs.values())).parent
    base.mkdir(parents=True, exist_ok=True)

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)

    os.environ["TEMP_DIR"] = str(dirs["temp"])
    os.environ["OUTPUT_DIR"] = str(dirs["output"])
    os.environ["JOBS_DIR"] = str(dirs["jobs"])
    os.environ["API_UPLOADS_DIR"] = str(dirs["uploads"])
    # Where the launcher and the server should look for a build, if one was
    # produced. Harmless when the frontend is not built.
    return dirs


def ensure_ffmpeg() -> bool:
    """
    Make sure FFmpeg is on PATH, installing it on Debian-based hosted images.

    Returns whether FFmpeg is available afterwards. Never raises: a failed
    install leaves the answer `False` and the caller decides what to do. On
    Colab and Kaggle `apt-get` works without a password prompt; elsewhere the
    install is skipped and the caller is expected to have provided FFmpeg.
    """
    if shutil.which("ffmpeg"):
        return True

    platform = detect_platform()
    if platform not in ("colab", "kaggle", "docker"):
        return False

    commands = [
        ["apt-get", "update", "-qq"],
        ["apt-get", "install", "-y", "-qq", "ffmpeg"],
    ]
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError):
            return False
        if result.returncode != 0 and cmd[1] != "update":
            # `apt-get update` can fail on a flaky mirror while the package is
            # already cached; only the install result decides.
            return False
    return shutil.which("ffmpeg") is not None


def hardware_report() -> dict:
    """
    What the machine offers, and which preset it can realistically run.

    Imports torch lazily so this can be called (and tested) before the pipeline
    is set up. Returns a plain dict with a `recommended_preset` key, which is the
    value the notebook shows and defaults the quality control to.

    The mapping is a judgement, stated so it can be argued with: lip-sync is the
    VRAM-hungry stage, so a GPU with little free memory gets `FAST` (no
    lip-sync), a mid-range GPU gets `BALANCED` (Wav2Lip), and a large one can
    afford `HIGH` (LivePortrait).
    """
    report: dict = {
        "platform": detect_platform(),
        "cuda_available": False,
        "gpu_name": None,
        "vram_gb": None,
        "device": "cpu",
        "recommended_preset": "FAST",
        "notes": [],
    }

    try:
        import torch  # local import: keep module import cheap
    except Exception as exc:  # noqa: BLE001 - torch may be absent entirely
        report["notes"].append(f"torch is not importable: {exc}")
        return report

    report["torch_version"] = getattr(torch, "__version__", None)
    if not torch.cuda.is_available():
        report["notes"].append(
            "No CUDA GPU. Renders will run on CPU: correct, but slow, and "
            "lip-sync is unavailable.")
        return report

    report["cuda_available"] = True
    report["device"] = "cuda"
    try:
        props = torch.cuda.get_device_properties(0)
        report["gpu_name"] = props.name
        report["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
    except Exception as exc:  # noqa: BLE001
        report["notes"].append(f"Could not read GPU properties: {exc}")
        return report

    vram = report["vram_gb"] or 0
    if vram >= 16:
        report["recommended_preset"] = "HIGH"
    elif vram >= 8:
        report["recommended_preset"] = "BALANCED"
    else:
        report["recommended_preset"] = "FAST"
        report["notes"].append(
            f"{vram} GB VRAM is tight for lip-sync; FAST avoids it.")
    return report


def public_url(port: int, host: str = "127.0.0.1") -> str:
    """
    The URL to show the user.

    A wildcard bind is not navigable, so it is shown as loopback. When
    `PUBLIC_HOST` is set (a tunnel, a proxy, a reserved hostname) that wins,
    because on Colab and Kaggle the loopback address is reachable only from
    inside the VM and the user is on a different machine.
    """
    override = os.environ.get("PUBLIC_HOST", "").strip()
    if override:
        scheme = os.environ.get("PUBLIC_SCHEME", "https").strip() or "https"
        return f"{scheme}://{override}"
    display = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    return f"http://{display}:{port}"


def notebook_public_url(port: int, host: str = "0.0.0.0") -> str:
    """
    The URL a notebook user should click, using the platform's own proxy.

    Colab can proxy a port with no account or token
    (`google.colab.kernel.proxyPort`), so that is preferred when available: the
    result is reachable from the user's browser, which `127.0.0.1:port` is not.
    Kaggle has no equivalent built-in proxy, so a tunnel must be supplied
    through `PUBLIC_HOST`; without one the loopback URL is returned and the
    notebook tells the user it is only reachable inside the VM.

    Never raises: any failure falls back to `public_url`.
    """
    if detect_platform() == "colab":
        try:
            from google.colab import output  # type: ignore

            proxy = output.eval_js(f"google.colab.kernel.proxyPort({port})")
            if isinstance(proxy, str) and proxy.startswith("http"):
                return proxy.rstrip("/")
        except Exception:  # noqa: BLE001 - not Colab, or the API moved
            pass
    return public_url(port, host)


def wait_until_ready(url: str, timeout: float = 120.0,
                     interval: float = 0.5) -> bool:
    """
    Poll `url` until it answers, or `timeout` elapses. Returns whether it did.

    Polls the health endpoint over HTTP rather than a bare TCP connect, so a
    port that is open but serving 500s does not read as ready.
    """
    health = url.rstrip("/") + "/api/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=5) as response:
                if 200 <= response.status < 300:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(interval)
    return False


@dataclasses.dataclass
class ServerHandle:
    """
    A running server in this process, and a way to stop it.

    `stop()` sets uvicorn's `should_exit` flag and joins the thread with a
    timeout, so re-running a notebook cell does not leave a server holding the
    port. It is safe to call more than once.
    """

    url: str
    host: str
    port: int
    server: object
    thread: threading.Thread

    def is_running(self) -> bool:
        return self.thread.is_alive()

    def stop(self, timeout: float = 10.0) -> None:
        try:
            self.server.should_exit = True  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - a non-uvicorn server has no flag
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def __enter__(self) -> "ServerHandle":
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def start_server(host: str = "0.0.0.0", port: int = 8000,
                 *, ready_timeout: float = 180.0,
                 log_level: str = "info") -> ServerHandle:
    """
    Start the API (and the built UI, if present) in a background thread.

    Runs the canonical `bootstrap()` first, then imports the app, then serves
    with uvicorn in a daemon thread of this process. Returns a `ServerHandle`
    whose `url` is the address to show the user and whose `stop()` shuts it
    down. `PUBLIC_HOST` overrides the displayed URL for tunnelled platforms.

    The bind defaults to `0.0.0.0` because a notebook is reached through a
    proxy or tunnel that connects from outside the loopback interface.
    """
    from src.api.bootstrap import bootstrap

    bootstrap()

    import uvicorn  # noqa: E402  (after bootstrap)

    from src.api.main import create_app  # noqa: E402  (after bootstrap)

    config = uvicorn.Config(
        create_app(), host=host, port=port, log_level=log_level, workers=1)
    server = uvicorn.Server(config)

    thread = threading.Thread(
        target=server.run, name="video-translator-server", daemon=True)
    thread.start()

    url = public_url(port, host)
    if not wait_until_ready(url, timeout=ready_timeout):
        server.should_exit = True
        thread.join(timeout=10)
        raise RuntimeError(
            f"The server did not become ready at {url} within "
            f"{ready_timeout:.0f}s. Check the log output above.")

    return ServerHandle(url=url, host=host, port=port, server=server,
                        thread=thread)
