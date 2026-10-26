"""
Production launcher.

One command starts the whole application: this process checks the environment,
runs the canonical import-order bootstrap, and serves the FastAPI API and the
built React frontend from a single origin.

    python run.py                # http://127.0.0.1:8000
    python run.py --open         # ...and open a browser
    python run.py --host 0.0.0.0 --port 8080

Design notes that matter on Windows:

* **No subprocess tree.** `uvicorn.run()` is called in-process with a single
  worker, so Ctrl+C reaches the server directly and there is no orphaned child
  to clean up. The alternative — spawning `uvicorn` as a child — is what makes
  Ctrl+C unreliable and leaves a stray process holding the port.
* **No shell assumptions.** Everything here is `pathlib` and `sys.executable`.
  Paths with spaces and non-ASCII characters are passed as a list, never
  interpolated into a command string, so a checkout under
  `C:\\Users\\José\\My Projects\\video-translator` works the same as any other.
* **`reload` is off.** The reloader runs the app in a child process and watches
  files; that is a development convenience, and development uses the Vite dev
  server plus `uvicorn --reload` directly (see the README). Production wants one
  process.

The heavy pipeline import order is *not* reimplemented here: `bootstrap()` is
the single canonical sequence (see `src/api/bootstrap.py`), and importing the
app runs it too. This module only does the checks that need to happen before any
of that, then hands off to uvicorn.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
import webbrowser
from pathlib import Path

# The minimum interpreter the dependency set supports (several pinned ML wheels
# have no 3.13 build; see pyproject.toml and RUNNING.md).
MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 12)

# How long to wait for the server to accept connections before opening a
# browser, and how often to check. Bounded so a slow start never hangs the
# launcher's own thread.
_BROWSER_READY_TIMEOUT = 30.0
_BROWSER_POLL_INTERVAL = 0.25


def project_root() -> Path:
    """The repository root, from this file's location."""
    return Path(__file__).resolve().parent


def check_python(version_info: tuple[int, ...] | None = None) -> list[str]:
    """
    Problems with the running interpreter, as human-readable strings.

    Returns a list rather than raising so the caller can print every problem at
    once instead of one per run. An empty list means "fine".
    """
    version = version_info or sys.version_info
    major_minor = (version[0], version[1])
    if major_minor < MIN_PYTHON:
        return [f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required; "
                f"this is {version[0]}.{version[1]}."]
    if major_minor > MAX_PYTHON:
        return [f"Python {version[0]}.{version[1]} is newer than the supported "
                f"{MAX_PYTHON[0]}.{MAX_PYTHON[1]}. Several pinned ML wheels have "
                f"no build for it; use Python {MAX_PYTHON[0]}.{MAX_PYTHON[1]} "
                f"(see RUNNING.md)."]
    return []


def check_dependencies() -> list[str]:
    """Whether the web-serving dependencies are importable."""
    missing = []
    for module in ("fastapi", "uvicorn", "pydantic"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if not missing:
        return []
    return [
        "Missing Python dependencies: " + ", ".join(missing) + ".",
        "Install them with:  pip install -e \".[api]\"  (see README.md).",
    ]


def ffmpeg_available() -> bool:
    """Whether FFmpeg is on PATH. The pipeline shells out to it for every job."""
    return shutil.which("ffmpeg") is not None


def frontend_build_dir() -> Path:
    """The React build directory the server will serve."""
    override = os.environ.get("API_FRONTEND_DIST", "").strip()
    if override:
        return Path(override)
    return project_root() / "frontend" / "dist"


def frontend_build_present(dist: Path | None = None) -> bool:
    """Whether a built `index.html` exists (the same check the server makes)."""
    dist = dist if dist is not None else frontend_build_dir()
    return (dist / "index.html").is_file()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Command-line arguments.

    Defaults come from the environment (`API_HOST`, `API_PORT`, `API_OPEN_BROWSER`)
    so the same knobs work from a shell or a service file; an explicit flag wins.
    """
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Launch the Video Translator (FastAPI API + React UI).",
    )
    parser.add_argument(
        "--host", default=os.environ.get("API_HOST", "127.0.0.1"),
        help="Interface to bind (default: 127.0.0.1; use 0.0.0.0 to expose).")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("API_PORT", "8000")),
        help="Port to listen on (default: 8000).")
    parser.add_argument(
        "--open", dest="open_browser", action="store_true",
        default=_env_true("API_OPEN_BROWSER"),
        help="Open a browser to the app once the server is ready.")
    parser.add_argument(
        "--no-open", dest="open_browser", action="store_false",
        help="Do not open a browser (overrides API_OPEN_BROWSER).")
    parser.add_argument(
        "--log-level", default=os.environ.get("API_LOG_LEVEL", "info"),
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level (default: info).")
    parser.add_argument(
        "--check", action="store_true",
        help="Run the environment checks and exit without starting the server.")
    return parser.parse_args(argv)


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _url(host: str, port: int) -> str:
    """
    The URL to show a human.

    A server bound to `0.0.0.0` or `::` is listening on every interface, which
    is not a navigable address; `127.0.0.1` is what the person at the machine
    should click.
    """
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    if ":" in display_host and not display_host.startswith("["):
        display_host = f"[{display_host}]"  # IPv6 literal
    return f"http://{display_host}:{port}"


def print_banner(host: str, port: int, *, build_present: bool,
                 ffmpeg: bool) -> None:
    """
    The startup summary.

    Deliberately short and free of secrets: no tokens, no model URLs, no
    filesystem layout beyond what the operator already knows. Model availability
    is not probed here (that would load or download nothing but still cost time);
    the dashboard and `/api/models` report it on demand.
    """
    line = "=" * 58
    print(f"\n{line}")
    print("  Video Translator")
    print(f"{line}")
    print(f"  API:       ready")
    if build_present:
        print(f"  Frontend:  ready (built React UI)")
    else:
        print(f"  Frontend:  not built — API only")
    print(f"  URL:       {_url(host, port)}")
    if not build_present:
        print()
        print("  The React UI has not been built, so only the API is served.")
        print("  To build it:")
        print("      cd frontend")
        print("      npm install")
        print("      npm run build")
        print("  Then restart, or reload this page.")
    if not ffmpeg:
        print()
        print("  WARNING: FFmpeg was not found on PATH. Dubbing needs it.")
        print("  Install FFmpeg and reopen the terminal (see README.md).")
    print(f"{line}\n")


def preflight(*, require_build: bool = False) -> list[str]:
    """
    Run every startup check. Returns a list of fatal problems (empty is fine).

    A missing frontend build or FFmpeg is *not* returned here: the API is useful
    on its own and the banner reports those. Only a wrong interpreter, missing
    web dependencies, or an explicitly required-but-absent build stops the
    launch.
    """
    problems: list[str] = []
    problems.extend(check_python())
    problems.extend(check_dependencies())

    if require_build and not frontend_build_present():
        problems.append(
            "No frontend build found and API_REQUIRE_FRONTEND was set. Build it:\n"
            "    cd frontend && npm install && npm run build")

    return problems


def open_browser_when_ready(url: str, host: str, port: int,
                            timeout: float = _BROWSER_READY_TIMEOUT) -> threading.Thread:
    """
    Open `url` in a browser once the port accepts connections.

    Runs on a daemon thread so a slow or failing open can never hold up the
    server or block Ctrl+C. It polls a TCP connect rather than sleeping a fixed
    time, so a fast start opens the browser promptly and a slow one still opens
    it when the server is actually up — or gives up quietly after `timeout`
    rather than opening a dead tab.
    """
    import socket

    def _worker() -> None:
        deadline = time.monotonic() + timeout
        connect_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((connect_host, port), timeout=1.0):
                    break
            except OSError:
                time.sleep(_BROWSER_POLL_INTERVAL)
        else:
            print(f"  (Browser was not opened automatically; go to {url})")
            return
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - a failed open must not matter
            print(f"  (Could not open a browser automatically; go to {url})")

    thread = threading.Thread(target=_worker, name="open-browser", daemon=True)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)

    # 1. Environment checks — stdlib only, before any heavy import. Every
    #    problem `preflight` returns is fatal; warnings that do not stop the
    #    launch (a missing build, a missing FFmpeg) are shown in the banner.
    require_build = os.environ.get("API_REQUIRE_FRONTEND", "").strip().lower() in (
        "1", "true", "yes", "on")
    problems = preflight(require_build=require_build)
    if problems:
        print("\nVideo Translator cannot start:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(file=sys.stderr)
        return 1

    build_present = frontend_build_present()

    if args.check:
        print_banner(args.host, args.port, build_present=build_present,
                     ffmpeg=ffmpeg_available())
        print("Environment check complete. Not starting the server (--check).")
        return 0

    # 2. Canonical bootstrap, then import the app. Importing `src.api.main`
    #    runs `bootstrap()` itself, but calling it first makes the sequence
    #    explicit and means the import happens against an already-prepared
    #    process — the order `app.py` relies on for Windows DLL loading.
    from src.api.bootstrap import bootstrap

    bootstrap()

    import uvicorn  # noqa: E402  (after bootstrap)

    from src.api.main import app  # noqa: E402  (after bootstrap)

    print_banner(args.host, args.port, build_present=build_present,
                 ffmpeg=ffmpeg_available())

    url = _url(args.host, args.port)
    if args.open_browser:
        open_browser_when_ready(url, args.host, args.port)

    # 3. Serve. One worker, in-process: the pipeline holds multi-gigabyte models
    #    and a job manager in this process, so a second worker would double the
    #    memory and split the job registry.
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            workers=1,
        )
    except KeyboardInterrupt:
        # uvicorn installs its own handler and shuts down cleanly; this is the
        # belt-and-braces path for a Ctrl+C that lands before it does.
        print("\nShutting down...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
