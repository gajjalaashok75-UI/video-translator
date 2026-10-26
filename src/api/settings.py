"""
API-layer tunables.

These belong to the HTTP layer, not to the pipeline, so they live here rather
than in `src/utils/config.py`: nothing in `src/core` reads them, and keeping them
separate means the pipeline's own configuration surface is untouched by F2.

Every value can be overridden with an environment variable so an operator can
tune a deployment without editing code. Defaults are deliberately conservative:
one render at a time (the pipeline loads multi-gigabyte models into one process,
so parallel renders would fight over memory), and a 4 GiB upload ceiling.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.utils import config


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# --- Uploads ---------------------------------------------------------------

# Staged uploads live under the temp directory. A `.gitignore` is written into
# the uploads root so staged media never shows up as untracked repository noise
# (the repository deliberately tracks `temp/` and `jobs/` for its own evidence).
UPLOADS_DIR = Path(os.environ.get("API_UPLOADS_DIR",
                                  str(config.TEMP_DIR / "uploads")))

# 4 GiB. A feature-length 1080p source fits; anything larger is almost certainly
# a mistake or an attack, and the stream is rejected part-way through rather
# than after the whole file has landed on disk.
MAX_UPLOAD_BYTES = _int_env("API_MAX_UPLOAD_BYTES", 4 * 1024 ** 3)

# Read the request body in 1 MiB chunks. Small enough that a 4 GiB upload never
# materialises in memory, large enough that the loop is not the bottleneck.
UPLOAD_CHUNK_BYTES = _int_env("API_UPLOAD_CHUNK_BYTES", 1024 * 1024)

# Extensions accepted for upload. The pipeline can mux any container FFmpeg
# reads, but the UI has always offered these four, and an allow-list is the only
# safe direction for a field that reaches the filesystem.
ALLOWED_EXTENSIONS = frozenset({
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv",
})

# Abandoned staged uploads are *not* collected automatically in F2. Cleanup
# policy: an upload that is never claimed stays under `UPLOADS_DIR` and can be
# removed by deleting that directory; a claimed upload is moved into the job's
# `source/` directory, so it becomes the job's own durable file and is covered
# by the existing job cleanup rules (`Job.clean`). A TTL sweeper is deferred to
# a later phase; see the F2 section of AGENTS.md.

# --- Jobs ------------------------------------------------------------------

# One render at a time. The pipeline holds Whisper, Demucs and optionally TTS
# models in one process, so concurrent renders would contend for the same GPU
# memory and the same model cache.
MAX_CONCURRENT_JOBS = _int_env("API_MAX_CONCURRENT_JOBS", 1)

# Finished runners are retained so a late SSE subscriber can still read their
# replay buffer, but only a bounded number: the durable state is the manifest,
# and keeping every runner alive would leak a buffer per render.
MAX_RETAINED_RUNNERS = _int_env("API_MAX_RETAINED_RUNNERS", 32)

# --- Event stream ----------------------------------------------------------

# Events kept per job for `Last-Event-ID` replay. Bounded on purpose: the
# durable record is the manifest plus the job log, and an unbounded buffer in
# RAM is exactly what the phase brief forbids.
EVENT_BUFFER_SIZE = _int_env("API_EVENT_BUFFER_SIZE", 500)

# Per-subscriber queue. If a client reads slower than the pipeline emits, the
# oldest events are dropped and the subscriber is told to resync rather than
# being allowed to block the worker thread.
SUBSCRIBER_QUEUE_SIZE = _int_env("API_SUBSCRIBER_QUEUE_SIZE", 256)

# Maximum concurrent SSE subscribers per job. Prevents one client opening
# thousands of connections from exhausting file descriptors.
MAX_SUBSCRIBERS_PER_JOB = _int_env("API_MAX_SUBSCRIBERS_PER_JOB", 16)

# Heartbeat comment interval. Model stages can be silent for minutes, and an
# idle connection is closed by proxies and browsers; a comment keeps it alive
# without consuming an event id.
HEARTBEAT_SECONDS = _float_env("API_SSE_HEARTBEAT_SECONDS", 15.0)


# How long an unclaimed staged upload is kept before startup cleanup removes it.
# A day is long enough that a slow client is never surprised, short enough that a
# forgotten upload does not accumulate forever. Zero or less disables the sweep.
UPLOAD_TTL_SECONDS = _float_env("API_UPLOAD_TTL_SECONDS", 24 * 60 * 60)


# --- Production serving (F7) ------------------------------------------------

# The React build, served by FastAPI in production. Relative to the repository
# root, resolved from `config.BASE_DIR` so it is correct regardless of the
# working directory the launcher is started from. This is a build artifact, not
# source: it is git-ignored and read-only to the server.
FRONTEND_DIST = Path(os.environ.get(
    "API_FRONTEND_DIST", str(config.BASE_DIR / "frontend" / "dist")))

# Serve the built frontend from this process. Defaults to on, so the canonical
# launcher needs no extra flag; a backend-only deployment (or a test that wants
# only the API) sets `API_SERVE_FRONTEND=false`. When on but no build is present,
# the app still starts and answers `/` with instructions — never a crash.
SERVE_FRONTEND = _bool_env("API_SERVE_FRONTEND", True)

# Allow the Vite dev-server origins in CORS. On by default so local development
# keeps working with no configuration; a production deployment that serves the
# frontend same-origin sets `API_CORS_DEV_ORIGINS=false` to drop them.
CORS_DEV_ORIGINS = _bool_env("API_CORS_DEV_ORIGINS", True)

# Where the launcher binds by default. `0.0.0.0` matches the Docker image's
# historical behavior; the launcher's own default is loopback (see `run.py`),
# because a local tool should not be reachable from the network unless asked.
HOST = os.environ.get("API_HOST", "127.0.0.1").strip() or "127.0.0.1"
PORT = _int_env("API_PORT", 8000)

# Open a browser to the app URL once the server is listening. Off by default so
# a scripted or headless launch is never surprised by a window; the launcher
# turns it on with `--open` or `API_OPEN_BROWSER=true`.
OPEN_BROWSER = _bool_env("API_OPEN_BROWSER", False)
