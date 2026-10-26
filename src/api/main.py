"""
FastAPI application factory.

F1 exposes read-only discovery. F2 adds uploads, job execution and an SSE event
stream on top of the same schemas; the pipeline modules themselves are unchanged
and are only imported once a job actually runs. Nothing in this module starts a
render or loads a model at import time.

Run it with:

    uvicorn src.api.main:app --host 127.0.0.1 --port 8000

The legacy Gradio UI is untouched and still runs with `python app.py`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import API_VERSION, settings, uploads
from src.api.bootstrap import bootstrap

# Apply the pipeline's import-order requirements before anything imports the
# pipeline. `bootstrap()` is idempotent, so importing this module from a test
# that already bootstrapped is harmless.
bootstrap()

from src.api import frontend  # noqa: E402  (after bootstrap)
from src.api.errors import install_error_handlers  # noqa: E402  (after bootstrap)
from src.api.routes import (  # noqa: E402
    jobs,
    samples,
    system,
    uploads as uploads_routes,
)

logger = logging.getLogger(__name__)

# Development origins only. The React dev server runs on 5173 (Vite default) and
# may be reached as localhost or 127.0.0.1. In production the built frontend is
# served from this same origin, which needs no CORS entry at all.
_DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def _cors_origins() -> list[str]:
    """
    Allowed CORS origins.

    An explicit list rather than `*`: the API accepts uploads and controls
    renders, and a wildcard origin is not something to carry into that.

    The dev-server origins are included because a frontend served from Vite on
    :5173 is a *different* origin from this process. In the integrated
    production deployment the frontend is same-origin, so those entries are dead
    weight and a locked-down deployment can drop them with
    `API_CORS_DEV_ORIGINS=false`. `API_CORS_ORIGINS` (comma-separated) extends
    the list for an unusual setup.
    """
    origins: list[str] = []
    if settings.CORS_DEV_ORIGINS:
        origins.extend(_DEV_ORIGINS)
    extra = os.environ.get("API_CORS_ORIGINS", "").strip()
    for origin in extra.split(","):
        origin = origin.strip()
        if origin and origin not in origins:
            origins.append(origin)
    return origins


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(
        title="Video Translator API",
        version=API_VERSION,
        description=(
            "Discovery, upload and job execution API for the Video Translator "
            "pipeline. The dubbing pipeline itself is unchanged: a job runs it "
            "through the same generator the Gradio UI consumes, on a worker "
            "thread, with its progress published as server-sent events."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        # F2 adds writes: an upload is a POST, cancelling is a POST and
        # discarding an upload is a DELETE. Still an explicit list, never `*`.
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    install_error_handlers(app)

    app.include_router(system.router)
    app.include_router(uploads_routes.router)
    app.include_router(samples.router)
    app.include_router(jobs.router)

    # Serve the React build last. The static-assets mount is added here; the SPA
    # fallback is installed as middleware, so it sees a *finished* response and
    # only ever substitutes the shell for a genuine 404 on a frontend path.
    # Every API and documentation route above is matched first, and an API path
    # or a non-GET request is never a candidate, so this cannot change the
    # API's status codes (including 405) or its error envelope. When the build
    # is absent the app still starts: `/` explains how to build it.
    if settings.SERVE_FRONTEND:
        frontend.mount_frontend(app)
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """
    Startup maintenance and shutdown.

    On startup, stale staged uploads are swept (see `uploads.cleanup_stale`). This
    is the one piece of housekeeping that has to happen somewhere, and startup is
    the right place: a scheduler would be a thread to babysit for a rare case, and
    a client that uploads and abandons a file is exactly the case a restart
    eventually cleans up. Jobs are *not* auto-resumed here — a render that was
    interrupted is left `interrupted` until a user chooses Resume, because
    restarting someone's render without being asked is not the server's call.

    On shutdown, running jobs are asked to stop so they reach a pipeline
    checkpoint and write their durable state. Worker threads are daemons, so the
    process would exit without this; the difference is that a render is not
    abandoned mid-stage with its manifest still claiming `RUNNING`.
    """
    try:
        removed = await asyncio.to_thread(uploads.cleanup_stale)
        if removed:
            logger.info("Startup cleanup removed %d stale upload(s)", len(removed))
    except Exception:  # noqa: BLE001 - maintenance must never block startup
        logger.warning("Upload cleanup at startup failed", exc_info=True)

    yield

    try:
        from src.api.manager import get_manager
        get_manager().shutdown(timeout=5.0)
    except Exception:  # noqa: BLE001 - shutdown must not mask the exit
        logger.warning("Job manager shutdown failed", exc_info=True)


app = create_app()
