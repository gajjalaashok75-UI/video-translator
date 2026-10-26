"""
Production static serving for the React build.

In production there is one process and one origin: FastAPI serves the built
frontend *and* `/api/*`. That removes the second server (the Vite dev server)
from the deployment and the CORS entry it needed. Development is unchanged —
Vite still runs on its own port and proxies `/api` here (see the frontend
README) — because this module only takes effect when a build is present and the
process is not told to stay API-only.

Two namespaces share the origin, and they must not blur:

    /api/*          always the API. A missing `/api/...` path is an API 404,
                    never the React shell — otherwise a typo'd request would
                    return 200 with an HTML body and the client's JSON parse
                    would fail with a confusing message instead of a clear
                    "no such endpoint".
    /docs, /redoc,  FastAPI's own routes. They keep working (or stay
    /openapi.json   deliberately disabled) rather than being swallowed.
    a known client  the React app. Real build files are served as themselves,
    route           and a route the client declares (`/jobs/<id>/result`, …)
                    returns `index.html` so `BrowserRouter` can resolve it on a
                    cold load or a refresh.
    anything else   a plain 404. This matters: the SPA fallback is deliberately
                    *not* a blanket "unknown path -> shell". A blanket fallback
                    would answer `GET /etc/passwd/media` with 200 and an HTML
                    body, which is exactly the signal the traversal tests exist
                    to forbid, and it would turn a mistyped URL into a
                    successful-looking response. Only routes the client actually
                    declares get the shell.

The interception is done at the ASGI layer, wrapping the finished application,
rather than by adding a catch-all *route*. A catch-all route changes routing
semantics for every path: `@app.get("/{path}")` makes an unmatched path a
*partial* match for every other method, so `POST /api/jobs/<path-like-id>/resume`
would come back `405` instead of the API's own `404`, and `PUT /api/jobs/<id>`
would lose its `405`. Wrapping instead lets Starlette route exactly as it did
before and only substitutes a response for a genuine `404` on a path the
frontend owns.

The build directory is a *build artifact*, read-only to this process and not
committed (see `frontend/.gitignore`). If it is absent the app still starts and
serves the API; `/` then answers with a short page that says how to build the
frontend, rather than a bare 404.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.api import settings

logger = logging.getLogger(__name__)

# Backend namespaces the SPA fallback must never answer. `/api` covers every
# data, media, download, artifact and SSE route; the other three are FastAPI's
# own documentation routes. Kept here as data so the test suite can assert the
# list rather than re-deriving it.
PROTECTED_PREFIXES: tuple[str, ...] = ("/api", "/docs", "/redoc", "/openapi.json")

# The client routes that may resolve to the shell on a cold load. Kept in sync
# with `frontend/src/App.tsx`; a route missing here still works *within* the app
# (client-side navigation never asks the server) but a direct load or refresh of
# it would 404. Each entry matches itself and any deeper path, so `/jobs` covers
# `/jobs/<id>` and `/jobs/<id>/result`. `/` matches the root only.
CLIENT_ROUTES: tuple[str, ...] = (
    "/", "/dashboard", "/languages", "/jobs", "/dub", "/demo",
)

# Where the build's hashed assets live. Vite writes everything with a content
# hash under this directory, so a mount here can cache aggressively and stream
# byte ranges without this module re-implementing file serving.
ASSETS_DIRNAME = "assets"

_MISSING_BUILD_PAGE = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Video Translator — frontend not built</title></head>
<body style="font-family:system-ui,sans-serif;max-width:40rem;margin:4rem auto;padding:0 1rem;color:#101218">
<h1>Frontend production build not found.</h1>
<p>The API is running, but the React application has not been built yet, so
there is nothing to serve at <code>/</code>.</p>
<p>Build it once:</p>
<pre style="background:#f4f5f8;padding:1rem;border-radius:.5rem">cd frontend
npm install
npm run build</pre>
<p>Then reload this page. The API itself is available under <code>/api</code>,
and its schema at <code>/docs</code>.</p>
</body>
</html>
"""


def frontend_dist() -> Path:
    """The build directory, from settings (env-overridable)."""
    return settings.FRONTEND_DIST


def build_is_present(dist: Path | None = None) -> bool:
    """Whether a usable build (`index.html`) exists at `dist`."""
    dist = dist if dist is not None else frontend_dist()
    return (dist / "index.html").is_file()


def _safe_file(dist: Path, url_path: str) -> Path | None:
    """
    Resolve a URL path to a regular file inside `dist`, or None.

    This is the one place a client string could become a filesystem path, so it
    is deliberately strict: the path is joined to `dist`, resolved, and then
    confirmed to still be *inside* `dist`. A `..` segment, an absolute path or a
    symlink that escapes the build directory all fail the containment check and
    return None.
    """
    relative = url_path.lstrip("/")
    if not relative:
        return None
    try:
        candidate = (dist / relative).resolve()
        root = dist.resolve()
    except (OSError, ValueError):
        return None
    if candidate != root and root not in candidate.parents:
        return None
    return candidate if candidate.is_file() else None


def _is_protected(path: str) -> bool:
    """Whether a URL path belongs to a backend namespace."""
    for prefix in PROTECTED_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def _is_client_route(path: str) -> bool:
    """
    Whether `path` is a route the React app declares.

    `/` is exact (it is the shell itself, not a prefix of everything); every
    other entry matches itself and any deeper path, so `/jobs` covers
    `/jobs/<id>` and `/jobs/<id>/result`. Only these paths are answered with the
    shell, which is what keeps an unknown path a real 404 instead of a
    successful-looking HTML response.
    """
    for route in CLIENT_ROUTES:
        if route == "/":
            if path == "/":
                return True
        elif path == route or path.startswith(route + "/"):
            return True
    return False


class FrontendFallback:
    """
    ASGI wrapper: answer a frontend path the API would otherwise 404 on.

    The wrapped app is called first and routes normally. Only a `404` on a
    frontend path is replaced:

        a real build file   served as itself (favicon, robots.txt, …)
        a client route      `index.html`, so `BrowserRouter` can resolve a deep
                            link or a refresh
        no build present    an actionable "build the frontend" page
        anything else       the app's own 404, reproduced unchanged

    A non-404 response (a real API reply, a `405` from Starlette, a redirect) is
    forwarded byte-for-byte, and a backend path or a non-GET method is never a
    candidate at all, so the API's behaviour is exactly what it was without this
    wrapper.
    """

    def __init__(self, app, dist: Path | None = None):
        self.app = app
        self.dist = Path(dist) if dist is not None else None

    @property
    def _dist(self) -> Path:
        return self.dist if self.dist is not None else frontend_dist()

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")
        method = scope.get("method", "GET")

        # Only a GET/HEAD navigation to a frontend path is a candidate; a
        # backend path never is. Everything else is forwarded with no
        # interception, which is what preserves the API's exact status codes
        # (including Starlette's 405) and error envelope.
        if method not in ("GET", "HEAD") or _is_protected(path):
            await self.app(scope, receive, send)
            return

        intercepted = False

        async def send_wrapper(message):
            nonlocal intercepted
            if (message["type"] == "http.response.start"
                    and int(message.get("status", 0)) == 404):
                # The app has nothing for this path. Swallow its 404 (and the
                # body that follows) and answer it below.
                intercepted = True
                return
            if intercepted:
                return
            await send(message)

        await self.app(scope, receive, send_wrapper)

        if not intercepted:
            return

        dist = self._dist
        served = _safe_file(dist, path)
        if served is not None:
            await _run(FileResponse(served), scope, receive, send)
            return

        if _is_client_route(path):
            index = dist / "index.html"
            if index.is_file():
                await _run(FileResponse(index), scope, receive, send)
            else:
                await _run(HTMLResponse(_MISSING_BUILD_PAGE, status_code=404),
                           scope, receive, send)
            return

        # Not a frontend path: reproduce the app's own 404 so the API contract
        # (JSON envelope, correct status) is unchanged.
        await _run(_not_found(path), scope, receive, send)


async def _run(response: Response, scope, receive, send) -> None:
    """Run a Starlette `Response` as a bare ASGI app, with the real scope."""
    await response(scope, receive, send)


def _not_found(path: str) -> Response:
    """The API error envelope for an unmatched path."""
    from src.api.errors import ErrorCode, error_body

    return JSONResponse(
        status_code=404,
        content=error_body(ErrorCode.NOT_FOUND, f"No route for {path!r}."),
    )


def mount_frontend(app: FastAPI, dist: Path | None = None) -> bool:
    """
    Serve the React build from `app`. Returns whether a build was mounted.

    A mount at `/assets` serves the hashed bundle directly (Starlette's
    `StaticFiles` streams it and answers byte ranges). Everything else goes
    through `FrontendFallback`, added as middleware so it sees the finished
    response — including the error handler's envelope — and decides whether to
    substitute the shell.

    When no build is present, `/` answers with an actionable page instead of
    `index.html`; the API routers are untouched in either case, so an API-only
    deployment (or a backend unit-test run that never built React) is unaffected.
    """
    dist = dist if dist is not None else frontend_dist()
    present = (dist / "index.html").is_file()

    if present and (dist / ASSETS_DIRNAME).is_dir():
        # check_dir=False: the directory exists (we just checked), and the
        # default raises at import if it ever does not, which would turn a
        # missing build into a crash rather than the fallback below.
        app.mount(
            f"/{ASSETS_DIRNAME}",
            StaticFiles(directory=str(dist / ASSETS_DIRNAME)),
            name="frontend-assets",
        )

    app.add_middleware(FrontendFallback, dist=dist)

    if present:
        logger.info("Serving frontend build from %s", dist)
    else:
        logger.warning(
            "No frontend build at %s; serving API only. Run "
            "`cd frontend && npm install && npm run build` to build it.",
            dist,
        )
    return present
