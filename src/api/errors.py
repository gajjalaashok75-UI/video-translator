"""
The API error contract.

Every failure the API returns, from any phase, is shaped the same way:

    {"error": {"code": "JOB_NOT_RESUMABLE", "message": "...", "field": null}}

`code` is the machine-readable part and is stable: a client switches on it.
`message` is for a human and may change. That split is the whole point — a
frontend must not parse prose to decide what to do, and it must not need a
different parser per endpoint.

The handler is installed for `HTTPException` too, so an error raised deep inside
a route (or by Starlette itself, for an unknown path or a wrong method) comes out
in the same shape rather than as FastAPI's default `{"detail": ...}`. A route
should therefore raise `ApiError` when it wants a specific code, and a bare
`HTTPException` when any code will do — the handler assigns one from the status.

Nothing here ever includes a traceback. An unexpected exception is logged in
full on the server and reported to the client as a generic 500, because an
exception message can carry a token, a filesystem path or a model URL.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """
    The machine-readable error vocabulary.

    Grouped by what a client would do about it: fix the request, pick another
    job, wait, or retry later.
    """

    # The request itself is wrong.
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_SETTINGS = "INVALID_SETTINGS"
    INVALID_JOB_ID = "INVALID_JOB_ID"
    INVALID_UPLOAD = "INVALID_UPLOAD"
    UPLOAD_TOO_LARGE = "UPLOAD_TOO_LARGE"

    # The named thing does not exist.
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    UPLOAD_NOT_FOUND = "UPLOAD_NOT_FOUND"
    ARTIFACT_UNAVAILABLE = "ARTIFACT_UNAVAILABLE"
    OUTPUT_UNAVAILABLE = "OUTPUT_UNAVAILABLE"
    NOT_FOUND = "NOT_FOUND"

    # The thing exists but is in the wrong state for this operation.
    JOB_ALREADY_ACTIVE = "JOB_ALREADY_ACTIVE"
    JOB_NOT_RESUMABLE = "JOB_NOT_RESUMABLE"
    JOB_ACTIVE = "JOB_ACTIVE"
    UPLOAD_ALREADY_CLAIMED = "UPLOAD_ALREADY_CLAIMED"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"

    # Try again later.
    TOO_MANY_SUBSCRIBERS = "TOO_MANY_SUBSCRIBERS"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"

    # A bug. The client can only report it.
    INTERNAL_ERROR = "INTERNAL_ERROR"


# HTTP status -> code, for an error raised without naming one.
_CODE_BY_STATUS = {
    400: ErrorCode.INVALID_REQUEST,
    401: ErrorCode.INVALID_REQUEST,
    403: ErrorCode.INVALID_REQUEST,
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.METHOD_NOT_ALLOWED,
    409: ErrorCode.INVALID_REQUEST,
    413: ErrorCode.UPLOAD_TOO_LARGE,
    422: ErrorCode.INVALID_SETTINGS,
    429: ErrorCode.SERVICE_UNAVAILABLE,
    503: ErrorCode.SERVICE_UNAVAILABLE,
}


class ApiError(Exception):
    """
    An error a route raises deliberately.

    `field` names the offending input when there is one, so a form can highlight
    it. `details` carries structured extras a client may want (a job's current
    state, a queue position) without inventing a second response shape.
    """

    def __init__(self, status_code: int, code: ErrorCode, message: str,
                 field: Optional[str] = None,
                 details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = code
        self.message = message
        self.field = field
        self.details = details

    def to_body(self) -> dict:
        error: dict[str, Any] = {
            "code": self.code.value,
            "message": self.message,
        }
        if self.field is not None:
            error["field"] = self.field
        if self.details:
            error["details"] = self.details
        return {"error": error}


def error_body(code: ErrorCode | str, message: str,
               field: Optional[str] = None,
               details: Optional[dict] = None) -> dict:
    """Build the error envelope by hand, for a handler that has no ApiError."""
    value = code.value if isinstance(code, ErrorCode) else str(code)
    error: dict[str, Any] = {"code": value, "message": message}
    if field is not None:
        error["field"] = field
    if details:
        error["details"] = details
    return {"error": error}


def _status_of(exc: StarletteHTTPException) -> int:
    return int(getattr(exc, "status_code", 500))


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_body())


async def http_exception_handler(request: Request,
                                 exc: StarletteHTTPException) -> JSONResponse:
    """
    Re-shape a plain `HTTPException` into the envelope.

    A route that raised `HTTPException(404, "No job 'x' on disk.")` gets
    `{"error": {"code": "NOT_FOUND", "message": "No job 'x' on disk."}}`. The
    detail may be a dict (FastAPI allows it), in which case a `message` key is
    lifted out and the rest is preserved as `details`.
    """
    status = _status_of(exc)
    detail = getattr(exc, "detail", None)
    field = None
    details = None
    if isinstance(detail, dict):
        message = str(detail.get("message") or detail.get("detail") or "")
        field = detail.get("field")
        rest = {k: v for k, v in detail.items()
                if k not in ("message", "detail", "field")}
        details = rest or None
    else:
        message = str(detail) if detail is not None else ""
    if not message:
        message = "The request could not be completed."
    code = _CODE_BY_STATUS.get(status, ErrorCode.INTERNAL_ERROR)
    return JSONResponse(
        status_code=status,
        content=error_body(code, message, field=field, details=details),
        headers=getattr(exc, "headers", None),
    )


async def validation_exception_handler(request: Request,
                                       exc: RequestValidationError) -> JSONResponse:
    """
    FastAPI's own request-validation failure, in the same envelope.

    The default body is a list of pydantic errors, which is a second shape a
    client would have to learn. The first error's location becomes `field` and
    its message becomes `message`; the full list is kept under `details` so
    nothing is lost.
    """
    errors = exc.errors() if hasattr(exc, "errors") else []
    first = errors[0] if errors else {}
    location = [str(part) for part in (first.get("loc") or [])]
    field = ".".join(location[1:]) if len(location) > 1 else (
        location[0] if location else None)
    message = str(first.get("msg") or "The request body is not valid.")
    return JSONResponse(
        status_code=422,
        content=error_body(ErrorCode.INVALID_REQUEST, message, field=field,
                           details={"errors": errors} if errors else None),
    )


async def unhandled_exception_handler(request: Request,
                                      exc: Exception) -> JSONResponse:
    """
    The catch-all.

    Logs the real exception with its traceback on the server, and tells the
    client only that something broke. The request path is included because it is
    the one piece of context that helps a user report the problem and is already
    public to them.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=error_body(
            ErrorCode.INTERNAL_ERROR,
            "The server could not complete this request. It has been logged.",
        ),
    )


def install_error_handlers(app: FastAPI) -> None:
    """Register the handlers. Order matters only for exact-type matching."""
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
