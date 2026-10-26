"""
Staged uploads.

An upload is a two-step handshake: the client streams a file to
`POST /api/uploads`, and a later `POST /api/jobs` *claims* it. Between those two
steps the file sits in a staging directory and the only handle to it is an
opaque upload id. The client never supplies a path, and the id is validated
before it is ever joined to a directory, so a request cannot reach an arbitrary
file.

The three states an upload can be in:

    staged   -> written and probed, waiting to be claimed
    claimed  -> moved into a job's `source/` directory; the id is now spent
    deleted  -> removed (the client failed, or cleanup ran)

Single-use by design: claiming an upload marks it claimed and moves the file, so
a second job cannot silently reuse the same source. If the caller needs to run
the same video twice, they upload it twice — that keeps "which file did this job
run" answerable from the job alone.

The id is derived from a random UUID and carries no client-supplied text, which
is what makes it safe to treat as a filename component. A registry file per
upload records the probe result and the claim, so a staging directory left over
from a previous process is still understandable.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.api import settings
from src.core import jobs as joblib

logger = logging.getLogger(__name__)

# Metadata filename inside each upload's staging directory.
_META_NAME = "upload.json"

# Status values.
STATUS_STAGED = "staged"
STATUS_CLAIMED = "claimed"
STATUS_DELETED = "deleted"


class UploadError(Exception):
    """An upload could not be staged, looked up or claimed."""


class UploadNotFound(UploadError):
    """No upload with that id (or it has been removed)."""


class UploadAlreadyClaimed(UploadError):
    """The upload has already been handed to a job."""


def is_valid_upload_id(upload_id: str) -> bool:
    """
    Whether `upload_id` is a plain name that cannot traverse or hide.

    The same rule as a job id: it must start with an alphanumeric and contain
    only `[A-Za-z0-9._-]`, so `..`, `/` and a leading dot are all impossible.
    """
    return joblib.is_valid_job_id(upload_id)


def new_upload_id() -> str:
    """An opaque, filesystem-safe upload id. Carries no client text."""
    return f"up-{uuid.uuid4().hex}"


@dataclass
class Upload:
    """One staged upload and its probe result."""

    upload_id: str
    root: Path
    filename: str
    bytes: int = 0
    media: dict = field(default_factory=dict)
    status: str = STATUS_STAGED
    created_at: float = 0.0
    claimed_by: Optional[str] = None
    source_path: Optional[str] = None

    @property
    def path(self) -> Path:
        """The staged media file."""
        return self.root / self.filename

    def to_dict(self) -> dict:
        return {
            "upload_id": self.upload_id,
            "filename": self.filename,
            "bytes": self.bytes,
            "media": dict(self.media),
            "status": self.status,
            "created_at": self.created_at,
            "claimed_by": self.claimed_by,
        }


def uploads_root() -> Path:
    """The staging root, created on demand."""
    root = Path(settings.UPLOADS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    # Keep staged media out of `git status`. The repository tracks `temp/` for
    # its own evidence, but uploads are runtime traffic, not evidence.
    marker = root / ".gitignore"
    if not marker.exists():
        try:
            marker.write_text("*\n!.gitignore\n", encoding="utf-8")
        except OSError:
            pass
    return root


def _safe_extension(filename: str) -> str:
    """
    The lower-cased extension of a client-supplied filename, or "".

    Only the final suffix is considered, and it is validated against the
    allow-list by the caller. `Path.suffix` on a name containing a path
    separator would still return something, so the name is reduced to its final
    component first.
    """
    base = Path(str(filename or "")).name
    return Path(base).suffix.lower()


def safe_filename(filename: str) -> str:
    """
    A safe *display* name for a client-supplied filename.

    Reduced to a single path component with everything outside
    `[A-Za-z0-9._-]` collapsed, so a name like `../../etc/passwd` becomes
    `etc_passwd`. This is what is stored and echoed; the extension is preserved
    so the pipeline and FFmpeg still see the right container.
    """
    base = Path(str(filename or "")).name
    cleaned = joblib.safe_name(base, max_length=120)
    if not cleaned:
        cleaned = "upload"
    # `safe_name` strips leading dots, so a name that was only an extension
    # ("..mp4" -> "mp4") loses its dot. Put it back when the original had one.
    if base.startswith(".") and not cleaned.startswith(".") and "." in base:
        cleaned = base[:1] + cleaned
    return cleaned


def validate_extension(filename: str) -> str:
    """
    The accepted extension for `filename`, or raise `UploadError`.

    Checked before a byte is written, so an obviously wrong upload is refused
    without spending disk on it.
    """
    ext = _safe_extension(filename)
    if not ext:
        raise UploadError("Uploaded file has no extension; expected a video "
                          "file such as .mp4 or .mkv")
    if ext not in settings.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(settings.ALLOWED_EXTENSIONS))
        raise UploadError(f"Unsupported file type {ext!r}; allowed: {allowed}")
    return ext


def staging_dir(upload_id: str) -> Path:
    """
    The staging directory for `upload_id`, guaranteed to stay inside the root.

    The id is validated first, so a traversal attempt fails here rather than
    being resolved.
    """
    if not is_valid_upload_id(upload_id):
        raise UploadError(f"Invalid upload id: {upload_id!r}")
    root = uploads_root()
    candidate = (root / upload_id).resolve()
    if candidate.parent != root.resolve():
        raise UploadError(f"Invalid upload id: {upload_id!r}")
    return candidate


def write_meta(upload: Upload) -> None:
    """Persist the upload's metadata, atomically."""
    try:
        joblib.atomic_write_json(upload.root / _META_NAME, upload.to_dict())
    except OSError as exc:
        logger.warning("Could not write upload metadata for %s: %s",
                       upload.upload_id, exc)


def load(upload_id: str) -> Upload:
    """
    Read a staged upload's metadata.

    Raises `UploadNotFound` when the id is unknown, malformed or its staging
    directory is gone, so callers cannot distinguish "never existed" from
    "cleaned up" — both are simply not there.
    """
    try:
        directory = staging_dir(upload_id)
    except UploadError:
        raise UploadNotFound(f"Unknown upload: {upload_id!r}")
    meta_path = directory / _META_NAME
    data = joblib.read_json(meta_path)
    if not isinstance(data, dict):
        raise UploadNotFound(f"Unknown upload: {upload_id!r}")
    filename = str(data.get("filename") or "")
    if not filename:
        raise UploadNotFound(f"Upload {upload_id!r} has no recorded filename")
    return Upload(
        upload_id=upload_id,
        root=directory,
        filename=filename,
        bytes=int(data.get("bytes") or 0),
        media=dict(data.get("media") or {}),
        status=str(data.get("status") or STATUS_STAGED),
        created_at=float(data.get("created_at") or 0.0),
        claimed_by=data.get("claimed_by"),
        source_path=data.get("source_path"),
    )


def begin(upload_id: str, filename: str) -> Upload:
    """
    Create a fresh staging directory for an upload that is about to be written.

    Raises if the directory already exists, so an id collision (astronomically
    unlikely with a UUID) can never overwrite someone else's file.
    """
    directory = staging_dir(upload_id)
    if directory.exists():
        raise UploadError(f"Upload id already in use: {upload_id!r}")
    directory.mkdir(parents=True)
    return Upload(
        upload_id=upload_id,
        root=directory,
        filename=filename,
        created_at=time.time(),
        status=STATUS_STAGED,
    )


def stage_file(source: Path, filename: str) -> Upload:
    """
    Stage an existing local file as if it had been uploaded.

    Used for the bundled demo samples. The file is *copied* into the staging
    directory, never moved, so the bundled original survives the job that later
    claims and moves the staged copy. The result is an ordinary `Upload`, probed
    and validated exactly as a client upload is, so the job created from it
    cannot tell the difference — there is no separate path for demos.

    Raises `UploadError` if the file cannot be staged or is not a usable video.
    """
    from src.api import media  # local: keeps this module free of a probe import

    display_name = safe_filename(filename)
    validate_extension(display_name)
    upload = begin(new_upload_id(), display_name)
    try:
        shutil.copy2(source, upload.path)
        upload.bytes = upload.path.stat().st_size
        ok, reason = media.validate_upload(upload.path)
        if not ok:
            raise UploadError(reason)
        upload.media = media.probe_media(upload.path)
        write_meta(upload)
    except UploadError:
        discard(upload)
        raise
    except OSError as exc:
        discard(upload)
        raise UploadError(f"Could not stage {filename!r}: {exc}") from exc
    logger.info("Staged sample %s as upload %s", filename, upload.upload_id)
    return upload


def discard(upload: Upload) -> None:
    """
    Remove a staging directory, best-effort.

    Called when a stream fails part-way, so a rejected or interrupted upload
    leaves no partial file behind.
    """
    try:
        shutil.rmtree(upload.root, ignore_errors=True)
    except OSError as exc:
        logger.warning("Could not remove staging dir %s: %s", upload.root, exc)


def claim(upload_id: str, job_root: Path) -> Path:
    """
    Move a staged upload into `job_root/source/` and mark it claimed.

    Returns the destination path. The id is validated and the upload must be
    `staged`: a second claim of the same upload raises
    `UploadAlreadyClaimed`, which is what stops two jobs sharing one source
    file.

    The move is a rename when possible and a copy otherwise (staging and jobs
    normally share a filesystem, but not necessarily). The metadata is updated
    last, so a crash mid-move leaves an unclaimed upload rather than a job that
    believes it has a source it does not.
    """
    upload = load(upload_id)
    if upload.status == STATUS_CLAIMED:
        raise UploadAlreadyClaimed(
            f"Upload {upload_id!r} was already claimed by job "
            f"{upload.claimed_by!r}")
    if upload.status == STATUS_DELETED:
        raise UploadNotFound(f"Upload {upload_id!r} has been removed")

    source_dir = Path(job_root) / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    destination = source_dir / upload.filename
    staged = upload.path
    if not staged.is_file():
        raise UploadNotFound(
            f"Upload {upload_id!r} has no staged file at {upload.filename!r}")

    if destination.exists():
        # A job directory is fresh, so this means the same filename was claimed
        # twice into one job; disambiguate rather than overwrite.
        destination = source_dir / f"{upload_id[:8]}_{upload.filename}"

    try:
        os.replace(staged, destination)
    except OSError:
        shutil.copy2(staged, destination)
        staged.unlink(missing_ok=True)

    upload.status = STATUS_CLAIMED
    upload.claimed_by = job_root.name
    upload.source_path = str(destination)
    write_meta(upload)
    logger.info("Upload %s claimed into %s", upload_id, destination)
    return destination


def release(upload_id: str) -> None:
    """
    Undo a claim's *metadata* side after the job it was claimed for failed to be
    created, so the upload can be claimed again.

    The file has already moved into the (now abandoned) job directory; this only
    makes the upload claimable again. It is deliberately narrow: F2's job
    creation is ordered so this is rarely needed.
    """
    try:
        upload = load(upload_id)
    except UploadNotFound:
        return
    upload.status = STATUS_STAGED
    upload.claimed_by = None
    upload.source_path = None
    write_meta(upload)


# ---------------------------------------------------------------------------
# Lifecycle: TTL cleanup
# ---------------------------------------------------------------------------
def iter_staged() -> list["Upload"]:
    """
    Every upload still in the staging root, newest first.

    A staging directory whose metadata is unreadable is skipped rather than
    guessed at: it may be a write in progress, and deleting it on a guess is
    worse than leaving a few kilobytes behind.
    """
    root = uploads_root()
    if not root.is_dir():
        return []
    found: list[Upload] = []
    for entry in root.iterdir():
        if not entry.is_dir() or not is_valid_upload_id(entry.name):
            continue
        try:
            found.append(load(entry.name))
        except UploadNotFound:
            continue
    found.sort(key=lambda u: u.created_at, reverse=True)
    return found


def cleanup_stale(ttl_seconds: Optional[float] = None,
                  now: Optional[float] = None) -> list[str]:
    """
    Remove staged uploads older than the TTL. Returns the ids removed.

    Only `staged` uploads are eligible, and that is the whole safety rule: an
    upload that has been *claimed* is a job's source file that has already moved
    into the job directory, and deleting it would destroy a render's input. The
    check is on the recorded status, so a claimed upload is skipped even if its
    staging directory somehow still exists.

    Called at startup (see `main._lifespan`) rather than by a scheduler: a
    periodic sweep would be a background thread to babysit for a case — a client
    that uploads and never creates a job — that is rare and not urgent. Startup
    is enough to stop the staging directory growing without bound across restarts,
    and `cleanup_stale` can be called again by a maintenance route if one is added.

    `ttl_seconds` defaults to `settings.UPLOAD_TTL_SECONDS`. A value of zero or
    less disables the sweep, which is how a deployment that wants to keep every
    upload opts out.
    """
    ttl = settings.UPLOAD_TTL_SECONDS if ttl_seconds is None else float(ttl_seconds)
    if ttl <= 0:
        return []
    reference = time.time() if now is None else float(now)
    removed: list[str] = []
    for upload in iter_staged():
        if upload.status == STATUS_CLAIMED:
            # A job owns this file now. Never touch it, whatever the clock says.
            continue
        age = reference - float(upload.created_at or 0.0)
        if age < ttl:
            continue
        discard(upload)
        removed.append(upload.upload_id)
    if removed:
        logger.info("Upload cleanup removed %d stale upload(s)", len(removed))
    return removed
