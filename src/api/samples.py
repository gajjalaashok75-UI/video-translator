"""
Bundled demo samples.

The repository ships a small amount of public-domain test media under
`samples/` (see `samples/README.md`). This module is the *only* way the API
reaches those files, and it does so through an explicit server-side allow-list:
a client names a sample by `id`, and every path is derived from the entry in
this file, never from client text.

Why an allow-list rather than a directory listing: `samples/` sits inside the
checkout next to source, configuration and credentials, so "serve whatever is in
this directory" would be a filesystem disclosure. A curated table means a file
only becomes reachable when someone adds it here on purpose, with a title and a
description that are safe to show.

Staging reuses the ordinary upload path. `stage()` copies a sample into the same
staging directory a browser upload lands in and returns the same `Upload`, so the
job that follows is created, claimed and run by the unchanged F2/F5 machinery —
there is no second job runner for demos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.api import media, uploads
from src.utils import config

logger = logging.getLogger(__name__)


class SampleError(Exception):
    """A sample could not be found, read or staged."""


class SampleNotFound(SampleError):
    """No allow-listed sample with that id."""


@dataclass(frozen=True)
class Sample:
    """
    One allow-listed bundled sample.

    `filename` is relative to the samples root and is supplied by this table,
    not by a client. `duration_seconds` is curated metadata used for display
    before a render; it is not re-probed on every listing, which keeps
    `GET /api/samples` cheap and offline-safe.
    """

    id: str
    filename: str
    title: str
    description: str
    source_language: str
    target_language: str
    duration_seconds: Optional[float]
    credit: str


# The allow-list. Adding a sample here is the only way to make it reachable.
_ALLOWLIST: tuple[Sample, ...] = (
    Sample(
        id="hgf_scene",
        filename="hgf_scene.webm",
        title="His Girl Friday — Sample Scene",
        description=(
            "A 96-second public-domain scene from His Girl Friday (1940), used "
            "throughout the project's own validation. Two speakers, English "
            "dialogue and a music bed, which makes it a realistic short dub."
        ),
        source_language="en",
        target_language="hi",
        duration_seconds=96.7,
        credit="His Girl Friday (1940) — public domain.",
    ),
)

_BY_ID: dict[str, Sample] = {sample.id: sample for sample in _ALLOWLIST}


def samples_root() -> Path:
    """
    The directory the allow-listed files live in.

    Resolved once from the project base directory. `API_SAMPLES_DIR` can
    override it for a deployment that ships the media elsewhere.
    """
    import os

    override = os.environ.get("API_SAMPLES_DIR", "").strip()
    if override:
        return Path(override).resolve()
    return (Path(config.BASE_DIR) / "samples").resolve()


def get(sample_id: str) -> Sample:
    """
    The allow-listed sample with this id, or raise `SampleNotFound`.

    The lookup is a dict hit, so an id that is not in the table — including a
    traversal attempt like `../../etc/passwd` — simply does not match. No client
    string is ever used as a path component.
    """
    sample = _BY_ID.get(str(sample_id or ""))
    if sample is None:
        raise SampleNotFound(f"Unknown sample: {sample_id!r}")
    return sample


def media_path(sample_id: str) -> Path:
    """
    The filesystem path of a sample's media, guaranteed to stay inside the root.

    The filename comes from the allow-list, but containment is still verified:
    an operator editing the table to an absolute or parent-relative path should
    fail loudly here rather than serve a file outside the samples directory.
    """
    sample = get(sample_id)
    root = samples_root()
    candidate = (root / sample.filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SampleError(
            f"Sample {sample_id!r} resolves outside the samples directory."
        ) from exc
    return candidate


def is_available(sample_id: str) -> bool:
    """Whether the sample's file is present and readable right now."""
    try:
        path = media_path(sample_id)
    except SampleError:
        return False
    return path.is_file()


def to_dict(sample: Sample) -> dict:
    """
    The public description of one sample.

    `available` is computed from the filesystem, so a checkout that omitted the
    media reports the sample as unavailable with a reason instead of advertising
    a demo that would fail.
    """
    path = None
    try:
        path = media_path(sample.id)
    except SampleError:
        path = None
    available = bool(path and path.is_file())
    reason = None
    if not available:
        reason = ("This sample's media is not present in the checkout "
                  "(samples/).")
    return {
        "id": sample.id,
        "title": sample.title,
        "description": sample.description,
        "filename": sample.filename,
        "duration_seconds": sample.duration_seconds,
        "source_language": sample.source_language,
        "target_language": sample.target_language,
        "credit": sample.credit,
        "size_bytes": path.stat().st_size if available and path else None,
        "media_type": _media_type(sample.filename),
        "available": available,
        "reason": reason,
    }


def list_samples() -> list[dict]:
    """Every allow-listed sample, in table order, with its availability."""
    return [to_dict(sample) for sample in _ALLOWLIST]


def _media_type(filename: str) -> str:
    """Content type for a sample, from its extension (never a fixed string)."""
    from src.api import artifacts

    return artifacts.media_type_for(Path(filename), "application/octet-stream")


def stage(sample_id: str) -> uploads.Upload:
    """
    Copy an allow-listed sample into the ordinary upload staging area.

    Returns a normal `Upload`, identical in shape to one produced by
    `POST /api/uploads`, so the client starts a demo with the same
    `POST /api/jobs` call it uses for an uploaded file. The sample is copied
    rather than referenced: the job claims and *moves* its source, so the
    bundled original must never be the file that moves.

    Raises `SampleNotFound` for an unknown id and `SampleError` when the media is
    missing or cannot be staged.
    """
    sample = get(sample_id)
    source = media_path(sample_id)
    if not source.is_file():
        raise SampleError(
            f"Sample {sample_id!r} is not available on this machine.")
    return uploads.stage_file(source, sample.filename)


def validate_media(sample_id: str) -> tuple[bool, str]:
    """
    Whether a sample is a usable video, and why not when it is not.

    Reuses the upload path's own check so a sample is held to exactly the same
    standard as a client upload.
    """
    try:
        path = media_path(sample_id)
    except SampleError as exc:
        return False, str(exc)
    if not path.is_file():
        return False, f"Sample {sample_id!r} is not available on this machine."
    return media.validate_upload(path)
