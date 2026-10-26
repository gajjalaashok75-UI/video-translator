"""
The artifact allow-list.

A job directory holds far more than a client should see: the separation stems,
per-segment TTS audio, the fingerprint cache, the model scratch files and the
speaker reference recordings. None of that is an API surface, and a recursive
listing of the directory would publish all of it.

So the set is enumerated here instead, by hand, and a request names one of these
entries rather than a path. `resolve` is a dictionary lookup, never a join: there
is no string from the client that is ever concatenated into a filesystem path, so
there is nothing to traverse. Each candidate is additionally confirmed to be a
regular file inside the job root before it is served, so a manifest that names
`/etc/passwd` as its output is refused rather than obeyed.

Speaker reference *audio* is deliberately absent. The profiles under
`speaker_refs/profiles/` are recordings of real people; exposing them by default
would publish voice data that a client did not ask for and cannot use. The JSON
that describes the speakers is offered instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.core import jobs as joblib
from src.core.manifest import Manifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactSpec:
    """
    One allow-listed artifact.

    `candidates` are paths relative to the job root, tried in order; the first
    that exists is used. A `glob` in a candidate is matched with `sorted()` so
    the choice is deterministic rather than filesystem-order dependent.
    """

    name: str
    type: str
    label: str
    candidates: tuple[str, ...]
    media_type: str
    description: str = ""
    # The key the pipeline records this artifact under in the manifest. When
    # present, the recorded path wins over the glob candidates: it is the file
    # the render actually wrote, so a job that produced two similarly named
    # files (a re-render, a chunked run) resolves to the current one rather than
    # whichever sorts first.
    manifest_key: Optional[str] = None


# The allow-list. Order is the order the API lists them in, which is roughly the
# order a person wants to look at them.
SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec(
        name="final_video",
        type="video",
        label="Dubbed video",
        # The pipeline records this in the manifest; the fallback covers a
        # manifest whose artifact entry was lost but whose file survives.
        candidates=("final/*.mp4", "final/*.webm", "final/*.mkv"),
        media_type="video/mp4",
        description="The finished dubbed video, if the render completed.",
        manifest_key="final_output",
    ),
    ArtifactSpec(
        name="subtitles",
        type="subtitle",
        label="Subtitles (SRT)",
        # SRT is what an editor imports; the WebVTT sibling is a separate entry
        # below, because a `<track>` needs the VTT MIME type and the two files
        # must not be confused for one another.
        candidates=("subtitles/*.srt",),
        media_type="application/x-subrip",
        description="Target-language subtitle track, timed to the dubbed audio.",
        manifest_key="subtitles_srt",
    ),
    ArtifactSpec(
        name="subtitles_vtt",
        type="subtitle",
        label="Subtitles (WebVTT)",
        candidates=("subtitles/*.vtt",),
        media_type="text/vtt",
        description="The same subtitle track as WebVTT, for in-browser playback.",
        manifest_key="subtitles_vtt",
    ),
    ArtifactSpec(
        name="audio_mixed",
        type="audio",
        label="Final mixed audio",
        candidates=("final/*_audio_mixed.wav",),
        media_type="audio/wav",
        description="The full dub mix: dialogue over the original background.",
        manifest_key="audio_mixed",
    ),
    ArtifactSpec(
        name="audio_dialogue",
        type="audio",
        label="Dialogue-only audio",
        candidates=("final/*_audio_dialogue.wav",),
        media_type="audio/wav",
        description="The generated speech alone, with no background, for re-cutting.",
        manifest_key="audio_dialogue",
    ),
    ArtifactSpec(
        name="audio_background",
        type="audio",
        label="Background audio",
        candidates=("final/*_audio_background.wav",),
        media_type="audio/wav",
        description="Music and effects with the original speech removed.",
        manifest_key="audio_background",
    ),
    ArtifactSpec(
        name="translated_transcript",
        type="transcript",
        label="Translated transcript",
        candidates=("translation/segments.json",),
        media_type="application/json",
        description="Per-segment translated text with its timing.",
    ),
    ArtifactSpec(
        name="transcript",
        type="transcript",
        label="Source transcript",
        candidates=("transcription/segments.json",),
        media_type="application/json",
        description="Per-segment source text produced by transcription.",
    ),
    ArtifactSpec(
        name="diarization",
        type="json",
        label="Speaker segments",
        candidates=("diarization/segments.json",),
        media_type="application/json",
        description="Which speaker was talking when.",
    ),
    ArtifactSpec(
        name="speakers",
        type="json",
        label="Speaker metadata",
        # The description of the speakers, not the recordings.
        candidates=("speaker_refs/references.json",),
        media_type="application/json",
        description="Speaker labels and reference metadata (not the recordings).",
    ),
    ArtifactSpec(
        name="report",
        type="json",
        label="Job report",
        candidates=("manifest.json",),
        media_type="application/json",
        description="The durable job manifest: stages, timings and diagnostics.",
    ),
)

SPECS_BY_NAME: dict[str, ArtifactSpec] = {spec.name: spec for spec in SPECS}

# Artifact names that exist but are withheld, so an API response can explain the
# refusal rather than claim the file does not exist.
WITHHELD: dict[str, str] = {
    "speaker_references": (
        "Speaker reference recordings are not published through the API."),
    "cache": "Pipeline cache internals are not published through the API.",
    "tts": "Per-segment audio is an implementation detail, not an artifact.",
    "separation": ("Separation stems are intermediates; the background track is "
                   "published as `audio_background`."),
}


def _job_root(job_id: str) -> Optional[Path]:
    if not joblib.is_valid_job_id(job_id):
        return None
    try:
        job = joblib.Job.load(job_id)
    except joblib.JobError:
        return None
    return job.root


def _contained(path: Path, root: Path) -> bool:
    """True when `path` resolves to a real file inside `root`."""
    try:
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return resolved.is_file()


def _manifest(job_id: str) -> Optional[Manifest]:
    try:
        job = joblib.Job.load(job_id)
    except joblib.JobError:
        return None
    return Manifest.load(job)


def resolve(job_id: str, name: str) -> Optional[Path]:
    """
    The file for an allow-listed artifact, or None.

    `name` is looked up, never joined. A name that is not in the allow-list is
    not a path to try — it is simply unknown, and the caller reports 404.
    """
    spec = SPECS_BY_NAME.get(name)
    if spec is None:
        return None
    root = _job_root(job_id)
    if root is None:
        return None

    # A manifest-recorded path is authoritative when it points at a contained
    # file; the glob candidates are the fallback for a manifest whose artifact
    # entry was lost while the file survived.
    if spec.manifest_key:
        manifest = _manifest(job_id)
        recorded = (manifest.artifacts or {}).get(spec.manifest_key) if manifest else None
        if recorded:
            candidate = Path(str(recorded))
            if _contained(candidate, root):
                return candidate

    for pattern in spec.candidates:
        if "*" in pattern:
            matches = sorted(root.glob(pattern))
        else:
            candidate = root / pattern
            matches = [candidate] if candidate.exists() else []
        for match in matches:
            if _contained(match, root):
                return match
    return None


def describe(job_id: str, name: str) -> Optional[dict]:
    """
    A descriptor for one artifact, or None when the file is absent.

    `available` is always true in the returned dict: an unavailable artifact is
    reported by the list as a descriptor with `available: false` and no URL,
    rather than being omitted, so a client can render a disabled control instead
    of guessing why a name is missing.
    """
    spec = SPECS_BY_NAME.get(name)
    if spec is None:
        return None
    path = resolve(job_id, name)
    if path is None:
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    return {
        "name": spec.name,
        "type": spec.type,
        "label": spec.label,
        "filename": path.name,
        "available": True,
        "size_bytes": size,
        "media_type": media_type_for(path, spec.media_type),
        "download_url": f"/api/jobs/{job_id}/artifacts/{spec.name}",
        "description": spec.description,
    }


def list_for(job_id: str) -> list[dict]:
    """
    Every allow-listed artifact, present or not.

    Names that are withheld are listed with `available: false` and a `reason`,
    so a client can distinguish "this render produced no subtitles" from "this
    API does not publish that".
    """
    out: list[dict] = []
    for spec in SPECS:
        descriptor = describe(job_id, spec.name)
        if descriptor is not None:
            out.append(descriptor)
            continue
        out.append({
            "name": spec.name,
            "type": spec.type,
            "label": spec.label,
            "filename": None,
            "available": False,
            "size_bytes": None,
            "media_type": spec.media_type,
            "download_url": None,
            "description": spec.description,
            "reason": "This job has no such artifact on disk.",
        })
    return out


# Extensions a job may actually produce, mapped to what a browser should be told.
_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".srt": "application/x-subrip",
    ".vtt": "text/vtt",
    ".json": "application/json",
    ".txt": "text/plain",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}


def media_type_for(path: Path, default: str = "application/octet-stream") -> str:
    """
    The content type for a file, from its extension.

    The real output is a `.webm`, not the `.mp4` the F2 code assumed, so this
    reads the extension rather than trusting a fixed string. A browser will not
    play a video whose content type is wrong.
    """
    return _MEDIA_TYPES.get(path.suffix.lower(), default)
