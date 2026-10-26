"""
The durable result summary for a finished job.

Built entirely from the manifest on disk, so the same URL returns the same
facts after a restart, in a fresh browser, or a week later. The alternative —
having the UI remember what it submitted — breaks the moment the user refreshes
the result page or opens the job on another machine, which is exactly when a
"what did this job do?" panel is most useful.

Nothing here recomputes a render or inspects a model: every value is a recorded
fact, or `None` when the job predates the field. Old manifests are therefore
readable, and a missing value renders as "unknown" rather than as an error.
"""

from __future__ import annotations

from typing import Optional

from src.api.schemas import JobSummary
from src.core.manifest import Manifest
from src.utils import config, languages


def _name(code: Optional[str]) -> Optional[str]:
    """
    A display name for a language code, or None when it is unknown.

    `is_known` is checked first because `get_language_name` falls back to
    "English" for anything it does not recognise, which would silently label an
    unrecognised code as English.
    """
    if not code:
        return None
    if code == "auto":
        return "Auto-detect"
    try:
        if not languages.is_known(code):
            return None
        return languages.get_language_name(code)
    except Exception:  # noqa: BLE001 - a display name must never fail a status
        return None


def _duration(manifest: Manifest) -> Optional[float]:
    """
    The render's duration, from the sync stage's recorded detail.

    Sync is where the dubbed timeline is built, so its duration is the length
    the subtitles and the final mix were both cut to — a better answer than the
    source file's duration, which is what the video had before dubbing.
    """
    record = manifest.stages.get("sync")
    if record is None:
        return None
    value = (record.detail or {}).get("duration")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _speaker_count(manifest: Manifest) -> int:
    """
    How many speakers were detected, from the most authoritative source.

    Diarization's own count is preferred; failing that the references stage's,
    which is one reference per speaker. A job with diarization disabled reports
    zero, which the UI reads as "single voice".
    """
    for stage in ("diarization", "references"):
        record = manifest.stages.get(stage)
        if record is not None:
            try:
                count = int((record.detail or {}).get("speakers") or 0)
            except (TypeError, ValueError):
                count = 0
            if count:
                return count
    return 0


def build_summary(manifest: Manifest, state: str) -> JobSummary:
    """
    The summary of a job, from durable state only.

    `state` is passed in rather than derived here so the summary and the rest of
    the status response cannot disagree about whether the job is finished.
    """
    cfg = dict(manifest.config or {})
    source = dict(manifest.source or {})

    source_code = source.get("language") or cfg.get("source_lang")
    target_code = source.get("target_language") or cfg.get("target_lang")

    enable_diarization = bool(cfg.get("enable_diarization"))
    enable_lipsync = bool(cfg.get("enable_lipsync"))

    try:
        subtitle_count = int((manifest.artifacts or {}).get("subtitle_count") or 0)
    except (TypeError, ValueError):
        subtitle_count = 0

    return JobSummary(
        source_language=source_code,
        source_language_name=_name(source_code),
        target_language=target_code,
        target_language_name=_name(target_code),
        translation_backend=cfg.get("translation_model_name"),
        context_backend=(cfg.get("context_model_name")
                         if cfg.get("optimize_translation") else None),
        optimize_translation=bool(cfg.get("optimize_translation")),
        # "Preserve speaker voice" is what diarization buys: a reference clip per
        # speaker, used to condition the TTS. Without it every line is spoken in
        # the backend's standard voice. Named here so the UI does not have to
        # interpret `enable_diarization` itself.
        voice_mode="preserve_speaker" if enable_diarization else "standard",
        tts_backend=cfg.get("tts_model_name"),
        tts_voice=cfg.get("tts_voice"),
        subtitle_mode=cfg.get("subtitle_mode") or config.DEFAULT_SUBTITLE_MODE,
        subtitle_count=subtitle_count,
        lipsync=enable_lipsync,
        lipsync_model=cfg.get("lipsync_model_name") if enable_lipsync else None,
        diarization=enable_diarization,
        speakers=_speaker_count(manifest),
        duration_seconds=_duration(manifest),
        status=str(manifest.status or "UNKNOWN"),
        state=state,
    )
