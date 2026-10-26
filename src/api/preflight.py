"""
Preflight: what the backend knows about a render *before* it starts.

The UI shows a readiness panel above the Start button. The checks behind it are
computed here, from the same registries the job-creation validator uses, so the
panel cannot promise something `POST /api/jobs` would then refuse. The frontend
does not re-derive any of this: it renders the list and disables Start when a
`blocking` check fails. The backend stays authoritative — preflight is advice,
and job creation re-validates everything from scratch.

Two severities, and the distinction matters:

  * **blocking** — the render cannot be correct as configured. A TTS backend
    that cannot speak the target language, an enabled feature whose model is not
    installed, a media file with no audio stream. Start is disabled.
  * **warning** — something is off but the render is still valid. An *optional*
    feature that is unavailable but not selected, for instance: the user should
    know it exists and cannot be used, but nothing is broken.

The rule for optional features is: unavailable is a warning while the feature is
off, and becomes blocking the moment the user selects it. That is what makes the
severity meaningful rather than a permanent amber light.
"""

from __future__ import annotations

from typing import Optional

from src.api import options as option_lists, uploads
from src.api.schemas import (
    JobCreateRequest,
    PreflightCheck,
    PreflightResponse,
)
from src.api.validation import (
    ValidationError,
    enumerated_checks,
    validate_languages,
)
from src.utils import config, languages

# Severity tokens. Kept as plain strings so the wire format is stable and a new
# severity is additive rather than a breaking enum change.
OK = "ok"
WARNING = "warning"
BLOCKING = "blocking"


def _check(identifier: str, label: str, severity: str, detail: str,
           field: Optional[str] = None) -> PreflightCheck:
    return PreflightCheck(id=identifier, label=label, severity=severity,
                          detail=detail, field=field)


def _media_checks(request: JobCreateRequest) -> list[PreflightCheck]:
    """
    Video and audio presence, from the upload's own probe.

    The probe is the same ffprobe pass the upload endpoint already ran, so this
    reads a recorded fact rather than running a second probe. A staged upload is
    the only way to know anything about the media; without one, the check is a
    warning rather than a failure, because "no video chosen yet" is a normal
    state of the form and the Start button is already unavailable.
    """
    upload_id = getattr(request, "upload_id", None)
    if not upload_id:
        return [
            _check("video", "Video validated", WARNING,
                   "No video staged yet."),
            _check("audio", "Audio detected", WARNING,
                   "No video staged yet."),
        ]
    try:
        upload = uploads.load(upload_id)
    except uploads.UploadError:
        return [
            _check("video", "Video validated", BLOCKING,
                   f"Upload {upload_id!r} is no longer staged; upload it again.",
                   field="upload_id"),
            _check("audio", "Audio detected", BLOCKING,
                   "The staged upload is gone.", field="upload_id"),
        ]

    info = dict(upload.media or {})
    # `has_video` is absent when ffprobe is unavailable, which is not a failure:
    # the pipeline's own `validate_video` has already accepted the file.
    has_video = info.get("has_video")
    has_audio = info.get("has_audio")

    checks = []
    if has_video is False:
        checks.append(_check("video", "Video validated", BLOCKING,
                             "The file has no video stream to dub."))
    else:
        container = info.get("container") or "video"
        checks.append(_check(
            "video", "Video validated", OK,
            f"{container} · {upload.filename}"
            + (f" · {info['width']}×{info['height']}"
               if info.get("width") and info.get("height") else "")))

    if has_audio is False:
        checks.append(_check("audio", "Audio detected", BLOCKING,
                             "The file has no audio stream, so there is nothing "
                             "to transcribe."))
    else:
        codec = info.get("audio_codec")
        checks.append(_check("audio", "Audio detected", OK,
                             f"{codec} audio" if codec else "Audio stream present."))
    return checks


def _language_checks(request: JobCreateRequest,
                     source: Optional[str],
                     target: Optional[str]) -> list[PreflightCheck]:
    """Whether the language pair is usable, using the resolved codes."""
    checks = []
    if source is None:
        checks.append(_check("source_language", "Source language", BLOCKING,
                             "The source language is not one this installation "
                             "supports.", field="source_lang"))
    else:
        label = "auto-detect" if source == "auto" else languages.get_language_name(source)
        checks.append(_check("source_language", "Source language configured", OK,
                             f"Source: {label}."))

    if target is None:
        checks.append(_check("target_language", "Target language", BLOCKING,
                             "The target language is not one this installation "
                             "can dub into.", field="target_lang"))
    else:
        checks.append(_check("target_language", "Target language configured", OK,
                             f"Target: {languages.get_language_name(target)}."))
    return checks


def _ffmpeg_check() -> PreflightCheck:
    """
    Whether FFmpeg is on PATH.

    Every stage that touches media shells out to it, so its absence is not a
    degradation — it is a hard stop.
    """
    import shutil
    if shutil.which("ffmpeg"):
        return _check("ffmpeg", "FFmpeg ready", OK, "ffmpeg is available on PATH.")
    return _check("ffmpeg", "FFmpeg ready", BLOCKING,
                  "ffmpeg was not found on PATH; nothing can be encoded or muxed.")


def _tts_language_check(target: Optional[str],
                        tts_model: str) -> PreflightCheck:
    """
    The one check that most often blocks an Indic render.

    `languages.validate_tts_pair` is the pipeline's own rule, so calling it here
    means the panel and the pipeline agree. A backend that cannot speak the
    target is blocking, not a warning: the render would fail or fall back to a
    language the user did not ask for.
    """
    if not target:
        return _check("tts_language", "TTS supports target", WARNING,
                      "No target language resolved yet.", field="target_lang")
    try:
        languages.validate_tts_pair(target, tts_model)
    except languages.UnsupportedLanguage as exc:
        return _check("tts_language", "TTS supports target", BLOCKING,
                      str(exc), field="tts_model_name")
    return _check("tts_language", "TTS supports target", OK,
                  f"{tts_model} can speak {languages.get_language_name(target)}.")


def _option_checks(request: JobCreateRequest, states: dict) -> list[PreflightCheck]:
    """
    Availability of every enumerated choice, split by whether it is required.

    A required backend (TTS, ASR, translation, separation) that is not available
    blocks the render — it runs on every job. An optional one blocks only when
    the feature that needs it is switched on; while the feature is off it is a
    warning, so the user learns the option exists without being stopped by it.
    """
    checks = []
    # (field, label, the config flag that turns the feature on, whether it always runs)
    plan = [
        ("audio_model_name", "Audio separation backend", None, True),
        ("transcription_model_name", "Transcription backend", None, True),
        ("translation_model_name", "Translation backend", None, True),
        ("tts_model_name", "TTS backend", None, True),
        ("diarization_model", "Diarization backend",
         request.enable_diarization, False),
        ("lipsync_model_name", "Lip-sync model", request.enable_lipsync, False),
        ("ocr_model_name", "OCR engine",
         request.enable_visual_translation, False),
        ("context_model_name", "Context translation model",
         request.optimize_translation, False),
    ]
    by_field = {field: items for field, items in enumerated_checks(states)}
    for field, label, enabled, required in plan:
        items = by_field.get(field) or []
        chosen = next((item for item in items if item.value == getattr(request, field)),
                      None)
        if chosen is None:
            checks.append(_check(f"option_{field}", label, BLOCKING,
                                 f"{getattr(request, field)!r} is not a known "
                                 f"value for {field}.", field=field))
            continue
        if chosen.available:
            checks.append(_check(f"option_{field}", label, OK, chosen.label))
            continue
        # Unavailable. Whether that stops the render depends on the feature.
        why = chosen.description or f"{label} is not available on this machine."
        if required or enabled:
            checks.append(_check(f"option_{field}", label, BLOCKING, why,
                                 field=field))
        else:
            checks.append(_check(f"option_{field}", label, WARNING,
                                 f"{why} The feature that uses it is off.",
                                 field=field))
    return checks


def _feature_checks(request: JobCreateRequest, states: dict) -> list[PreflightCheck]:
    """
    The boolean features that do not have a model choice of their own.

    Each is only a check when it is switched on; a disabled feature is not a
    problem and does not deserve a row that says so.
    """
    from src.api.schemas import ComponentState

    plan = [
        ("enable_audio_enhancement", "voicefixer", "Audio enhancement",
         "VoiceFixer is not installed; the enhancement stage will be skipped."),
        ("enable_time_stretch", "rubberband", "Time stretching",
         "Rubber Band is not installed; segments will not be stretched to fit."),
    ]
    checks = []
    for flag, probe, label, reason in plan:
        if not getattr(request, flag):
            continue
        available = states.get(probe) == ComponentState.AVAILABLE
        checks.append(_check(
            f"feature_{flag}", label, OK if available else BLOCKING,
            f"{label} is ready." if available else reason, field=flag))
    return checks


def _output_check(request: JobCreateRequest) -> PreflightCheck:
    """
    Whether the requested deliverables can be produced.

    Subtitles are pure pipeline work with no external dependency, so this is
    about the *request* being coherent rather than about a model: a known
    subtitle mode and a valid speaker range. The speaker range is the one place
    a pair of values can disagree.
    """
    mode = request.subtitle_mode
    if mode not in config.SUBTITLE_MODES:
        return _check("output", "Output configuration valid", BLOCKING,
                      f"{mode!r} is not a subtitle mode. Valid: "
                      f"{', '.join(config.SUBTITLE_MODES)}.", field="subtitle_mode")
    if request.min_speakers > request.max_speakers:
        return _check("output", "Output configuration valid", BLOCKING,
                      f"min_speakers ({request.min_speakers}) is greater than "
                      f"max_speakers ({request.max_speakers}).",
                      field="min_speakers")
    described = {
        "none": "no subtitles",
        "external": "sidecar .srt and .vtt files",
        "embedded": "sidecar files and an embedded track",
        "burned": "burned-in subtitles",
    }.get(mode, mode)
    return _check("output", "Output configuration valid", OK,
                  f"Video with dubbed audio, {described}.")


def build_preflight(request: JobCreateRequest,
                    states: Optional[dict] = None) -> PreflightResponse:
    """
    Run every check for a prospective render.

    `states` is injectable so a test (or a caller that already probed) does not
    pay for a second round of component probes. When omitted, the same probe set
    `/api/models` uses is consulted, so the panel and the dashboard agree.
    """
    if states is None:
        from src.api import capabilities as caps
        states = caps.component_states()

    # Resolve the language pair through the real validator, so the panel reports
    # exactly the message a rejected request would carry. It is called directly
    # rather than through `validate_request` because preflight must report *all*
    # problems at once, while `validate_request` stops at the first: a user
    # fixing a form wants the whole list, not one item per submit.
    source: Optional[str] = None
    target: Optional[str] = None
    try:
        source, target = validate_languages(request.source_lang, request.target_lang)
    except ValidationError as exc:
        checks = [_check("languages", "Languages valid", BLOCKING,
                         exc.message, field=exc.field)]
        target = None
        source = None
    else:
        checks = []
        checks.extend(_language_checks(request, source, target))
        checks.append(_tts_language_check(target, request.tts_model_name))

    checks = _media_checks(request) + checks
    checks.append(_ffmpeg_check())
    checks.extend(_option_checks(request, states))
    checks.extend(_feature_checks(request, states))
    checks.append(_output_check(request))

    blocking = [c for c in checks if c.severity == BLOCKING]
    warnings = [c for c in checks if c.severity == WARNING]
    return PreflightResponse(
        ready=not blocking,
        checks=checks,
        blocking=blocking,
        warnings=warnings,
    )
