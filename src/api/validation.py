"""
Validation of job-creation requests.

Everything a client sends is checked here, *before* a job directory exists and
before any model is loaded, so a bad request costs a few microseconds instead of
a failed render minutes later. The rules are drawn from the same registries the
pipeline and the option endpoints use, so the API cannot accept a value the
pipeline would reject:

  * languages come from `src.utils.languages` (`normalize`, `is_target`, and
    `validate_tts_pair` for the TTS/language compatibility the pipeline enforces)
  * quality presets come from `src.utils.config.preset_config`
  * option values are checked against the same lists `/api/options` publishes

This is deliberately *stricter* than the pipeline in one respect and no stricter
anywhere else: the pipeline tolerates some inconsistent combinations and
resolves them at run time, while the API refuses them up front with a message
that names the field. Accepting a request the pipeline will silently reinterpret
is how a client ends up debugging the wrong thing.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.api import options as option_lists
from src.api.schemas import JobCreateRequest
from src.utils import config, languages

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """A request was rejected. `field` names the offending input."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.field = field


def _allowed_values(items) -> list[str]:
    return [item.value for item in items]


def _check_choice(value: str, items, field: str) -> None:
    allowed = _allowed_values(items)
    if value not in allowed:
        raise ValidationError(
            f"{value!r} is not a valid {field}. Valid values: "
            f"{', '.join(allowed)}", field=field)


def validate_languages(source_lang: str, target_lang: str) -> tuple[str, str]:
    """
    Resolve and check the language pair.

    Returns the *normalised* codes, which are what the pipeline should be given:
    the UI passes display names and the registry resolves them, but the API
    records and compares codes, so a client sending "Hindi" and one sending "hi"
    must produce the same job.
    """
    try:
        source = languages.normalize(source_lang)
    except languages.UnsupportedLanguage as exc:
        raise ValidationError(
            f"Unknown source language {source_lang!r}. See /api/languages for "
            f"the supported set.", field="source_lang") from exc

    try:
        target = languages.normalize(target_lang)
    except languages.UnsupportedLanguage as exc:
        raise ValidationError(
            f"Unknown target language {target_lang!r}. See /api/languages for "
            f"the supported set.", field="target_lang") from exc

    if target == "auto":
        raise ValidationError(
            "'auto' is only valid as a source language; a target must be a "
            "specific language.", field="target_lang")
    if not languages.is_target(target):
        raise ValidationError(
            f"{target_lang!r} cannot be used as a dubbing target. See "
            f"/api/languages for the supported set.", field="target_lang")
    if source != "auto" and not languages.is_known(source):
        raise ValidationError(
            f"{source_lang!r} is not a language this installation can detect.",
            field="source_lang")
    return source, target


def validate_tts_compatibility(target_code: str, tts_model: str) -> None:
    """
    The TTS/language check the pipeline itself performs.

    `languages.validate_tts_pair` is the pipeline's own rule (IndicF5 speaks only
    the Indic set, XTTS only its list, Piper only languages with a voice), so
    calling it here means the API and the pipeline agree by construction.
    """
    try:
        languages.validate_tts_pair(target_code, tts_model)
    except languages.UnsupportedLanguage as exc:
        raise ValidationError(str(exc), field="tts_model_name") from exc


def enumerated_checks(states: dict) -> list[tuple[str, list]]:
    """
    The `(field, allowed options)` pairs every request's enumerated fields.

    Exposed so the preflight check list and the job-creation validator consult
    the *same* table: a field added here is checked in both places, and the two
    can never disagree about which values exist.
    """
    return [
        ("audio_model_name", option_lists.audio_separators(states)),
        ("tts_model_name", option_lists.tts_models(states)),
        ("translation_model_name", option_lists.translation_models(states)),
        ("context_model_name", option_lists.context_models(states)),
        ("transcription_model_name", option_lists.transcription_models(states)),
        ("diarization_model", option_lists.diarization_backends(states)),
        ("lipsync_model_name", option_lists.lipsync_models(states)),
        ("live_portrait_mode", option_lists.live_portrait_modes(states)),
        ("ocr_model_name", option_lists.ocr_engines(states)),
    ]


def validate_options(request: JobCreateRequest,
                     states: Optional[dict] = None) -> None:
    """
    Check every enumerated field against the lists `/api/options` publishes.

    Availability is *not* enforced: a model whose weights are not downloaded yet
    can still be requested (the pipeline downloads on first use), and refusing it
    would make the API stricter than the product. What is enforced is that the
    value names something that exists, so a typo is caught here.
    """
    if states is None:
        from src.api import capabilities as caps
        states = caps.component_states()

    for field, items in enumerated_checks(states):
        _check_choice(getattr(request, field), items, field)


def validate_subtitle_mode(mode: Optional[str]) -> str:
    """
    Check a subtitle mode against the registry.

    An absent value means the default. An unknown value is a 422 rather than a
    silent fallback: `burned` and `embedded` change the delivered video, so a
    client that asks for something this server does not implement must be told,
    not handed a different product.
    """
    if mode is None:
        return config.DEFAULT_SUBTITLE_MODE
    value = str(mode).strip().lower()
    if value not in config.SUBTITLE_MODES:
        raise ValidationError(
            f"Unknown subtitle mode {mode!r}. Valid values: "
            f"{', '.join(config.SUBTITLE_MODES)}", field="subtitle_mode")
    return value


def validate_preset(name: Optional[str]) -> Optional[str]:
    """Check a quality preset name against the registry."""
    if name is None:
        return None
    if name not in config.QUALITY_PRESET_NAMES:
        raise ValidationError(
            f"Unknown quality preset {name!r}. Valid values: "
            f"{', '.join(config.QUALITY_PRESET_NAMES)}", field="quality_preset")
    return name


def validate_request(request: JobCreateRequest) -> tuple[str, str]:
    """
    Validate a job request end to end.

    Returns the normalised `(source_lang, target_lang)`. Raises
    `ValidationError` on the first problem, so the client gets one clear message
    rather than a pile of them.
    """
    if not request.upload_id or not request.upload_id.strip():
        raise ValidationError("upload_id is required.", field="upload_id")

    if request.min_speakers > request.max_speakers:
        raise ValidationError(
            f"min_speakers ({request.min_speakers}) cannot exceed max_speakers "
            f"({request.max_speakers}).", field="min_speakers")

    validate_preset(request.quality_preset)
    request.subtitle_mode = validate_subtitle_mode(request.subtitle_mode)

    source, target = validate_languages(request.source_lang, request.target_lang)
    validate_tts_compatibility(target, request.tts_model_name)
    validate_options(request)
    return source, target
