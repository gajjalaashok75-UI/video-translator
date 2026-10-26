"""
Read-only discovery and capability endpoints.

Every handler here is a GET that inspects the filesystem, installed packages and
configuration. None of them starts a render, downloads a model, or performs
inference, so they stay safe to poll and safe to call on a machine with no GPU,
no token and no optional packages.

No endpoint returns a secret. In particular `HF_TOKEN` is only ever read to
decide whether it is *configured* — a boolean — never echoed.
"""

from __future__ import annotations

import logging
import platform
import shutil
import sys
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from src.api import API_VERSION
from src.api import capabilities as caps
from src.api import discovery, options
from src.api.schemas import (
    CapabilitiesResponse,
    HealthResponse,
    LanguagesResponse,
    OptionsResponse,
    SystemResponse,
    VoicesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["system"])

# Set once when the module is imported, which is when the process starts serving.
_STARTED_AT = time.monotonic()


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health() -> HealthResponse:
    """
    Cheap liveness check.

    Reads no models, touches no pipeline state and does not even look at the
    filesystem, so it is safe as a container health check.
    """
    return HealthResponse(
        status="ok",
        api_version=API_VERSION,
        uptime_seconds=round(time.monotonic() - _STARTED_AT, 3),
    )


def _jobs_dir_writable() -> bool:
    """Whether the durable jobs directory can be created and written."""
    try:
        from src.utils import config
        target = Path(config.JOBS_DIR)
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:  # noqa: BLE001 - a read-only mount must not 500
        return False


@router.get("/system", response_model=SystemResponse, summary="Runtime environment")
def system() -> SystemResponse:
    """
    Runtime facts for the dashboard.

    Every field is best-effort and nullable: a missing GPU, a missing FFmpeg
    binary or an unreadable version must degrade to null, never to an error.
    """
    from src.utils import config

    torch_info = caps.torch_runtime_info()
    return SystemResponse(
        api_version=API_VERSION,
        app_version=caps.app_version(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.system(),
        platform_release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor() or None,
        device=str(config.DEVICE),
        cuda_available=bool(torch_info["cuda_available"]),
        cuda_version=torch_info["cuda_version"],
        cuda_runtime_version=torch_info["cuda_runtime_version"],
        gpu_name=torch_info["gpu_name"],
        gpu_count=int(torch_info["gpu_count"]),
        gpu_memory_total_mb=torch_info["gpu_memory_total_mb"],
        torch_version=torch_info["torch_version"],
        ffmpeg_available=bool(shutil.which("ffmpeg")),
        ffmpeg_version=caps.binary_version("ffmpeg"),
        ffprobe_available=bool(shutil.which("ffprobe")),
        jobs_dir_writable=_jobs_dir_writable(),
    )


@router.get(
    "/models",
    response_model=list[caps.ComponentStatus],
    summary="Optional component availability",
)
def models() -> list[caps.ComponentStatus]:
    """
    Availability of every optional runtime component.

    States are explicit (`available`, `not_installed`, `model_not_downloaded`,
    `token_required`, `unsupported_platform`, `unavailable`, `unknown`) so a
    client can offer the right remedy instead of a generic failure.

    This endpoint never downloads weights and never initialises a model: it
    checks the filesystem, import availability and configuration only. IndicF5
    in particular is reported without any inference, and its runtime availability
    is kept separate from its weight/access availability.
    """
    return list(caps.probe_components().values())


@router.get("/languages", response_model=LanguagesResponse, summary="Language registry")
def languages() -> LanguagesResponse:
    """
    Every language the pipeline can consume or produce.

    Derived from `src/utils/languages.py`, never duplicated. Includes both
    `cloning_supported_by_architecture` (does a cloning engine exist for this
    language?) and `cloning_model_available_on_this_machine` (are its weights
    usable here?), because the UI needs both to avoid promising a voice this
    machine cannot produce.
    """
    return discovery.languages_response()


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    summary="Application capabilities",
)
def capabilities(
    include_routes: bool = Query(
        default=False,
        description=(
            "Also compute the source/target route matrix. This imports the "
            "translation package (and therefore transformers), so it costs a "
            "few seconds on the first call and is cached afterwards."
        ),
    ),
) -> CapabilitiesResponse:
    """
    What this installation can actually do.

    Translation and TTS backends with their availability, cloning engines,
    quality presets, supported languages and optional features. The route matrix
    is opt-in via `include_routes` because building it imports the translation
    package; pass `include_routes=true` for the language explorer. No inference
    is run either way.
    """
    return discovery.capabilities_response(include_routes=include_routes)


@router.get("/options", response_model=OptionsResponse, summary="UI option lists")
def option_lists() -> OptionsResponse:
    """
    The dropdown option lists the interface offers.

    During F1 these are reproduced here rather than shared with `app.py`, which
    is deliberately left unmodified; the `source` field records that provenance
    so the temporary duplication is visible to clients.
    """
    states = caps.component_states()
    return OptionsResponse(**options.all_options(states))


# Backends whose "voices" are a clone of a supplied reference rather than a
# named voice. Asking one for a named voice is a client error, not a 500.
_CLONING_BACKENDS = {"xtts", "f5", "indicf5", "vibevoice", "vibevoice-7b"}

# One engine instance for voice listing. `get_available_voices` is a pure lookup
# over the language registry and loads no model, so a shared instance is safe
# here. This is deliberately NOT the engine a render would use: F2 gives every
# job its own VideoTranslator (and therefore its own engine), because the
# pipeline does hold per-run state.
_voice_engine = None


def _get_voice_engine():
    global _voice_engine
    if _voice_engine is None:
        from src.synthesis.tts import TTSEngine
        _voice_engine = TTSEngine()
    return _voice_engine


@router.get("/voices", response_model=VoicesResponse, summary="TTS voices")
def voices(
    tts_model: str = Query(description="TTS backend key, e.g. 'edge' or 'indicf5'."),
    language: str = Query(description="Language code or display name, e.g. 'hi'."),
) -> VoicesResponse:
    """
    Voices a backend can offer for a language.

    Reuses `TTSEngine.get_available_voices` rather than maintaining a second
    voice registry. An unknown backend or language is a 4xx with an explanatory
    message, not an internal error.
    """
    from src.utils import languages as lang

    try:
        code = lang.normalize(language)
    except lang.UnsupportedLanguage:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown language {language!r}. See /api/languages.",
        )
    if code == "auto":
        raise HTTPException(
            status_code=400,
            detail="'auto' is not a synthesis language; pass a concrete target "
                   "language code.",
        )
    if not lang.is_target(code):
        raise HTTPException(
            status_code=400,
            detail=f"{language!r} is a source-only language and has no TTS voices.",
        )

    engine = _get_voice_engine()
    available = set(engine.backends)
    if tts_model not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown TTS backend {tts_model!r}. Known backends: "
                   f"{', '.join(sorted(available))}.",
        )

    try:
        found = engine.get_available_voices(tts_model, code)
    except Exception as exc:  # noqa: BLE001 - a backend must not 500 the API
        logger.warning("Voice lookup failed for %s/%s: %s", tts_model, code, exc)
        raise HTTPException(
            status_code=503,
            detail=f"Could not list voices for {tts_model!r} ({code}): "
                   f"{type(exc).__name__}.",
        )

    cloning = tts_model in _CLONING_BACKENDS
    note = None
    if cloning:
        note = ("This backend clones the reference voice extracted from the "
                "source audio, so it reports no named voices.")
    elif not found:
        note = (f"{tts_model!r} has no voices configured for {lang.display_name(code)}. "
                "Try 'edge', or a language this backend supports.")

    return VoicesResponse(
        tts_model=tts_model, language=code,
        language_name=lang.display_name(code), voices=list(found),
        cloning=cloning, note=note,
    )
