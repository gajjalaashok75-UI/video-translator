"""
Discovery responses built from the existing registries.

Nothing here is a second source of truth:

  * languages come from `src/utils/languages.py`
  * TTS capabilities come from `src/synthesis/capabilities.py`
  * translation backends come from `src/translation/router.py`
  * quality presets come from `src/utils/config.py`

The only new information is *availability*, which is attached from the
read-only probes in `src/api/capabilities.py`.

`src/translation/router.py` imports only the language registry, so importing it
here does not pull in torch or transformers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.api import capabilities as caps
from src.api.schemas import (
    BackendInfo,
    CapabilitiesResponse,
    ComponentState,
    LanguageInfo,
    LanguagesResponse,
    PresetInfo,
    RouteInfo,
)
from src.utils import languages as languages_module

logger = logging.getLogger(__name__)

# A route matrix over every source/target pair is bounded but can still be
# large; this caps the response so the endpoint cannot grow without limit if the
# registry gains hundreds of languages.
MAX_ROUTES = 4000

# Route matrices keyed by the set of available backends. Building one imports the
# translation package (and transformers), so the result is reused across requests
# instead of paying that cost on every call.
_ROUTE_CACHE: dict[frozenset, tuple[list[RouteInfo], list[str]]] = {}


def _script_from_flores(flores: Optional[str]) -> Optional[str]:
    """'hin_Deva' -> 'Deva'. The flores-200 suffix is the script."""
    if not flores or "_" not in flores:
        return None
    return flores.rsplit("_", 1)[1] or None


def _cloning_engine_for(code: str, dedicated: Optional[str]) -> Optional[str]:
    """
    Which cloning engine, if any, can clone this language's voice.

    A dedicated engine (IndicF5 for the 11 Indic languages) wins. Otherwise the
    general cloning backends are consulted: XTTS-v2 clones for its 17 native
    languages, so English is cloning-capable through XTTS even though it has no
    dedicated engine.
    """
    if dedicated and dedicated in caps_cloning_engines():
        return dedicated
    if code in languages_module.XTTS_NATIVE:
        return "xtts"
    return None


def caps_cloning_engines() -> list[str]:
    """Backends declared `clones=True` in the synthesis capability registry."""
    from src.synthesis.capabilities import CAPABILITIES
    return sorted(name for name, cap in CAPABILITIES.items() if cap.clones)


# `src/translation/__init__.py` eagerly imports `text_translator`, which imports
# torch, transformers and deep-translator. Importing the package therefore drags
# in the whole ML stack, and fails outright when those packages are absent —
# which makes `from src.translation.router import ...` unusable on a minimal
# install. `router.py` itself needs only the language registry, so it is loaded
# directly from its file, skipping the package `__init__` entirely.
#
# The module object is loaded once and reused, so `default_capabilities()` and
# `TranslationRouter` remain the real implementations — no second copy of the
# backend list exists.
_ROUTER_MODULE = None


def router_module():
    """
    The `src/translation/router.py` module, imported without the package.

    Falls back to a normal import if direct loading fails, so behaviour is never
    worse than the straightforward approach.
    """
    global _ROUTER_MODULE
    if _ROUTER_MODULE is not None:
        return _ROUTER_MODULE

    import importlib.util
    import sys

    name = "src._api_router_standalone"
    path = Path(__file__).resolve().parents[1] / "translation" / "router.py"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        # `dataclasses` resolves annotations through `sys.modules[cls.__module__]`,
        # so the module must be registered before it is executed.
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    except Exception as exc:  # noqa: BLE001 - fall back rather than fail
        logger.warning("Direct router load failed (%s); importing the package.",
                       exc)
        from src.translation import router as module
    _ROUTER_MODULE = module
    return module


def language_info(code: str, states: dict[str, ComponentState]) -> LanguageInfo:
    """Full metadata for one registered language."""
    entry = languages_module.LANGUAGES[code]
    dedicated = entry.get("engine")
    engine = _cloning_engine_for(code, dedicated)
    return LanguageInfo(
        code=code,
        name=entry["name"],
        aliases=list(entry.get("aliases") or []),
        script=_script_from_flores(entry.get("flores")),
        flores=entry.get("flores"),
        source=bool(entry.get("source")),
        target=True,
        indic=languages_module.is_indic(code),
        priority_indic=code in languages_module.PRIORITY_INDIC,
        dedicated_engine=dedicated,
        synth_lang=languages_module.synth_lang(code),
        edge_supported=languages_module.supports_edge(code),
        piper_supported=code in languages_module.PIPER_MODEL_MAP,
        chars_per_second=languages_module.chars_per_second(code),
        cloning_supported_by_architecture=engine is not None,
        cloning_model_available_on_this_machine=caps.cloning_model_available(
            engine, states),
        cloning_engine=engine,
    )


def _source_only_info(code: str, name: str) -> LanguageInfo:
    """
    A language Whisper can transcribe but we cannot synthesize.

    It is a source and never a target, and it has no TTS engine at all, so every
    synthesis field is false. `EXTRA_SOURCE_FLORES` carries its flores code even
    when `EXTRA_SOURCE_NAMES` has no display name for it.
    """
    flores = languages_module.EXTRA_SOURCE_FLORES.get(code)
    return LanguageInfo(
        code=code,
        name=name,
        aliases=[],
        script=_script_from_flores(flores),
        flores=flores,
        source=True,
        target=False,
        indic=languages_module.is_indic(code),
        priority_indic=False,
        dedicated_engine=None,
        synth_lang=code,
        edge_supported=False,
        piper_supported=False,
        chars_per_second=14.0,
        cloning_supported_by_architecture=False,
        cloning_model_available_on_this_machine=False,
        cloning_engine=None,
    )


def _auto_detect_info() -> LanguageInfo:
    """
    The synthetic 'auto' entry.

    It is a source choice, not a language, so it is never a target and has no
    synthesis properties. It is kept in `sources` (and only there) so the UI can
    render "Auto Detect" in the same list as real languages.
    """
    return LanguageInfo(
        code="auto",
        name="Auto Detect",
        aliases=["Auto Detect"],
        script=None,
        flores=None,
        source=True,
        target=False,
        indic=False,
        priority_indic=False,
        dedicated_engine=None,
        synth_lang="auto",
        edge_supported=False,
        piper_supported=False,
        chars_per_second=14.0,
        cloning_supported_by_architecture=False,
        cloning_model_available_on_this_machine=False,
        cloning_engine=None,
    )


def languages_response(states: Optional[dict[str, ComponentState]] = None) -> LanguagesResponse:
    """
    Every language, plus the ordered source and target lists the UI uses.

    Order is preserved from the registry: `targets` follows the registry's own
    priority ordering (priority Indic first), and `sources` follows
    `supported_sources()` so auto-detect stays first.
    """
    states = states or caps.component_states()

    all_langs = [language_info(code, states) for code in languages_module.LANGUAGES]

    # Extra source-only languages, using the registry's own display names so a
    # name is never invented here.
    extras = [
        _source_only_info(code, name)
        for code, name in languages_module.EXTRA_SOURCE_NAMES.items()
    ]

    by_code = {lang.code: lang for lang in all_langs}
    by_code.update({lang.code: lang for lang in extras})

    # Reuse the registry's ordering rather than reimplementing it. 'auto' is a
    # source choice with no registry entry, so it is synthesised here to keep
    # `sources` complete and in the registry's own order.
    auto = _auto_detect_info()
    by_code[auto.code] = auto
    sources = [
        by_code[entry["code"]]
        for entry in languages_module.supported_sources()
        if entry["code"] in by_code
    ]
    targets = [
        by_code[entry["code"]]
        for entry in languages_module.supported_targets()
        if entry["code"] in by_code
    ]

    return LanguagesResponse(
        languages=all_langs + extras, sources=sources, targets=targets,
        auto_detect_code="auto",
    )


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------
# Which probe backs each translation backend. The LLM translators share one
# runtime, so they share one state.
_TRANSLATION_PROBE = {
    "nllb": "nllb",
    "hymt": "llm_translation",
    "llama": "llm_translation",
    "alma": "llm_translation",
    "google": "google_translate",
}

_TTS_PROBE = {
    "edge": "edge_tts",
    "piper": "piper",
    "xtts": "xtts",
    "f5": "f5",
    "indicf5": "indicf5",
    "vibevoice": "vibevoice",
}

_TTS_LABEL = {
    "edge": "Edge TTS (online)",
    "piper": "Piper (local)",
    "xtts": "XTTS-v2 (cloning)",
    "f5": "F5-TTS (cloning)",
    "indicf5": "IndicF5 (Indic cloning)",
    "vibevoice": "VibeVoice (long-form)",
}


def _backend(name: str, label: str, kind: str, probe_key: str,
             states: dict[str, ComponentState]) -> BackendInfo:
    state = states.get(probe_key, ComponentState.UNKNOWN)
    available = state == ComponentState.AVAILABLE
    reasons = {
        ComponentState.AVAILABLE: f"{label} can run on this machine.",
        ComponentState.NOT_INSTALLED: f"{label} is not installed.",
        ComponentState.MODEL_NOT_DOWNLOADED: f"{label} weights are not present.",
        ComponentState.TOKEN_REQUIRED: f"{label} needs a Hugging Face token.",
        ComponentState.UNSUPPORTED_PLATFORM: f"{label} is not supported on this platform.",
        ComponentState.UNAVAILABLE: f"{label} cannot run right now.",
        ComponentState.UNKNOWN: f"{label} availability is unknown.",
    }
    return BackendInfo(
        name=name, label=label, kind=kind, available=available, state=state,
        reason=reasons.get(state, f"{label} availability is unknown."),
    )


def translation_backends(states: dict[str, ComponentState]) -> list[BackendInfo]:
    default_capabilities = router_module().default_capabilities
    labels = {
        "nllb": "NLLB-200 (local, 200 languages)",
        "hymt": "Tencent HY-MT1.5 (local)",
        "llama": "Llama 3.1 8B (local)",
        "alma": "ALMA-R 7B (local)",
        "google": "Google Translate (online)",
    }
    out = []
    for cap in default_capabilities():
        probe = _TRANSLATION_PROBE.get(cap.name, "llm_translation")
        out.append(_backend(cap.name, labels.get(cap.name, cap.name),
                            "translation", probe, states))
    return out


def tts_backends(states: dict[str, ComponentState]) -> list[BackendInfo]:
    from src.synthesis.capabilities import CAPABILITIES
    out = []
    for name in CAPABILITIES:
        probe = _TTS_PROBE.get(name, name)
        out.append(_backend(name, _TTS_LABEL.get(name, name), "tts", probe, states))
    return out


def quality_presets() -> list[PresetInfo]:
    from src.utils import config
    return [
        PresetInfo(name=name, settings=dict(settings))
        for name, settings in config.QUALITY_PRESETS.items()
    ]


def _features(states: dict[str, ComponentState]) -> dict[str, bool]:
    ok = ComponentState.AVAILABLE

    def any_ok(*keys: str) -> bool:
        return any(states.get(k) == ok for k in keys)

    return {
        "transcription": states.get("faster_whisper") == ok,
        "separation": states.get("demucs") == ok,
        "diarization": (states.get("pyannote") == ok
                        and states.get("speechbrain") == ok),
        "translation_local": (states.get("nllb") == ok
                              or states.get("llm_translation") == ok),
        "translation_online": states.get("google_translate") == ok,
        "tts": any_ok("edge_tts", "piper", "xtts", "f5", "indicf5", "vibevoice"),
        "cloning": any_ok(*[e for e in caps_cloning_engines()]),
        "lip_sync": any_ok("wav2lip", "live_portrait"),
        "audio_enhancement": states.get("voicefixer") == ok,
        "visual_translation": any_ok("easyocr", "paddleocr"),
        "time_stretch": states.get("rubberband") == ok,
        # Chunked processing is pure pipeline logic with no external dependency.
        "chunked_processing": True,
        "resume": True,
    }


def _routes(states: dict[str, ComponentState]) -> tuple[list[RouteInfo], list[str]]:
    """
    The source/target route matrix, for the language explorer.

    Every shipped backend is `any_to_any`, so a direct route always exists when
    at least one backend is available; the value here is telling the client
    *which* backend serves each pair, and flagging pairs that need a pivot.

    This is the one expensive call in the API. `src/translation/__init__.py`
    imports `text_translator`, which imports transformers, so importing the
    router costs a few seconds the first time. Results are cached per set of
    available backends, and the endpoint keeps this opt-in so a dashboard call
    does not pay for it.

    Returns the routes plus notes about anything that limited the matrix.
    """
    TranslationRouter = router_module().TranslationRouter

    notes: list[str] = []
    available = {
        name for name, state in (
            ("nllb", states.get("nllb")),
            ("hymt", states.get("llm_translation")),
            ("llama", states.get("llm_translation")),
            ("alma", states.get("llm_translation")),
            ("google", states.get("google_translate")),
        ) if state == ComponentState.AVAILABLE
    }
    if not available:
        notes.append(
            "No translation backend is installed, so no routes are listed. "
            "Install deep-translator or the local NLLB/LLM runtime to enable "
            "translation.")
        return [], notes

    cache_key = frozenset(available)
    cached = _ROUTE_CACHE.get(cache_key)
    if cached is not None:
        return list(cached[0]), list(cached[1])

    sources = [code for code in languages_module.LANGUAGES
               if languages_module.LANGUAGES[code].get("source")]
    sources += list(languages_module.EXTRA_SOURCE_NAMES)
    targets = list(languages_module.LANGUAGES)

    router = TranslationRouter()
    routes: list[RouteInfo] = []
    for src in sources:
        for tgt in targets:
            if len(routes) >= MAX_ROUTES:
                notes.append(f"Route matrix truncated at {MAX_ROUTES} entries.")
                _ROUTE_CACHE[cache_key] = (list(routes), list(notes))
                return routes, notes
            try:
                plan = router.route(src, tgt, available=available)
            except Exception as exc:  # noqa: BLE001 - one bad pair must not fail the request
                logger.debug("No route for %s->%s: %s", src, tgt, exc)
                continue
            pivots = list(plan.steps[1:-1]) if plan.is_pivot else []
            routes.append(RouteInfo(
                source=plan.source, target=plan.target,
                backend=None if plan.backend == "none" else plan.backend,
                pivots=pivots, is_passthrough=plan.is_passthrough,
            ))
    _ROUTE_CACHE[cache_key] = (list(routes), list(notes))
    return routes, notes


def capabilities_response(
    states: Optional[dict[str, ComponentState]] = None,
    include_routes: bool = False,
) -> CapabilitiesResponse:
    """
    Build the capabilities response.

    `include_routes` is off by default: building the matrix imports the
    translation package (and therefore transformers), which is a multi-second
    cost that a dashboard does not need. The language explorer asks for it
    explicitly.
    """
    states = states or caps.component_states()

    if include_routes:
        routes, notes = _routes(states)
    else:
        routes = []
        notes = [
            "The translation route matrix was not included. Request "
            "/api/capabilities?include_routes=true to compute it."
        ]

    if states.get("cuda_gpu") != ComponentState.AVAILABLE:
        notes.append(
            "No CUDA GPU detected; GPU-accelerated lip-sync (TensorRT) is "
            "unavailable and processing will run on CPU.")

    return CapabilitiesResponse(
        translation_backends=translation_backends(states),
        tts_backends=tts_backends(states),
        cloning_engines=caps_cloning_engines(),
        quality_presets=quality_presets(),
        features=_features(states),
        sources=[code for code in languages_module.LANGUAGES
                 if languages_module.LANGUAGES[code].get("source")]
        + list(languages_module.EXTRA_SOURCE_NAMES),
        targets=list(languages_module.LANGUAGES),
        priority_indic=list(languages_module.PRIORITY_INDIC),
        routes=routes,
        notes=notes,
    )
