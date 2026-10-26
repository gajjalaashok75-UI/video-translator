"""
UI-facing option lists.

These are the dropdown choices the interface offers. During F1 they are
*reproduced* here rather than shared with `app.py`, because `app.py` must not be
modified: its literals remain the Gradio UI of record until migration. This
module is the new authoritative source for the API, and the duplication is
intentional and temporary — see `OPTION_PROVENANCE` below.

Where a list can be derived from an existing registry it is derived rather than
retyped, so a change to the registry cannot silently desynchronise the API:

  * TTS backends come from `src.synthesis.capabilities.CAPABILITIES`
  * translation backends come from `src.translation.router.default_capabilities`
  * quality presets come from `src.utils.config.QUALITY_PRESETS` (see
    `/api/capabilities`)

Availability is attached from the capability probes, so a client can grey out an
option that cannot run on this machine instead of offering a guaranteed failure.
"""

from __future__ import annotations

import logging

from src.api.capabilities import module_available
from src.api.schemas import ComponentState, OptionItem
from src.utils import config

logger = logging.getLogger(__name__)

# Provenance of each list, surfaced in the response so the duplication is
# visible in the API itself rather than only in this docstring.
OPTION_PROVENANCE = {
    "translation_models": "app.py literal + src/translation/router.py",
    "context_models": "app.py literal",
    "transcription_models": "app.py literal (faster-whisper sizes)",
    "diarization_backends": "app.py literal",
    "lipsync_models": "app.py literal",
    "ocr_engines": "app.py literal",
    "audio_separators": "app.py literal",
    "live_portrait_modes": "app.py literal",
    "tts_models": "app.py literal + src/synthesis/capabilities.py",
    "subtitle_modes": "src/utils/config.py SUBTITLE_MODES",
}


def _item(value: str, label: str, description: str | None = None,
          state: ComponentState = ComponentState.AVAILABLE,
          available: bool | None = None) -> OptionItem:
    return OptionItem(
        value=value, label=label, description=description,
        state=state, available=(state == ComponentState.AVAILABLE
                                if available is None else available),
    )


def _state_of(states: dict[str, ComponentState], key: str) -> ComponentState:
    return states.get(key, ComponentState.UNKNOWN)


def translation_models(states: dict[str, ComponentState]) -> list[OptionItem]:
    """
    Translation backends, in the order the UI presents them.

    The first entry is a routing choice, not a backend: it lets the pipeline
    pick NLLB for Indic pairs and the user's selection otherwise. It is always
    offered, because the routing itself is local logic.
    """
    llm = _state_of(states, "llm_translation")
    return [
        _item("Auto (NLLB-200 for Indic, else selected)",
              "Auto (NLLB-200 for Indic, else selected)",
              "Let the router choose: NLLB-200 for Indic pairs, the selected "
              "backend otherwise."),
        _item("Google Translate (Online, Fast)", "Google Translate (Online, Fast)",
              "Online machine translation. Needs network access.",
              _state_of(states, "google_translate")),
        _item("NLLB-200 (Local, Indic + 200 Languages)",
              "NLLB-200 (Local, Indic + 200 Languages)",
              "Local 200-language model. Weights download on first use.",
              _state_of(states, "nllb")),
        _item("Tencent HY-MT1.5 (Local, Better Context)",
              "Tencent HY-MT1.5 (Local, Better Context)",
              "Local LLM translator with stronger context handling.", llm),
        _item("Llama 3.1 8B (Local, Instruct)", "Llama 3.1 8B (Local, Instruct)",
              "Local instruction-tuned LLM translator.", llm),
        _item("ALMA-R 7B (Local, Advanced)", "ALMA-R 7B (Local, Advanced)",
              "Local advanced translation model.", llm),
    ]


def context_models(states: dict[str, ComponentState]) -> list[OptionItem]:
    """The context-aware translation models offered when optimisation is on."""
    llm = _state_of(states, "llm_translation")
    return [
        _item("Tencent HY-MT1.5 (Local, Better Context)",
              "Tencent HY-MT1.5 (Local, Better Context)",
              "Local LLM translator with stronger context handling.", llm),
        _item("Llama 3.1 8B (Local, Instruct)", "Llama 3.1 8B (Local, Instruct)",
              "Local instruction-tuned LLM translator.", llm),
        _item("ALMA-R 7B (Local, Advanced)", "ALMA-R 7B (Local, Advanced)",
              "Local advanced translation model.", llm),
    ]


def transcription_models(states: dict[str, ComponentState]) -> list[OptionItem]:
    """
    Speech-to-text model sizes.

    All are Whisper variants served by the same faster-whisper runtime, so they
    share one availability: the weights for any given size download on first
    use, which this endpoint does not trigger.
    """
    state = _state_of(states, "faster_whisper")
    sizes = [
        ("Large v3 Turbo (Fast)", "Large v3 Turbo — fastest large model, "
                                  "recommended."),
        ("Large v3", "Large v3 — highest accuracy, slowest."),
        ("Medium", "Medium — balanced accuracy and speed."),
        ("Base", "Base — fast, lower accuracy."),
        ("Small", "Small — fast, moderate accuracy."),
    ]
    return [_item(value, value, desc, state) for value, desc in sizes]


def diarization_backends(states: dict[str, ComponentState]) -> list[OptionItem]:
    """
    Speaker-diarization backends.

    The default needs SpeechBrain for speaker embeddings; the Community-1 and
    NeMo options are independent backends. NeMo is probed directly here because
    it is not a component the pipeline reports elsewhere.
    """
    pyannote = _state_of(states, "pyannote")
    speechbrain = _state_of(states, "speechbrain")
    default_state = (ComponentState.AVAILABLE
                     if pyannote == speechbrain == ComponentState.AVAILABLE
                     else ComponentState.NOT_INSTALLED)
    nemo_state = (ComponentState.AVAILABLE if module_available("nemo")
                  else ComponentState.NOT_INSTALLED)
    return [
        _item("pyannote/SpeechBrain (Default)", "pyannote / SpeechBrain (Default)",
              "pyannote diarization with SpeechBrain speaker embeddings. "
              "Requires a Hugging Face token.", default_state),
        _item("pyannote/Community-1 (Advanced)", "pyannote / Community-1 (Advanced)",
              "pyannote community pipeline. Requires a Hugging Face token.",
              pyannote),
        _item("NVIDIA NeMo (Advanced)", "NVIDIA NeMo (Advanced)",
              "NVIDIA NeMo diarization.", nemo_state),
    ]


def lipsync_models(states: dict[str, ComponentState]) -> list[OptionItem]:
    """Lip-sync engines, cheapest first."""
    wav2lip = _state_of(states, "wav2lip")
    gfpgan = _state_of(states, "gfpgan")
    combined = (ComponentState.AVAILABLE
                if wav2lip == gfpgan == ComponentState.AVAILABLE
                else (wav2lip if wav2lip != ComponentState.AVAILABLE else gfpgan))
    return [
        _item("Wav2Lip-GAN (Low Quality - Fast)",
              "Wav2Lip-GAN (Low quality, fast)",
              "Fastest lip-sync, lowest visual quality.", wav2lip),
        _item("Wav2Lip + GFPGAN (Low Quality - Medium)",
              "Wav2Lip + GFPGAN (Low quality, medium)",
              "Wav2Lip with GFPGAN face restoration.", combined),
        _item("LivePortrait (High Quality - Slow)",
              "LivePortrait (High quality, slow)",
              "Generative face animation; best quality, slowest.",
              _state_of(states, "live_portrait")),
    ]


def ocr_engines(states: dict[str, ComponentState]) -> list[OptionItem]:
    return [
        _item("PaddleOCR", "PaddleOCR", "Faster OCR engine.",
              _state_of(states, "paddleocr")),
        _item("EasyOCR", "EasyOCR", "More robust OCR engine.",
              _state_of(states, "easyocr")),
    ]


def audio_separators(states: dict[str, ComponentState]) -> list[OptionItem]:
    """
    Audio-separation models.

    Only one is offered today, so the list is a single entry; it is kept as a
    list so the UI contract does not change when a second separator is added.
    """
    return [
        _item("Torchaudio HDemucs (Recommended)",
              "Torchaudio HDemucs (Recommended)",
              "Demucs-based vocal/background separation.",
              _state_of(states, "demucs")),
    ]


def live_portrait_modes(states: dict[str, ComponentState]) -> list[OptionItem]:
    """
    LivePortrait acceleration modes.

    `tensorrt` needs a CUDA-capable GPU; `ort` is the portable fallback. GPU
    availability is checked without initialising anything.
    """
    onnx = _state_of(states, "live_portrait")
    gpu = _state_of(states, "cuda_gpu")
    return [
        _item("ort", "ONNX Runtime (standard)",
              "Portable execution provider; works on CPU and GPU.", onnx),
        _item("tensorrt", "TensorRT (GPU accelerated)",
              "Requires a CUDA-capable GPU.", gpu),
    ]


def tts_models(states: dict[str, ComponentState]) -> list[OptionItem]:
    """
    Text-to-speech backends, in the order the UI presents them.

    The set is validated against `src.synthesis.capabilities.CAPABILITIES` (see
    the consistency test), so adding a backend there without adding it here — or
    vice versa — fails the suite rather than silently diverging.
    """
    entries = [
        ("edge", "Edge TTS (online)", "Online Microsoft voices.", "edge_tts"),
        ("piper", "Piper (local)", "Fast local neural voices.", "piper"),
        ("xtts", "XTTS-v2 (cloning)", "Clones a reference voice.", "xtts"),
        ("f5", "F5-TTS (cloning)", "Clones a reference voice.", "f5"),
        ("indicf5", "IndicF5 (Indic cloning)",
         "Clones a reference voice for the 11 supported Indian languages.",
         "indicf5"),
        ("vibevoice", "VibeVoice (1.5B, long-form)",
         "Long-form multi-speaker synthesis.", "vibevoice"),
        ("vibevoice-7b", "VibeVoice 7B (long-form)",
         "Larger long-form model.", "vibevoice"),
    ]
    return [
        _item(value, label, desc, _state_of(states, key))
        for value, label, desc, key in entries
    ]


def subtitle_modes(states: dict[str, ComponentState]) -> list[OptionItem]:
    """
    Subtitle delivery modes.

    Derived from `config.SUBTITLE_MODES`, so the list the client is offered and
    the set the pipeline accepts cannot drift apart. None of them depends on a
    model being installed, so every entry is available.
    """
    descriptions = {
        "none": "No subtitles are written.",
        "external": "Sidecar .srt and .vtt files next to the video.",
        "embedded": "Sidecar files plus a toggleable subtitle track in the video.",
        "burned": "Subtitles rendered into the picture (cannot be turned off).",
    }
    labels = {
        "none": "None",
        "external": "Sidecar files (SRT + WebVTT)",
        "embedded": "Embedded track (toggleable)",
        "burned": "Burned into the picture",
    }
    return [
        _item(value, labels.get(value, value), descriptions.get(value))
        for value in config.SUBTITLE_MODES
    ]


def all_options(states: dict[str, ComponentState]) -> dict:
    """Every option list plus its provenance, ready for the response model."""
    return {
        "translation_models": translation_models(states),
        "context_models": context_models(states),
        "transcription_models": transcription_models(states),
        "diarization_backends": diarization_backends(states),
        "lipsync_models": lipsync_models(states),
        "ocr_engines": ocr_engines(states),
        "audio_separators": audio_separators(states),
        "live_portrait_modes": live_portrait_modes(states),
        "tts_models": tts_models(states),
        "subtitle_modes": subtitle_modes(states),
        "source": dict(OPTION_PROVENANCE),
    }
