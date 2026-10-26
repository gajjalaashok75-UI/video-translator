"""
Read-only capability probing.

Everything here answers "can this machine do X right now?" using only:

  * filesystem checks (are the weights on disk?)
  * import availability (is the package installed?)
  * environment/config inspection (is a token configured? is the platform supported?)
  * cheap version probes of already-present binaries

It never initialises a model, never downloads anything and never runs inference.
A call to `/api/models` must stay fast and must be safe on a CPU-only box with
no Hugging Face token and no optional packages installed.

The state vocabulary is deliberately finer than a boolean, because the remedies
differ: installing a package, accepting a model licence, downloading weights,
or nothing at all (the platform simply cannot do it).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

from src.api.schemas import ComponentState, ComponentStatus

logger = logging.getLogger(__name__)

# Repo-relative locations of the assets the pipeline expects. Kept here as
# literals rather than imported, because the owning modules hold them as
# instance attributes and importing those modules would pull in torch.
WAV2LIP_WEIGHTS = Path("models/wav2lip/wav2lip_gan.pth")
GFPGAN_WEIGHTS = Path("models/gfpgan/GFPGANv1.4.pth")
LIVE_PORTRAIT_DIR = Path("models/live_portrait_onnx/liveportrait_onnx")
SPEECHBRAIN_DIR = Path("models/speechbrain/spkrec-ecapa-voxceleb")

# The ONNX graphs LivePortrait loads at startup; all must be present.
LIVE_PORTRAIT_FILES = (
    "appearance_feature_extractor.onnx",
    "motion_extractor.onnx",
    "warping_spade.onnx",
    "landmark.onnx",
    "stitching.onnx",
    "stitching_lip.onnx",
)

# IndicF5 is gated on Hugging Face and downloaded on first use.
INDICF5_REPO_ID = "ai4bharat/IndicF5"
INDICF5_WEIGHTS_FILE = "model.safetensors"


# ---------------------------------------------------------------------------
# Low-level probes
# ---------------------------------------------------------------------------
def _is_mocked(module: object) -> bool:
    """
    True when a module is a test double rather than a real install.

    `tests/conftest.py` injects `MagicMock` objects into `sys.modules` for heavy
    dependencies that are not installed. Without this check a mocked module
    would be reported as available, and worse, its attributes would look like
    real CUDA devices.
    """
    return type(module).__name__ in ("MagicMock", "Mock", "SmartMock")


def _find_spec_without_import(name: str):
    """
    Locate a module's spec without importing it or any parent package.

    `importlib.util.find_spec("pyannote.audio")` imports `pyannote` first, and
    that import pulls in transformers and sklearn — seconds of work, and exactly
    the heavy stack the API is meant to stay away from. Walking the search
    locations by hand avoids the import entirely.
    """
    parts = name.split(".")
    try:
        spec = importlib.util.find_spec(parts[0])
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    path = spec.submodule_search_locations
    for part in parts[1:]:
        if path is None:
            return None
        try:
            spec = importlib.machinery.PathFinder.find_spec(part, path)
        except (ImportError, ModuleNotFoundError, ValueError):
            return None
        if spec is None:
            return None
        path = spec.submodule_search_locations
    return spec


def module_available(name: str) -> bool:
    """Whether `name` can be imported, without importing it."""
    root = name.split(".")[0]
    existing = sys.modules.get(root)
    if existing is not None:
        # Already imported, so the root is real; a dotted submodule is then a
        # plain attribute check rather than a fresh path search.
        if _is_mocked(existing):
            return False
        if "." not in name:
            return True
        return hasattr(existing, name.split(".", 1)[1].split(".")[0])
    return _find_spec_without_import(name) is not None


def _repo_root() -> Path:
    """Repository root, resolved from this file (src/api/capabilities.py)."""
    return Path(__file__).resolve().parents[2]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else _repo_root() / path


def _weights_present(path: Path) -> bool:
    try:
        p = _resolve(path)
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def _hf_cache_roots() -> list[Path]:
    """Candidate Hugging Face hub cache directories, most specific first."""
    roots: list[Path] = []
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(env)
        if value:
            roots.append(Path(value))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def hf_cached_file(repo_id: str, filename: str) -> Optional[Path]:
    """
    Locate a file in the Hugging Face cache without touching the network.

    The cache layout is `models--<owner>--<name>/snapshots/<rev>/<file>`. A
    blob may be absent while the snapshot symlink exists (or vice versa), so the
    file itself is what is checked. Returns None when the file is not cached.
    """
    dir_name = "models--" + repo_id.replace("/", "--")
    for root in _hf_cache_roots():
        snapshots = root / dir_name / "snapshots"
        try:
            if not snapshots.is_dir():
                continue
            for revision in snapshots.iterdir():
                candidate = revision / filename
                if candidate.is_file() and candidate.stat().st_size > 0:
                    return candidate
        except OSError:
            continue
    return None


def binary_version(binary: str, args: Iterable[str] = ("-version",)) -> Optional[str]:
    """
    First line of a binary's version output, or None when it is missing.

    Bounded by a short timeout: a hung binary must not hang `/api/system`.
    """
    path = shutil.which(binary)
    if not path:
        return None
    try:
        proc = subprocess.run(
            [path, *args], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (proc.stdout or proc.stderr or "").strip()
    return output.splitlines()[0].strip() if output else None


# ---------------------------------------------------------------------------
# Component probes
# ---------------------------------------------------------------------------
def _probe_ffmpeg() -> ComponentStatus:
    version = binary_version("ffmpeg")
    if version:
        return ComponentStatus(
            name="ffmpeg", label="FFmpeg", state=ComponentState.AVAILABLE,
            available=True, reason="FFmpeg is installed and on PATH.",
            required_for=["core"], detail={"version": version},
        )
    return ComponentStatus(
        name="ffmpeg", label="FFmpeg", state=ComponentState.NOT_INSTALLED,
        available=False,
        reason="FFmpeg was not found on PATH. It is required for all audio and "
               "video decoding and muxing.",
        required_for=["core"],
    )


def _probe_edge_tts() -> ComponentStatus:
    if module_available("edge_tts"):
        return ComponentStatus(
            name="edge_tts", label="Edge TTS", state=ComponentState.AVAILABLE,
            available=True,
            reason="edge-tts is installed. It is an online service, so synthesis "
                   "needs network access.",
            required_for=["tts"], detail={"network_required": "true"},
        )
    return ComponentStatus(
        name="edge_tts", label="Edge TTS", state=ComponentState.NOT_INSTALLED,
        available=False,
        reason="The 'edge-tts' package is not installed (pip install edge-tts).",
        required_for=["tts"],
    )


def _probe_piper() -> ComponentStatus:
    if not module_available("piper"):
        return ComponentStatus(
            name="piper", label="Piper TTS", state=ComponentState.NOT_INSTALLED,
            available=False,
            reason="The 'piper-tts' package is not installed. It has "
                   "platform-specific build requirements and is installed "
                   "manually.",
            required_for=["tts"],
        )
    return ComponentStatus(
        name="piper", label="Piper TTS", state=ComponentState.AVAILABLE,
        available=True, reason="piper-tts is installed.",
        required_for=["tts"],
    )


def _probe_wav2lip() -> ComponentStatus:
    if not module_available("torch"):
        return ComponentStatus(
            name="wav2lip", label="Wav2Lip lip-sync",
            state=ComponentState.NOT_INSTALLED, available=False,
            reason="PyTorch is not installed, so no lip-sync engine can run.",
            required_for=["lip_sync"],
        )
    if not _weights_present(WAV2LIP_WEIGHTS):
        return ComponentStatus(
            name="wav2lip", label="Wav2Lip lip-sync",
            state=ComponentState.MODEL_NOT_DOWNLOADED, available=False,
            reason="The Wav2Lip checkpoint is missing. Run "
                   "scripts/download_models.py to fetch it.",
            required_for=["lip_sync"],
            detail={"expected_path": str(WAV2LIP_WEIGHTS)},
        )
    return ComponentStatus(
        name="wav2lip", label="Wav2Lip lip-sync", state=ComponentState.AVAILABLE,
        available=True, reason="The Wav2Lip checkpoint is present.",
        required_for=["lip_sync"],
    )


def _probe_gfpgan() -> ComponentStatus:
    if not (module_available("gfpgan") and module_available("basicsr")):
        return ComponentStatus(
            name="gfpgan", label="GFPGAN face restoration",
            state=ComponentState.NOT_INSTALLED, available=False,
            reason="The 'gfpgan' and 'basicsr' packages are not both installed.",
            required_for=["lip_sync"],
        )
    if not _weights_present(GFPGAN_WEIGHTS):
        return ComponentStatus(
            name="gfpgan", label="GFPGAN face restoration",
            state=ComponentState.MODEL_NOT_DOWNLOADED, available=False,
            reason="The GFPGAN checkpoint is missing. Run "
                   "scripts/download_models.py to fetch it.",
            required_for=["lip_sync"],
            detail={"expected_path": str(GFPGAN_WEIGHTS)},
        )
    return ComponentStatus(
        name="gfpgan", label="GFPGAN face restoration", state=ComponentState.AVAILABLE,
        available=True, reason="GFPGAN and its checkpoint are present.",
        required_for=["lip_sync"],
    )


def _probe_live_portrait() -> ComponentStatus:
    if not module_available("onnxruntime"):
        return ComponentStatus(
            name="live_portrait", label="LivePortrait",
            state=ComponentState.NOT_INSTALLED, available=False,
            reason="onnxruntime is not installed.",
            required_for=["lip_sync"],
        )
    directory = _resolve(LIVE_PORTRAIT_DIR)
    if not directory.is_dir():
        return ComponentStatus(
            name="live_portrait", label="LivePortrait",
            state=ComponentState.MODEL_NOT_DOWNLOADED, available=False,
            reason="The LivePortrait ONNX model directory is missing.",
            required_for=["lip_sync"],
            detail={"expected_path": str(LIVE_PORTRAIT_DIR)},
        )
    missing = [f for f in LIVE_PORTRAIT_FILES if not (directory / f).is_file()]
    if missing:
        return ComponentStatus(
            name="live_portrait", label="LivePortrait",
            state=ComponentState.MODEL_NOT_DOWNLOADED, available=False,
            reason=f"The LivePortrait ONNX model directory is incomplete "
                   f"({len(missing)} of {len(LIVE_PORTRAIT_FILES)} graphs missing).",
            required_for=["lip_sync"],
            detail={"missing": ", ".join(missing)},
        )
    return ComponentStatus(
        name="live_portrait", label="LivePortrait", state=ComponentState.AVAILABLE,
        available=True,
        reason="All LivePortrait ONNX graphs are present. TensorRT acceleration "
               "additionally requires a CUDA-capable GPU.",
        required_for=["lip_sync"],
    )


def _probe_speechbrain() -> ComponentStatus:
    if not module_available("speechbrain"):
        return ComponentStatus(
            name="speechbrain", label="SpeechBrain (speaker embeddings)",
            state=ComponentState.NOT_INSTALLED, available=False,
            reason="The 'speechbrain' package is not installed.",
            required_for=["diarization"],
        )
    if not _resolve(SPEECHBRAIN_DIR).is_dir():
        return ComponentStatus(
            name="speechbrain", label="SpeechBrain (speaker embeddings)",
            state=ComponentState.MODEL_NOT_DOWNLOADED, available=False,
            reason="The speaker-recognition model directory is missing; it is "
                   "downloaded on first use.",
            required_for=["diarization"],
            detail={"expected_path": str(SPEECHBRAIN_DIR)},
        )
    return ComponentStatus(
        name="speechbrain", label="SpeechBrain (speaker embeddings)",
        state=ComponentState.AVAILABLE, available=True,
        reason="SpeechBrain and its speaker model are present.",
        required_for=["diarization"],
    )


def _indicf5_implementation_missing() -> list[str]:
    """Runtime packages IndicF5 needs, minus the ones that are installed."""
    required = ("huggingface_hub", "safetensors", "vocos", "x_transformers",
                "torchdiffeq", "torch")
    return [pkg for pkg in required if not module_available(pkg)]


def _probe_indicf5() -> ComponentStatus:
    """
    IndicF5 reported with implementation and weight/access availability apart.

    Three distinct situations produce three distinct states, because the user
    action differs for each:
      * a runtime package is missing        -> not_installed
      * no token and no cached weights      -> token_required
      * token present (or cache populated)  -> available / model_not_downloaded
    """
    required_for = ["tts", "cloning"]
    missing = _indicf5_implementation_missing()
    if missing:
        return ComponentStatus(
            name="indicf5", label="IndicF5 (Indic voice cloning)",
            state=ComponentState.NOT_INSTALLED, available=False,
            reason="The IndicF5 runtime is incomplete; missing: "
                   f"{', '.join(missing)}.",
            required_for=required_for,
            detail={"implementation_available": "false",
                    "missing_packages": ", ".join(missing)},
        )

    cached = hf_cached_file(INDICF5_REPO_ID, INDICF5_WEIGHTS_FILE)
    has_token = bool(os.environ.get("HF_TOKEN"))

    if cached is not None:
        return ComponentStatus(
            name="indicf5", label="IndicF5 (Indic voice cloning)",
            state=ComponentState.AVAILABLE, available=True,
            reason="The IndicF5 runtime is installed and its checkpoint is "
                   "cached locally.",
            required_for=required_for,
            detail={"implementation_available": "true",
                    "weights_cached": "true",
                    "token_configured": str(has_token).lower()},
        )

    if not has_token:
        return ComponentStatus(
            name="indicf5", label="IndicF5 (Indic voice cloning)",
            state=ComponentState.TOKEN_REQUIRED, available=False,
            reason="The IndicF5 runtime is installed, but ai4bharat/IndicF5 is a "
                   "gated repository: accept its conditions on Hugging Face and "
                   "provide an HF_TOKEN, or copy a populated Hugging Face cache "
                   "onto this machine. Weights are downloaded on first use, not "
                   "by this endpoint.",
            required_for=required_for,
            detail={"implementation_available": "true",
                    "weights_cached": "false",
                    "token_configured": "false",
                    "repo_id": INDICF5_REPO_ID},
        )

    return ComponentStatus(
        name="indicf5", label="IndicF5 (Indic voice cloning)",
        state=ComponentState.MODEL_NOT_DOWNLOADED, available=False,
        reason="The IndicF5 runtime is installed and a token is configured, but "
               "the checkpoint is not cached yet; it is downloaded on first use.",
        required_for=required_for,
        detail={"implementation_available": "true",
                "weights_cached": "false",
                "token_configured": "true",
                "repo_id": INDICF5_REPO_ID},
    )


def _probe_voicefixer() -> ComponentStatus:
    if module_available("voicefixer"):
        return ComponentStatus(
            name="voicefixer", label="VoiceFixer audio enhancement",
            state=ComponentState.AVAILABLE, available=True,
            reason="VoiceFixer is installed.",
            required_for=["audio_enhancement"],
        )
    return ComponentStatus(
        name="voicefixer", label="VoiceFixer audio enhancement",
        state=ComponentState.NOT_INSTALLED, available=False,
        reason="The 'voicefixer' package is not installed. It is optional and "
               "has broken package metadata under pip >= 24.1, so it is "
               "installed manually.",
        required_for=["audio_enhancement"],
    )


def _probe_rubberband() -> ComponentStatus:
    if not module_available("pyrubberband"):
        return ComponentStatus(
            name="rubberband", label="Rubberband time-stretch",
            state=ComponentState.NOT_INSTALLED, available=False,
            reason="The 'pyrubberband' package is not installed; time-stretch "
                   "falls back to a librosa-based method.",
            required_for=["time_stretch"],
        )
    if not shutil.which("rubberband") and sys.platform == "win32":
        return ComponentStatus(
            name="rubberband", label="Rubberband time-stretch",
            state=ComponentState.UNAVAILABLE, available=False,
            reason="pyrubberband is installed but the 'rubberband' executable is "
                   "not on PATH, so stretching will fall back to librosa.",
            required_for=["time_stretch"],
        )
    return ComponentStatus(
        name="rubberband", label="Rubberband time-stretch",
        state=ComponentState.AVAILABLE, available=True,
        reason="pyrubberband is installed.",
        required_for=["time_stretch"],
    )


def _probe_xtts() -> ComponentStatus:
    if module_available("TTS"):
        return ComponentStatus(
            name="xtts", label="XTTS-v2 voice cloning",
            state=ComponentState.AVAILABLE, available=True,
            reason="The Coqui TTS package is installed; XTTS-v2 weights download "
                   "on first use.",
            required_for=["tts", "cloning"],
        )
    return ComponentStatus(
        name="xtts", label="XTTS-v2 voice cloning",
        state=ComponentState.NOT_INSTALLED, available=False,
        reason="The Coqui 'TTS' package is not installed.",
        required_for=["tts", "cloning"],
    )


def _probe_f5() -> ComponentStatus:
    if module_available("f5_tts"):
        return ComponentStatus(
            name="f5", label="F5-TTS voice cloning", state=ComponentState.AVAILABLE,
            available=True, reason="f5-tts is installed.",
            required_for=["tts", "cloning"],
        )
    return ComponentStatus(
        name="f5", label="F5-TTS voice cloning", state=ComponentState.NOT_INSTALLED,
        available=False, reason="The 'f5-tts' package is not installed.",
        required_for=["tts", "cloning"],
    )


def _probe_vibevoice() -> ComponentStatus:
    if not module_available("torch"):
        return ComponentStatus(
            name="vibevoice", label="VibeVoice", state=ComponentState.NOT_INSTALLED,
            available=False, reason="PyTorch is not installed.",
            required_for=["tts", "cloning"],
        )
    if module_available("transformers"):
        return ComponentStatus(
            name="vibevoice", label="VibeVoice", state=ComponentState.AVAILABLE,
            available=True,
            reason="VibeVoice runs through transformers; weights download on "
                   "first use.",
            required_for=["tts", "cloning"],
        )
    return ComponentStatus(
        name="vibevoice", label="VibeVoice", state=ComponentState.NOT_INSTALLED,
        available=False, reason="The 'transformers' package is not installed.",
        required_for=["tts", "cloning"],
    )


def _probe_faster_whisper() -> ComponentStatus:
    if module_available("faster_whisper") and module_available("ctranslate2"):
        return ComponentStatus(
            name="faster_whisper", label="faster-whisper (speech-to-text)",
            state=ComponentState.AVAILABLE, available=True,
            reason="faster-whisper and CTranslate2 are installed; the Whisper "
                   "weights download on first use.",
            required_for=["transcription"],
        )
    return ComponentStatus(
        name="faster_whisper", label="faster-whisper (speech-to-text)",
        state=ComponentState.NOT_INSTALLED, available=False,
        reason="'faster-whisper' and 'ctranslate2' are not both installed.",
        required_for=["transcription"],
    )


def _probe_pyannote() -> ComponentStatus:
    if module_available("pyannote.audio") or module_available("pyannote"):
        return ComponentStatus(
            name="pyannote", label="PyAnnote diarization",
            state=ComponentState.AVAILABLE, available=True,
            reason="PyAnnote is installed. Its pipeline is gated on Hugging "
                   "Face, so diarization additionally needs an HF_TOKEN.",
            required_for=["diarization"],
            detail={"token_required": "true"},
        )
    return ComponentStatus(
        name="pyannote", label="PyAnnote diarization",
        state=ComponentState.NOT_INSTALLED, available=False,
        reason="The 'pyannote.audio' package is not installed.",
        required_for=["diarization"],
    )


def _probe_demucs() -> ComponentStatus:
    if module_available("demucs"):
        return ComponentStatus(
            name="demucs", label="Demucs audio separation",
            state=ComponentState.AVAILABLE, available=True,
            reason="Demucs is installed; separation weights download on first use.",
            required_for=["separation"],
        )
    return ComponentStatus(
        name="demucs", label="Demucs audio separation",
        state=ComponentState.NOT_INSTALLED, available=False,
        reason="The 'demucs' package is not installed.",
        required_for=["separation"],
    )


def _probe_nllb() -> ComponentStatus:
    """The local NLLB translation backend: needs transformers + torch."""
    missing = [p for p in ("transformers", "torch", "sentencepiece")
               if not module_available(p)]
    if missing:
        return ComponentStatus(
            name="nllb", label="NLLB-200 (local translation)",
            state=ComponentState.NOT_INSTALLED, available=False,
            reason=f"Missing packages: {', '.join(missing)}.",
            required_for=["translation"],
        )
    return ComponentStatus(
        name="nllb", label="NLLB-200 (local translation)",
        state=ComponentState.AVAILABLE, available=True,
        reason="The NLLB runtime is installed; model weights download on first "
               "use.",
        required_for=["translation"],
    )


def _probe_llm_translation() -> ComponentStatus:
    """
    The local LLM translators (HY-MT, Llama, ALMA).

    `bitsandbytes` is Linux-only, so on Windows the quantised path is genuinely
    unsupported rather than merely uninstalled — which is what the
    `unsupported_platform` state is for.
    """
    missing = [p for p in ("transformers", "torch", "accelerate")
               if not module_available(p)]
    if missing:
        return ComponentStatus(
            name="llm_translation", label="Local LLM translation",
            state=ComponentState.NOT_INSTALLED, available=False,
            reason=f"Missing packages: {', '.join(missing)}.",
            required_for=["translation"],
        )
    if sys.platform == "win32" and not module_available("bitsandbytes"):
        return ComponentStatus(
            name="llm_translation", label="Local LLM translation",
            state=ComponentState.UNSUPPORTED_PLATFORM, available=False,
            reason="bitsandbytes has no official Windows build, so quantised "
                   "local LLM translation is not supported on this platform.",
            required_for=["translation"],
        )
    return ComponentStatus(
        name="llm_translation", label="Local LLM translation",
        state=ComponentState.AVAILABLE, available=True,
        reason="The LLM translation runtime is installed; model weights "
               "download on first use.",
        required_for=["translation"],
    )


def _probe_google_translate() -> ComponentStatus:
    if module_available("deep_translator"):
        return ComponentStatus(
            name="google_translate", label="Google Translate (online)",
            state=ComponentState.AVAILABLE, available=True,
            reason="deep-translator is installed. This backend is online and "
                   "needs network access.",
            required_for=["translation"], detail={"network_required": "true"},
        )
    return ComponentStatus(
        name="google_translate", label="Google Translate (online)",
        state=ComponentState.NOT_INSTALLED, available=False,
        reason="The 'deep-translator' package is not installed.",
        required_for=["translation"],
    )


def _probe_easyocr() -> ComponentStatus:
    if module_available("easyocr"):
        return ComponentStatus(
            name="easyocr", label="EasyOCR (visual text)",
            state=ComponentState.AVAILABLE, available=True,
            reason="EasyOCR is installed; its detection weights download on "
                   "first use.",
            required_for=["visual_translation"],
        )
    return ComponentStatus(
        name="easyocr", label="EasyOCR (visual text)",
        state=ComponentState.NOT_INSTALLED, available=False,
        reason="The 'easyocr' package is not installed.",
        required_for=["visual_translation"],
    )


def _probe_paddleocr() -> ComponentStatus:
    if not (module_available("paddleocr") and module_available("paddle")):
        return ComponentStatus(
            name="paddleocr", label="PaddleOCR (visual text)",
            state=ComponentState.NOT_INSTALLED, available=False,
            reason="'paddleocr' and 'paddlepaddle' are not both installed.",
            required_for=["visual_translation"],
        )
    return ComponentStatus(
        name="paddleocr", label="PaddleOCR (visual text)",
        state=ComponentState.AVAILABLE, available=True,
        reason="PaddleOCR is installed.",
        required_for=["visual_translation"],
    )


def gpu_state() -> ComponentState:
    """
    Availability of a usable CUDA GPU.

    Kept separate from the component probes because it is not a package: it is
    whether TensorRT acceleration and GPU-only backends can run at all. A CPU-only
    machine is `unavailable`, not an error.
    """
    info = torch_runtime_info()
    if not module_available("torch"):
        return ComponentState.NOT_INSTALLED
    if info.get("cuda_available"):
        return ComponentState.AVAILABLE
    return ComponentState.UNAVAILABLE


# name -> probe. Order is the order `/api/models` reports.
_PROBES: dict[str, Callable[[], ComponentStatus]] = {
    "ffmpeg": _probe_ffmpeg,
    "edge_tts": _probe_edge_tts,
    "piper": _probe_piper,
    "indicf5": _probe_indicf5,
    "xtts": _probe_xtts,
    "f5": _probe_f5,
    "vibevoice": _probe_vibevoice,
    "voicefixer": _probe_voicefixer,
    "rubberband": _probe_rubberband,
    "faster_whisper": _probe_faster_whisper,
    "demucs": _probe_demucs,
    "pyannote": _probe_pyannote,
    "speechbrain": _probe_speechbrain,
    "wav2lip": _probe_wav2lip,
    "gfpgan": _probe_gfpgan,
    "live_portrait": _probe_live_portrait,
    "nllb": _probe_nllb,
    "llm_translation": _probe_llm_translation,
    "google_translate": _probe_google_translate,
    "easyocr": _probe_easyocr,
    "paddleocr": _probe_paddleocr,
}


def probe_components() -> dict[str, ComponentStatus]:
    """
    Probe every component. A probe that raises is reported as `unknown` rather
    than failing the whole request: one broken probe must not take down
    `/api/models`.
    """
    results: dict[str, ComponentStatus] = {}
    for name, probe in _PROBES.items():
        try:
            results[name] = probe()
        except Exception as exc:  # noqa: BLE001 - probing must never 500
            logger.warning("Capability probe %r failed: %s", name, exc)
            results[name] = ComponentStatus(
                name=name, label=name, state=ComponentState.UNKNOWN,
                available=False,
                reason=f"Availability could not be determined ({type(exc).__name__}).",
            )
    return results


def component_states() -> dict[str, ComponentState]:
    """
    Compact name -> state view, used by `/api/capabilities` and `/api/options`.

    Includes the synthetic `cuda_gpu` entry so callers can reason about GPU-only
    options without a second probe.
    """
    states = {name: status.state for name, status in probe_components().items()}
    states["cuda_gpu"] = gpu_state()
    return states


def cloning_model_available(
    engine: Optional[str], states: Optional[dict[str, ComponentState]] = None
) -> bool:
    """
    Whether a cloning engine's weights are usable on this machine right now.

    `engine` is the value `languages.engine_for()` returns, which for the
    general backends is None — those clone only if the user's chosen backend
    happens to clone, which is a per-request decision, not a language property.

    `states` lets a caller that has already probed reuse the result instead of
    re-probing (which would shell out to ffmpeg again).
    """
    if not engine:
        return False
    if states is None:
        states = component_states()
    return states.get(engine) == ComponentState.AVAILABLE


# ---------------------------------------------------------------------------
# Environment / system information
# ---------------------------------------------------------------------------
def torch_runtime_info() -> dict:
    """
    Torch and CUDA facts, all best-effort.

    Returns empty values when torch is missing or is a test double, so
    `/api/system` stays truthful on a CPU-only or mocked box instead of
    reporting a fake GPU.
    """
    info: dict = {
        "torch_version": None, "cuda_available": False, "cuda_version": None,
        "cuda_runtime_version": None, "gpu_name": None, "gpu_count": 0,
        "gpu_memory_total_mb": None,
    }
    torch = sys.modules.get("torch")
    if torch is None:
        if not module_available("torch"):
            return info
        try:
            import torch  # noqa: F401
        except Exception:  # noqa: BLE001
            return info
    if _is_mocked(torch):
        return info

    try:
        info["torch_version"] = getattr(torch, "__version__", None)
    except Exception:  # noqa: BLE001
        pass
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return info
    try:
        info["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        info["cuda_runtime_version"] = getattr(
            getattr(torch, "version", None), "cuda", None)
    except Exception:  # noqa: BLE001
        pass
    if info["cuda_available"]:
        try:
            info["gpu_count"] = int(torch.cuda.device_count())
        except Exception:  # noqa: BLE001
            info["gpu_count"] = 0
        try:
            info["gpu_name"] = str(torch.cuda.get_device_name(0))
        except Exception:  # noqa: BLE001
            info["gpu_name"] = None
        try:
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_total_mb"] = int(props.total_memory // (1024 * 1024))
        except Exception:  # noqa: BLE001
            info["gpu_memory_total_mb"] = None
    return info


def app_version() -> Optional[str]:
    """
    Version from pyproject.toml, read without importing the package metadata.

    Falls back to a line scan so Python 3.10 (no `tomllib`) still works.
    """
    path = _repo_root() / "pyproject.toml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        import tomllib
        with path.open("rb") as fh:
            return tomllib.load(fh).get("project", {}).get("version")
    except Exception:  # noqa: BLE001 - tomllib missing on 3.10, or malformed
        pass
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version") and "=" in stripped:
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return None
