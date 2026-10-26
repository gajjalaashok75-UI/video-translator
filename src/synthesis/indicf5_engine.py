"""
IndicF5 engine — reference-conditioned TTS for the 11 supported Indian languages.

ai4bharat/IndicF5 is a near-human polyglot TTS model (1417 h of Indian speech,
MIT licensed) that clones a speaker zero-shot from a reference clip plus the
transcript of that clip. It is the strongest reference-cloning backend available
for Telugu, Hindi, Tamil, Kannada, Malayalam, Bengali, Marathi, Gujarati,
Punjabi, Odia and Assamese.

This module owns the model itself. The thin `TTSBackend` adapter that plugs it
into the pipeline's backend registry lives in
`src/synthesis/backends/indicf5.py`.

Four things about this model are not obvious and each one fails silently or
confusingly if missed. All four are handled here:

1. CHECKPOINT KEY PREFIX. Every tensor in `model.safetensors` is named
   `ema_model._orig_mod.…` / `vocoder._orig_mod.…`, because the checkpoint was
   saved from a `torch.compile()`-wrapped module and compile inserts `_orig_mod`
   into the module path. The instantiated model has no such level, so a plain
   `from_pretrained` matches ZERO weights and leaves the DiT and vocoder
   randomly initialised. The result is audio of exactly the right length
   containing pure silence, reported as success. `_load_real_weights` strips the
   prefix, then refuses to proceed unless every tensor matched and the weights
   are finite and non-degenerate.

2. DEVICE AND SILENCE HANDLING IN `forward()`. The model's own `forward()`
   hardcodes `device="cpu"` and strips silence / re-normalises loudness, which
   fights the duration control this pipeline relies on. We therefore bypass it
   and call `f5_tts.infer.utils_infer.infer_batch_process` directly with a
   pre-loaded reference tensor.

3. TORCHAUDIO. `infer_process` routes through `torchaudio.load`, which on
   torchaudio >= 2.11 demands TorchCodec. Handing `infer_batch_process` a tensor
   avoids that path entirely; `src.utils.torch_compat` repairs `torchaudio.load`
   for the libraries that call it internally.

4. `ref_text` IS REQUIRED. It is the transcript of the reference clip and
   conditions the model's prosody. An empty `ref_text` does not raise, it just
   produces a worse voice. `synthesize` refuses an empty reference transcript
   unless the caller explicitly opts into a degraded run.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

from src.utils import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors — every one of these carries an actionable message
# ---------------------------------------------------------------------------
class IndicF5Error(RuntimeError):
    """Base class for IndicF5 failures."""


class IndicF5Unavailable(IndicF5Error):
    """A required package is missing, so the model cannot be used at all."""


class IndicF5AccessError(IndicF5Error):
    """The model could not be downloaded — usually a missing or rejected token."""


class IndicF5WeightError(IndicF5Error):
    """The checkpoint downloaded but did not map onto the model."""


class IndicF5ReferenceError(IndicF5Error):
    """The reference audio or its transcript is missing or unusable."""


class IndicF5SynthesisError(IndicF5Error):
    """Inference ran but did not produce usable audio."""


# ---------------------------------------------------------------------------
# Pure helpers (module level so they can be tested without torch installed)
# ---------------------------------------------------------------------------
def strip_orig_mod_prefix(state_dict: dict) -> dict:
    """
    Remove the `._orig_mod.` segment that `torch.compile()` inserts into module
    paths, so checkpoint keys match the instantiated model.

    `ema_model._orig_mod.transformer.x` -> `ema_model.transformer.x`
    """
    return {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}


def normalize_ref_text(text: str) -> str:
    """
    Condition the reference transcript the way IndicF5 expects.

    F5 concatenates `ref_text` and the text to speak into one sequence, so the
    reference needs terminal punctuation and a separating space. An empty
    transcript becomes "." rather than "", which keeps the model on its
    conditioned path instead of an unpunctuated one.
    """
    text = (text or "").strip()
    if not text:
        # A bare "." keeps the model on its punctuated path; the trailing space
        # still separates the reference span from the text to be spoken.
        return ". "
    if text[-1] not in ".?!।,":
        text += "."
    return text + " "


def seconds_per_byte(ref_voiced_seconds: float, ref_bytes: int) -> float:
    """Seconds of speech per byte of reference transcript."""
    return float(ref_voiced_seconds) / float(max(1, ref_bytes))


def clamp_seconds_per_byte(spb: float, ref_text: str, lang: str) -> float:
    """
    Keep the seconds-per-byte calibration inside a physically plausible band.

    A short or badly transcribed reference yields a wild rate, and every
    duration downstream inherits it. The expected rate is derivable: characters
    per second for the language, times the bytes-per-character of its script.
    Anything far from that is a broken reference, not an unusual speaker.

    Returns the (possibly clamped) rate; never raises.
    """
    from src.utils import languages as L

    try:
        cps = L.chars_per_second(lang)
    except Exception:  # noqa: BLE001 - unknown language: skip the sanity bound
        return spb

    text = (ref_text or "").strip()
    chars = max(1, len(text))
    bytes_per_char = len(text.encode("utf-8")) / chars
    if bytes_per_char <= 0 or cps <= 0:
        return spb

    expected = 1.0 / (cps * bytes_per_char)
    lo, hi = expected * 0.65, expected * 1.60
    if spb < lo or spb > hi:
        clamped = min(hi, max(lo, spb))
        logger.warning(
            "IndicF5 reference implies %.1f ms/byte, outside the plausible "
            "%.1f-%.1f for '%s' — clamping to %.1f. The reference clip is "
            "probably too short or mistranscribed.",
            spb * 1000, lo * 1000, hi * 1000, lang, clamped * 1000,
        )
        return clamped
    return spb


def voiced_fraction(waveform) -> float:
    """
    Fraction of a reference clip that is actually speech, used to calibrate the
    speaking rate on voiced time rather than wall time.

    A clip's own pauses (~13% of it on real footage) otherwise make every byte
    look slower than it is, and the model is handed more time than the words
    need — audible as gaps between phrases. Returns 1.0 when librosa is
    unavailable or the estimate is implausible.
    """
    try:
        import numpy as np
        import librosa

        y = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return 1.0
        sr = IndicF5Engine.SAMPLE_RATE
        rms = librosa.feature.rms(
            y=y, frame_length=int(sr * 0.02), hop_length=int(sr * 0.01)
        )[0]
        if rms.max() <= 0:
            return 1.0
        db = librosa.amplitude_to_db(rms, ref=np.max)
        frac = float((db > -35.0).mean())
        return frac if 0.3 < frac <= 1.0 else 1.0
    except Exception:  # noqa: BLE001
        return 1.0


def peak_amplitude(waveform) -> float:
    """Peak absolute amplitude, or 0.0 for an empty waveform."""
    try:
        import numpy as np

        arr = np.asarray(waveform, dtype=np.float32).reshape(-1)
        return float(np.max(np.abs(arr))) if arr.size else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def validate_reference_audio(ref_path: str | Path, min_duration: float = 3.0) -> float:
    """
    Check that a reference clip is usable for cloning and return its duration.

    Raises `IndicF5ReferenceError` with a specific reason rather than letting a
    bad reference degrade the clone silently.
    """
    if not ref_path:
        raise IndicF5ReferenceError(
            "No reference audio supplied. IndicF5 needs a reference clip to "
            "clone a voice; pass speaker_wav."
        )

    path = Path(ref_path)
    if not path.exists():
        raise IndicF5ReferenceError(f"Reference audio does not exist: {path}")

    try:
        import numpy as np
        import soundfile as sf

        info = sf.info(str(path))
    except Exception as exc:  # noqa: BLE001
        raise IndicF5ReferenceError(
            f"Reference audio could not be read ({path}): {exc}"
        ) from exc

    if info.frames <= 0 or info.samplerate <= 0:
        raise IndicF5ReferenceError(f"Reference audio is empty: {path}")

    duration = info.frames / float(info.samplerate)
    if duration < min_duration:
        raise IndicF5ReferenceError(
            f"Reference audio is too short for cloning: {duration:.2f}s "
            f"< {min_duration:.1f}s ({path}). IndicF5 needs a stable speaker "
            f"embedding; extract a longer clean speech window."
        )

    try:
        import numpy as np
        import soundfile as sf

        data, _ = sf.read(str(path), dtype="float32", frames=int(info.samplerate * 10))
        if data.ndim > 1:
            data = data.mean(axis=1)
        rms = float(np.sqrt(np.mean(data ** 2))) if data.size else 0.0
    except Exception as exc:  # noqa: BLE001
        raise IndicF5ReferenceError(
            f"Reference audio could not be analysed ({path}): {exc}"
        ) from exc

    if rms < config.REFERENCE_RMS_THRESHOLD:
        raise IndicF5ReferenceError(
            f"Reference audio is effectively silent (RMS={rms:.4f} < "
            f"{config.REFERENCE_RMS_THRESHOLD}): {path}"
        )

    return duration


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class IndicF5Engine:
    """Warm, lazily-loaded IndicF5 singleton wrapper."""

    SAMPLE_RATE = 24000
    REPO_ID = "ai4bharat/IndicF5"

    # Below this peak amplitude the model produced no audio. Length alone is not
    # evidence of content: a mis-loaded checkpoint yields exactly-right-length
    # silence.
    SILENCE_PEAK = 1e-4

    # Default flow-matching steps. Cost is linear in this; 16 keeps latency
    # reasonable, raise it for final renders.
    DEFAULT_NFE_STEP = 16

    def __init__(self, nfe_step: Optional[int] = None, seed: Optional[int] = None):
        self._model = None
        self._lock = threading.RLock()
        self._ref_cache: dict = {}
        self._spb_cache: dict = {}
        self.device: Optional[str] = None

        self.nfe_step = int(
            nfe_step or os.getenv("INDICF5_NFE_STEP") or self.DEFAULT_NFE_STEP
        )

        # Flow matching starts from noise, so identical inputs give different
        # audio on every run. A fixed default seed makes a run reproducible,
        # which is what allows A/B comparison and per-segment re-rolls.
        # Set INDICF5_SEED= (empty) to opt back into stochastic sampling.
        env_seed = os.getenv("INDICF5_SEED", "1234")
        self.seed = int(seed) if seed is not None else (
            int(env_seed) if env_seed.strip() else None
        )

    # -- lifecycle ----------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _ensure_mps_fallback() -> None:
        """Some F5/vocos ops have no MPS kernel; let torch fall back silently."""
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    @staticmethod
    def pick_device() -> str:
        """FORCE_DEVICE override, else the main repo's device decision."""
        forced = os.getenv("FORCE_DEVICE")
        if forced:
            return forced
        try:
            return config.DEVICE
        except Exception:  # noqa: BLE001
            return "cpu"

    def load(self):
        """Load the model (idempotent, thread-safe). Returns the model."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            from src.utils.torch_compat import (
                allow_torch_load_globals,
                patch_torchaudio_load,
            )

            self._ensure_mps_fallback()
            patch_torchaudio_load()
            allow_torch_load_globals()

            try:
                import torch  # noqa: F401 - torch must load before transformers
                from transformers import AutoModel
            except ImportError as exc:
                raise IndicF5Unavailable(
                    "IndicF5 requires torch and transformers. Install them with "
                    "`pip install -e \".[all]\"`. Underlying error: "
                    f"{exc}"
                ) from exc

            self.device = self.pick_device()
            logger.info("Loading %s on %s...", self.REPO_ID, self.device)

            try:
                model = AutoModel.from_pretrained(
                    self.REPO_ID,
                    trust_remote_code=True,
                    token=os.getenv("HF_TOKEN") or None,
                )
            except Exception as exc:  # noqa: BLE001
                raise self._classify_load_error(exc) from exc

            self._load_real_weights(model)

            try:
                model.ema_model.to(self.device)
                model.vocoder.to(self.device)
                model.ema_model.eval()
                # The model's own forward() reads this; set it so any internal
                # path that does use forward() lands on the right device.
                model.runtime_device = self.device
            except Exception as exc:  # noqa: BLE001
                raise IndicF5Error(
                    f"IndicF5 loaded but could not be moved to {self.device}: {exc}"
                ) from exc

            self._model = model
            logger.info("IndicF5 ready on %s (nfe_step=%d, seed=%s)",
                        self.device, self.nfe_step, self.seed)
            return self._model

    @staticmethod
    def _classify_load_error(exc: Exception) -> IndicF5Error:
        """Turn a download/instantiation failure into an actionable error."""
        text = f"{type(exc).__name__}: {exc}"
        low = text.lower()

        if "gated" in low or "401" in low or "403" in low or "unauthorized" in low \
                or "token" in low or "authenticat" in low or "permission" in low:
            return IndicF5AccessError(
                "Could not access ai4bharat/IndicF5. The repository is gated: "
                "accept its conditions at "
                "https://huggingface.co/ai4bharat/IndicF5 and supply a Hugging "
                "Face token (HF_TOKEN, or the token field in the UI). "
                f"Underlying error: {text}"
            )
        if "connection" in low or "timed out" in low or "network" in low \
                or "resolve" in low or "offline" in low:
            return IndicF5AccessError(
                "Could not download ai4bharat/IndicF5 — no network access to the "
                "Hugging Face Hub. Download it once on a connected machine and "
                f"copy the HF cache. Underlying error: {text}"
            )
        return IndicF5Error(
            f"Failed to load ai4bharat/IndicF5: {text}"
        )

    def _load_real_weights(self, model) -> None:
        """
        Load the checkpoint by hand, stripping the `_orig_mod.` prefix.

        A plain `from_pretrained` matches none of the 447 tensors and leaves the
        DiT and vocoder randomly initialised, which produces silent audio of the
        correct length. This raises instead of letting that happen.
        """
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as exc:
            raise IndicF5Unavailable(
                "IndicF5 needs huggingface_hub and safetensors to load its "
                f"checkpoint. Underlying error: {exc}"
            ) from exc

        try:
            path = hf_hub_download(
                self.REPO_ID, "model.safetensors",
                token=os.getenv("HF_TOKEN") or None,
            )
        except Exception as exc:  # noqa: BLE001
            raise self._classify_load_error(exc) from exc

        try:
            raw = load_file(path)
        except Exception as exc:  # noqa: BLE001
            raise IndicF5WeightError(
                f"Could not read the IndicF5 checkpoint at {path}: {exc}"
            ) from exc

        state = strip_orig_mod_prefix(raw)

        try:
            result = model.load_state_dict(state, strict=False)
        except Exception as exc:  # noqa: BLE001
            raise IndicF5WeightError(
                f"IndicF5 checkpoint did not fit the model: {exc}. The upstream "
                f"model definition may have changed."
            ) from exc

        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        if missing or unexpected:
            raise IndicF5WeightError(
                "IndicF5 checkpoint did not map cleanly onto the model "
                f"({len(missing)} missing, {len(unexpected)} unexpected). "
                f"First missing: {missing[:3]}. The model would generate "
                "silence, so this is treated as fatal."
            )

        # A NaN or all-zero probe tensor means the weights never landed. Fail
        # loudly rather than emit silence that looks like a successful run.
        try:
            probe = model.ema_model.transformer.proj_out.weight
            if torch.isnan(probe).any() or float(probe.std()) == 0.0:
                raise IndicF5WeightError(
                    "IndicF5 weights are NaN or constant after loading — the "
                    "model would generate silence. The checkpoint is corrupt or "
                    "the key-prefix handling is wrong."
                )
        except IndicF5WeightError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("IndicF5 weight probe skipped: %s", exc)

        logger.info("IndicF5 loaded %d tensors (stripped '_orig_mod.' prefix).", len(state))

    def unload(self) -> None:
        """Drop the model and release accelerator memory."""
        with self._lock:
            if self._model is None:
                return
            logger.info("Unloading IndicF5...")
            try:
                del self._model
            finally:
                self._model = None
            self._ref_cache.clear()
            self._spb_cache.clear()
            try:
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    # -- reference handling -------------------------------------------------
    def _load_ref(self, ref_path: str | Path):
        """
        Load and cache the reference waveform as a (1, N) float32 tensor at
        24 kHz. Cached so a long video resamples its reference once, not once
        per segment.
        """
        import torch
        import soundfile as sf

        path = Path(ref_path)
        key = (str(path), path.stat().st_mtime)

        with self._lock:
            cached = self._ref_cache.get(key)
            if cached is not None:
                return cached

            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)

            tensor = torch.from_numpy(data).unsqueeze(0)
            if sr != self.SAMPLE_RATE:
                import torchaudio

                tensor = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)(tensor)
                sr = self.SAMPLE_RATE

            # Only ever one reference per job, so keep the cache tiny.
            self._ref_cache.clear()
            self._ref_cache[key] = (tensor, sr)
            return tensor, sr

    def _seconds_per_byte(self, ref_path: str | Path, ref_bytes: int) -> float:
        """
        Seconds of SPEECH per byte of text, from the reference clip.

        Calibrates on the reference's voiced time, not its wall time: the clip
        contains the speaker's own pauses and counting those makes every byte
        look slower than it is.
        """
        path = Path(ref_path)
        key = (str(path), path.stat().st_mtime, int(ref_bytes))
        with self._lock:
            cached = self._spb_cache.get(key)
            if cached is not None:
                return cached

        wave, sr = self._load_ref(path)
        wall = wave.shape[-1] / float(sr)
        frac = voiced_fraction(wave.reshape(-1).cpu().numpy())
        rate = seconds_per_byte(wall * frac, ref_bytes)

        with self._lock:
            self._spb_cache.clear()
            self._spb_cache[key] = rate
        return rate

    def natural_duration(self, text: str, ref_path: str | Path, ref_text: str,
                         lang: str = "hi") -> float:
        """
        How long this utterance will take if the model is left alone, in
        seconds.

        Mirrors the model's own internal heuristic
        (`duration = ref_len + ref_len / ref_text_len * gen_text_len`, byte
        counted). Asking for a duration that matches what the model already
        wants means it never has to internally compress or pad, which is the
        difference between natural delivery and a rushed, garbled one.
        """
        ref_norm = normalize_ref_text(ref_text)
        ref_bytes = max(1, len(ref_norm.encode("utf-8")))
        gen_bytes = len((text or "").strip().encode("utf-8"))
        spb = self._seconds_per_byte(ref_path, ref_bytes)
        spb = clamp_seconds_per_byte(spb, ref_norm, lang)
        return spb * gen_bytes

    def _seed_rng(self) -> None:
        """Reset the RNG so a given input always yields the same audio."""
        if self.seed is None:
            return
        try:
            import torch

            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            try:
                torch.mps.manual_seed(self.seed)
            except Exception:  # noqa: BLE001 - not on MPS
                pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not seed RNG: %s", exc)

    # -- synthesis ----------------------------------------------------------
    def synthesize(self, text: str, ref_path: str | Path, ref_text: str,
                   lang: str = "hi", speed: float = 1.0,
                   fix_duration: Optional[float] = None,
                   require_ref_text: bool = True):
        """
        Generate one utterance. Returns (float32 waveform, sample_rate).

        `ref_text` is the transcript of the reference clip and is required for a
        good clone. Pass `require_ref_text=False` only when a caller knowingly
        accepts a degraded (unconditioned) voice.

        `fix_duration` asks the model for a specific TOTAL length (reference plus
        generated) in seconds; it tracks the request to roughly 10 ms, which is
        far better than stretching the result afterwards.
        """
        text = (text or "").strip()
        if not text:
            import numpy as np

            return np.zeros(0, dtype=np.float32), self.SAMPLE_RATE

        if not (ref_text or "").strip() and require_ref_text:
            raise IndicF5ReferenceError(
                "IndicF5 needs the transcript of the reference clip (ref_text). "
                "Without it the clone is conditioned on nothing and the voice "
                "quality drops sharply. Supply ref_text, or pass "
                "require_ref_text=False to accept the degraded result."
            )

        validate_reference_audio(ref_path, min_duration=config.F5_MIN_DURATION)

        model = self.load()

        try:
            from f5_tts.infer.utils_infer import infer_batch_process
        except ImportError as exc:
            raise IndicF5Unavailable(
                "IndicF5 needs the f5-tts package for inference. Install it with "
                "`pip install f5-tts vocos`. Underlying error: "
                f"{exc}"
            ) from exc

        self._seed_rng()
        audio_t, sr = self._load_ref(ref_path)
        ref_norm = normalize_ref_text(ref_text)

        kwargs = dict(
            mel_spec_type="vocos",
            device=self.device,
            nfe_step=self.nfe_step,
            cross_fade_duration=0.0,  # single batch: nothing to cross-fade
            speed=speed,
            progress=None,
        )
        if fix_duration is not None and fix_duration > 0:
            kwargs["fix_duration"] = float(fix_duration)

        try:
            wave, out_sr, _ = infer_batch_process(
                (audio_t, sr),
                ref_norm,
                [text],
                model.ema_model,
                model.vocoder,
                **kwargs,
            )
        except IndicF5Error:
            raise
        except Exception as exc:  # noqa: BLE001
            raise IndicF5SynthesisError(
                f"IndicF5 inference failed: {type(exc).__name__}: {exc}"
            ) from exc

        import numpy as np

        wave = np.asarray(wave, dtype=np.float32).reshape(-1)

        peak = peak_amplitude(wave)
        if wave.size == 0 or peak < self.SILENCE_PEAK:
            raise IndicF5SynthesisError(
                "IndicF5 produced silence (peak %.2e). This usually means the "
                "checkpoint weights did not load — check the '_orig_mod.' key "
                "handling and the HF token, then retry." % peak
            )

        return wave, out_sr


# ---------------------------------------------------------------------------
# Warm singleton registry (mirrors the pipeline's one-heavy-model-at-a-time rule)
# ---------------------------------------------------------------------------
_ENGINE: Optional[IndicF5Engine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine(nfe_step: Optional[int] = None) -> IndicF5Engine:
    """Return the shared warm engine, constructing it once."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = IndicF5Engine(nfe_step=nfe_step)
        return _ENGINE


def unload_engine() -> None:
    """Unload and drop the shared engine, so the next call reloads."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            _ENGINE.unload()
            _ENGINE = None


def engine_available() -> tuple[bool, str]:
    """
    Whether IndicF5 can run on this machine, and why not when it cannot.

    Lets the UI refuse an Indic dub up front instead of failing minutes into a
    render.
    """
    import importlib.util

    for module in ("torch", "transformers", "f5_tts", "vocos", "safetensors",
                   "huggingface_hub"):
        if importlib.util.find_spec(module) is None:
            return False, (
                f"IndicF5 requires the '{module}' package, which is not "
                f"installed. Install with `pip install f5-tts vocos`."
            )
    return True, ""
