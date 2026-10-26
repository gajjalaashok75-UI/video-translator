"""
IndicF5 TTS backend — adapts `src.synthesis.indicf5_engine` to the pipeline's
`TTSBackend` interface.

This is deliberately thin. All model behaviour lives in the engine; this class
only translates the pipeline's task shape into engine calls and back. Routing
(which target languages use IndicF5) lives in `src/utils/languages.py`, and the
backend registry lives in `src/synthesis/tts.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.synthesis.backends.base import TTSBackend
from src.synthesis import indicf5_engine
from src.synthesis.indicf5_engine import IndicF5Error, IndicF5ReferenceError
from src.utils import config
from src.utils import languages as L

logger = logging.getLogger(__name__)


class IndicF5Backend(TTSBackend):
    """Reference-conditioned TTS for the 11 Indian languages IndicF5 supports."""

    def __init__(self):
        self.engine = indicf5_engine.get_engine()

    # -- lifecycle ----------------------------------------------------------
    def load_model(self):
        self.engine.load()

    def unload_model(self):
        indicf5_engine.unload_engine()
        # Re-bind so a later load() goes through the registry again.
        self.engine = indicf5_engine.get_engine()

    def supports_language(self, language_code: str) -> bool:
        try:
            return L.normalize(language_code) in L.INDICF5_NATIVE
        except L.UnsupportedLanguage:
            return False

    # -- generation ---------------------------------------------------------
    def generate(self, text, output_path: str | Path, language="en",
                 speaker_wav=None, **kwargs) -> str | Path | None:
        """
        Generate one segment in the reference speaker's voice.

        Recognised kwargs:
            ref_text (str): transcript of the reference clip. Required for a
                good clone; the engine raises without it unless
                `allow_degraded_ref_text` is set.
            fix_duration (float): total (reference + generated) seconds to ask
                for, which is how the timeline controls length precisely.
            allow_degraded_ref_text (bool): proceed with an empty transcript.
            speed (float): the model's own duration prior; keep near 1.0 and
                prefer `fix_duration`.
        """
        lang = self._resolve_synth_lang(language)

        ref_text = kwargs.get("ref_text") or kwargs.get("reference_text") or ""
        allow_degraded = bool(kwargs.get("allow_degraded_ref_text", False))
        fix_duration = kwargs.get("fix_duration")
        speed = float(kwargs.get("speed", 1.0) or 1.0)

        if not speaker_wav:
            raise IndicF5ReferenceError(
                f"IndicF5 cannot synthesize '{lang}' without a reference clip. "
                f"No speaker_wav was supplied for this segment."
            )

        if not (ref_text or "").strip():
            message = (
                f"IndicF5 received no ref_text for language '{lang}'. The clone "
                f"will be conditioned on nothing and the voice quality will drop."
            )
            if allow_degraded:
                logger.warning(message + " Continuing because the caller opted in.")
            else:
                raise IndicF5ReferenceError(
                    message + " Supply the transcript of the reference clip, or "
                    "pass allow_degraded_ref_text=True to accept the degraded result."
                )

        logger.info(
            "IndicF5: lang=%s, ref=%s, ref_text=%d chars, fix_duration=%s",
            lang, Path(speaker_wav).name, len((ref_text or "").strip()),
            f"{fix_duration:.2f}s" if fix_duration else "auto",
        )

        wave, sr = self.engine.synthesize(
            text=text,
            ref_path=speaker_wav,
            ref_text=ref_text,
            lang=lang,
            speed=speed,
            fix_duration=fix_duration,
            require_ref_text=not allow_degraded,
        )

        return self._write_wav(wave, sr, output_path)

    def generate_batch(self, tasks: list) -> list:
        """
        Sequential batch.

        IndicF5 is a single-reference diffusion-style model: batching several
        utterances into one call does not amortise anything (measured), and each
        segment may carry a different speaker reference, so a per-task loop is
        both simpler and no slower.
        """
        results = []
        for task in tasks:
            try:
                results.append(self.generate(**task))
            except IndicF5Error as exc:
                logger.error("IndicF5 failed for one segment: %s", exc)
                results.append(None)
        return results

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _resolve_synth_lang(language: str) -> str:
        """
        Map whatever the pipeline passed as `language` onto an IndicF5 code.

        Accepts a short code ('te'), a flores code ('tel_Telu') or a display
        name ('Telugu'). Falls back to the raw value so the engine's own
        validation reports the real problem.
        """
        if not language:
            return "hi"
        try:
            return L.synth_lang(language)
        except L.UnsupportedLanguage:
            return str(language)

    @staticmethod
    def _write_wav(wave, sr, output_path) -> str:
        import soundfile as sf

        out = Path(output_path)
        if out.suffix.lower() != ".wav":
            out = out.with_suffix(".wav")
        out.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(out), wave, sr)

        peak = indicf5_engine.peak_amplitude(wave)
        if peak < indicf5_engine.IndicF5Engine.SILENCE_PEAK:
            raise indicf5_engine.IndicF5SynthesisError(
                f"IndicF5 wrote a silent file to {out} (peak {peak:.2e})."
            )

        if not out.exists() or out.stat().st_size < config.TTS_MIN_AUDIO_SIZE:
            raise indicf5_engine.IndicF5SynthesisError(
                f"IndicF5 produced no usable output at {out}."
            )
        return str(out)
