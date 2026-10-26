"""
Phase 5, layers 2 and 3: decide how long each segment should be, then measure
what actually came back.

`DurationModel` estimates the natural spoken length of a translated line, using
this segment's own evidence (what the original speaker said here and how long
they took) rather than a global words-per-minute table. That is what keeps the
estimate honest across scripts and across fast and slow speakers.

`plan_native_fit` turns those estimates into an optional per-segment duration
request for backends that can honour one (IndicF5's `fix_duration`). Backends
that cannot are left alone; they are corrected later by measurement + a bounded
stretch instead.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.utils import config
from src.utils import languages as L

logger = logging.getLogger(__name__)


def _safe_cps(code: Optional[str]) -> Optional[float]:
    """Registry speaking rate, or None for a code the registry does not know."""
    if not code:
        return None
    try:
        return L.chars_per_second(code)
    except Exception:  # noqa: BLE001 - an unknown code must not break a render
        return None


class DurationModel:
    """
    Estimates how long a piece of text takes to say *in this speaker's voice*.

    Two effects are kept separate:

      * script density  - characters per second typical for the TARGET language
      * speaker tempo   - how fast THIS speaker talks relative to typical for
                          the SOURCE language, measured from the reference clip

    When the original speaker's own line and its measured slot are available,
    that evidence beats any table: it is this speaker, this sentence, this
    recording. The estimate then reduces to `slot` scaled by how much longer the
    translation is, which degrades correctly to `natural == slot` for a
    same-language re-voice.
    """

    # Guards against a wild factor from a mistimed or mistranscribed reference.
    MIN_SPEAKER_FACTOR = 0.6
    MAX_SPEAKER_FACTOR = 1.7

    def __init__(self, ref_duration: float = 0.0, ref_text: str = "",
                 source_lang: Optional[str] = None, target_lang: str = "en"):
        self.source_cps = _safe_cps(source_lang)
        self.target_cps = _safe_cps(target_lang) or 14.0

        factor = 1.0
        if ref_duration > 0.5 and ref_text and len(ref_text.strip()) > 8:
            ref_cps = L.chars(ref_text) / ref_duration
            typical = self.source_cps or ref_cps
            if ref_cps > 0.1:
                factor = typical / ref_cps
        self.speaker_factor = min(self.MAX_SPEAKER_FACTOR,
                                  max(self.MIN_SPEAKER_FACTOR, factor))

    def estimate(self, text: str, source_text: Optional[str] = None,
                 slot: Optional[float] = None) -> float:
        """
        How long `text` should take to say.

        Prefers segment-level evidence; falls back to the cps table.
        """
        if (source_text and slot and slot > 0.05
                and len(source_text.strip()) > 1 and self.source_cps):
            ratio = L.chars(text) / L.chars(source_text)
            density = self.source_cps / self.target_cps
            return slot * ratio * density

        return (L.chars(text) / self.target_cps) * self.speaker_factor


def measure_reference_durations(speaker_ref_audio) -> dict[str, float]:
    """
    Duration in seconds of each speaker's reference clip.

    Needed because IndicF5's `fix_duration` is the TOTAL length of reference plus
    generated audio, so a caller that forgets to add the reference length gets
    every segment back roughly one reference too long. Reading the header is
    enough and costs nothing.
    """
    durations: dict[str, float] = {}
    for speaker, path in (speaker_ref_audio or {}).items():
        if not path:
            continue
        try:
            import soundfile as sf
            durations[speaker] = float(sf.info(str(path)).duration)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not measure reference for %s: %s", speaker, exc)
    return durations


def plan_native_fit(timings, model: DurationModel,
                    model_name: Optional[str] = None,
                    tolerance: Optional[float] = None,
                    ref_durations: Optional[dict[str, float]] = None) -> dict[int, float]:
    """
    Work out which segments to ask the backend to fit, and to what length.

    Returns `{segment_id: requested_seconds}`, where the requested value is the
    TOTAL duration the backend should produce: the speaker's reference clip plus
    the desired generated speech. IndicF5 counts both, so omitting the reference
    would make every segment come back one reference-length too long.

    Only backends that can honour an exact duration are asked; for everyone else
    this returns `{}` and the measurement + stretch path handles the mismatch.

    A segment whose reference length cannot be measured is skipped rather than
    guessed at, because a wrong total is worse than no request at all.

    A request is only made when the natural length is off by more than
    `tolerance`, so the model keeps its own prosody on the lines that fit.
    """
    from src.synthesis.capabilities import honours_duration

    if not honours_duration(model_name):
        return {}

    tol = config.TIMING_NATIVE_FIT_TOLERANCE if tolerance is None else tolerance
    requests: dict[int, float] = {}

    for t in timings:
        target = t.target_duration
        if target <= 0 or not t.translated_text:
            continue

        ref_dur = (ref_durations or {}).get(t.speaker_id)
        if ref_dur is None:
            # No measurable reference for this speaker: cannot compute a total.
            continue

        natural = model.estimate(t.translated_text, source_text=t.source_text,
                                 slot=target)
        if natural <= 0:
            continue
        if abs(natural - target) / target <= tol:
            continue

        # Never ask for a length outside what the model can render sensibly.
        lo = target * config.TIMING_MIN_STRETCH
        hi = target * config.TIMING_MAX_STRETCH
        desired = min(max(natural, lo), hi)
        requests[t.segment_id] = ref_dur + desired

    return requests
