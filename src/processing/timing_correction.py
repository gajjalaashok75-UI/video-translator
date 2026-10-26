"""
Phase 5, layer 4: measure a generated segment, decide one correction, apply it.

This module owns exactly one decision per segment, so nothing downstream has to
guess again. The previous pipeline corrected in two places with different clamps
(`merge_segments` used [0.8, 1.35] while `sync_segment` used [0.25, 4.0]) and
applied both, which is how a single line could end up stretched twice.

The rule here is simple: a segment gets at most ONE lossy transform, and the
band it falls into says which one.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.processing.timing import (
    BAND_EXTREME_SHORT,
    BAND_OK,
    BAND_SHORT,
    SegmentTiming,
    clamp_stretch,
    classify_ratio,
)

logger = logging.getLogger(__name__)


def allowed_end(timing: SegmentTiming, next_timing: Optional[SegmentTiming],
                total_duration: float) -> float:
    """
    The latest moment this segment's audio may occupy.

    Preserving intentional overlap is the point. If the ORIGINAL segments
    overlapped (the next one starts before this one ends), the two speakers were
    talking over each other on purpose and that must survive, so the window
    extends to this segment's own end. If the original was sequential, the next
    speaker's start is a hard boundary and any overflow is a collision.
    """
    if next_timing is None:
        return max(total_duration, timing.end)
    return max(next_timing.start, timing.end)


class TimingCorrector:
    """
    Decide one segment's final duration from its measured duration and window.

    Returns a `(action, stretch_ratio, final_duration)` triple. `stretch_ratio`
    follows the generated/target convention, so >1 means "play faster".
    """

    def __init__(self, enable_stretch: bool = True):
        self.enable_stretch = enable_stretch

    def correct(self, timing: SegmentTiming, actual: float) -> tuple[str, float, float]:
        """
        Decide the one correction a segment gets.

        Returns `(action, stretch_ratio, expected_duration)`. The action is one
        of `none`, `native_fit`, `compressed`, `uncorrected`; the caller decides
        whether the result then needs extending into a pause or trimming,
        because only the caller knows the next speaker's start.
        """
        target = timing.target_duration

        if target <= 0:
            return "zero_window", 1.0, actual
        if actual <= 0:
            return "failed", 1.0, 0.0

        # The backend already fitted this segment to its window. Stretching it
        # again would undo the model's own (better) timing work, so stop here.
        if timing.duration_honoured:
            return "native_fit", 1.0, actual

        ratio = actual / target
        band = classify_ratio(ratio)

        if band == BAND_OK:
            return "none", 1.0, actual

        if not self.enable_stretch:
            return "uncorrected", 1.0, actual

        # Only overruns are corrected. A line that came back SHORTER than its
        # window is not a problem: the speaker simply paused, and that pause is
        # the natural rhythm of the original. Slowing speech down to fill the
        # gap is the classic dubbing mistake that makes a voice sound drugged,
        # so it is deliberately not done here.
        if band in (BAND_SHORT, BAND_EXTREME_SHORT):
            return "none", 1.0, actual

        stretch = clamp_stretch(ratio)
        return "compressed", stretch, actual / stretch


def classify_fit(target: float, final: float, limit: float) -> str:
    """
    Describe how a segment ended up sitting relative to its window.

      fits      - inside its own dialogue window
      extended  - ran past the window but into a following PAUSE, which is
                  natural; the original speaker paused there and the dub may use
                  it. Not a defect.
      collision - reached the next speaker's start; the tail was trimmed.
    """
    eps = 1e-3
    if final <= target + eps:
        return "fits"
    if final <= limit + eps:
        return "extended"
    return "collision"


def apply_stretch(wave, sr: int, ratio: float):
    """
    Time-stretch `wave` by `ratio` (generated/target, >1 shortens), pitch intact.

    Tries Rubberband first for quality, then librosa. Returns
    `(wave, method)`; `method` is `"none"` when nothing was applied, which the
    caller must treat as "the ratio was not achieved" rather than success.
    """
    if abs(ratio - 1.0) < 1e-3 or wave is None or len(wave) == 0:
        return wave, "none"

    import numpy as np

    y = np.asarray(wave, dtype=np.float64)
    if y.ndim > 1:
        y = y[:, 0]

    # Rubberband preserves transients best, but the binary is optional.
    try:
        import pyrubberband as pyrb
        return pyrb.time_stretch(y, sr, ratio), "rubberband"
    except Exception:
        pass

    try:
        import librosa
        # librosa's `rate` is playback rate, so >1 shortens: same convention.
        return librosa.effects.time_stretch(y, rate=ratio).astype(np.float64), "librosa"
    except Exception as exc:  # noqa: BLE001
        logger.warning("No time-stretcher available (%s); leaving audio unstretched.", exc)
        return wave, "none"


def trim_with_fade(wave, target_samples: int):
    """
    Cut `wave` down to `target_samples`, fading the tail so the cut cannot click.

    Only used for the residual overflow a clamped stretch could not remove, and
    the caller records it as a degraded segment: losing the tail of one line is
    better than talking over the next speaker, but it is still a loss and is
    reported as one.
    """
    import numpy as np

    if wave is None or len(wave) <= target_samples or target_samples <= 0:
        return wave
    out = np.array(wave[:target_samples], copy=True)
    fade = min(int(0.005 * 24000), target_samples)
    if fade > 1:
        ramp = np.linspace(1.0, 0.0, fade, dtype=out.dtype)
        out[-fade:] *= ramp
    return out
