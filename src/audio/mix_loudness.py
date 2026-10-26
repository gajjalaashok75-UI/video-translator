"""
Dialogue-vs-background loudness handling for the final mix.

The rule this module exists to enforce: a natural mix must keep the dialogue
clearly on top of the music and effects without flattening the movie to one
loudness. Two things make that hard in practice and both are handled here.

*Generated dialogue arrives with no common reference level.* A backend emits
whatever level it likes per segment, so Speaker A can come back loud and
Speaker B quiet and the alternation is audible as a defect. The correction is
a bounded gain toward a target, never a full normalisation: a whisper stays
quieter than a shout.
"""

from __future__ import annotations

import numpy as np

# Per-segment dialogue loudness is pulled toward the target but moved by at
# most this much either way, so intentional dynamics survive and only gross
# inconsistency is removed.
DEFAULT_MAX_DIALOGUE_ADJUST_DB = 6.0

# Speech should sit this far above the background so it stays intelligible
# without burying the music. Measured on the real sample the gap was ~32 dB,
# which is the failure this default fixes.
DEFAULT_DIALOGUE_OVER_BG_DB = 18.0

# Ceiling for the mix. The limiter keeps peaks under this rather than dividing
# the whole track by its peak, which would pump the music.
DEFAULT_CEILING = 0.95

# Short equal-power fades at a segment boundary. This is enough to stop a click
# on a cut; it is deliberately not long enough to blur rapid dialogue.
FADE_IN_MS = 8.0
FADE_OUT_MS = 10.0


def rms_dbfs(wave: np.ndarray) -> float:
    """RMS level of `wave` in dBFS. Silence returns a large negative number."""
    if wave is None or len(wave) == 0:
        return -200.0
    r = float(np.sqrt(np.mean(np.square(np.asarray(wave, dtype=np.float64)))))
    return 20.0 * np.log10(r + 1e-12)


def active_dbfs(wave: np.ndarray, frame_ms: float = 20.0,
                sample_rate: int = 24000) -> float:
    """
    RMS over the loudest frames, ignoring the leading/trailing silence a
    backend usually leaves.

    A whole-segment RMS under-reports a line with a long silent tail, which
    would then be boosted too far. This estimates the level of the speech
    itself. Returns the overall RMS when the signal is too short to frame.
    """
    x = np.asarray(wave, dtype=np.float64)
    if x.size == 0:
        return -200.0
    hop = max(1, int(sample_rate * frame_ms / 1000.0))
    n = x.size // hop
    if n < 2:
        return rms_dbfs(x)
    frames = x[: n * hop].reshape(n, hop)
    levels = 20.0 * np.log10(np.sqrt(np.mean(np.square(frames), axis=1)) + 1e-12)
    # The 90th percentile is a robust "speech level": above the quiet frames,
    # below a single click.
    return float(np.percentile(levels, 90))


def bounded_gain_db(level_db: float, target_db: float,
                    max_adjust_db: float = DEFAULT_MAX_DIALOGUE_ADJUST_DB) -> float:
    """Gain (dB) that moves `level_db` toward `target_db`, clamped to ±max."""
    if level_db <= -199.0:
        return 0.0
    adjust = target_db - level_db
    return float(max(-max_adjust_db, min(max_adjust_db, adjust)))


def apply_gain(wave: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply a dB gain to `wave`, returning float64."""
    x = np.asarray(wave, dtype=np.float64)
    if abs(gain_db) < 1e-6:
        return x
    return x * (10.0 ** (gain_db / 20.0))


def match_dialogue_levels(levels_db, target_db: float,
                          max_adjust_db: float = DEFAULT_MAX_DIALOGUE_ADJUST_DB):
    """
    Bounded gains that bring a group of segments to a shared dialogue level.

    `levels_db` is one measured level per segment. Returns one gain (dB) per
    segment. Segments already at the target are left alone; the clamp is what
    keeps loud and quiet lines different instead of flattening them.
    """
    return [bounded_gain_db(l, target_db, max_adjust_db) for l in levels_db]


def equal_power_fades(n_samples: int, sample_rate: int,
                      fade_in_ms: float = FADE_IN_MS,
                      fade_out_ms: float = FADE_OUT_MS) -> np.ndarray:
    """
    An equal-power fade envelope for one segment (`n_samples` long).

    Equal-power (sin/cos) rather than linear, so two segments that meet keep a
    constant perceived level across the join instead of dipping.
    """
    env = np.ones(max(0, int(n_samples)), dtype=np.float64)
    if env.size == 0:
        return env
    fi = min(int(sample_rate * fade_in_ms / 1000.0), env.size)
    fo = min(int(sample_rate * fade_out_ms / 1000.0), env.size)
    if fi > 0:
        env[:fi] *= np.sin(np.linspace(0.0, np.pi / 2.0, fi)) ** 2
    if fo > 0:
        env[-fo:] *= np.cos(np.linspace(0.0, np.pi / 2.0, fo)) ** 2
    return env


def limit_peaks(wave: np.ndarray, ceiling: float = DEFAULT_CEILING) -> np.ndarray:
    """
    Scale `wave` down so its peak reaches `ceiling`, and only if it exceeds it.

    A global scale by the peak is exactly what makes music pump when dialogue
    swells; but for a hard ceiling a single scale is the correct, artefact-free
    choice, and it is applied only when the mix would otherwise clip. A mix
    already under the ceiling is returned untouched.
    """
    x = np.asarray(wave, dtype=np.float64)
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    if peak <= ceiling or peak <= 0.0:
        return x
    return x * (ceiling / peak)


def estimate_mix_gains(dialogue: np.ndarray, background: np.ndarray,
                       dialogue_over_bg_db: float = DEFAULT_DIALOGUE_OVER_BG_DB,
                       max_bg_reduction_db: float = 24.0):
    """
    Return `(dialogue_gain_db, bg_gain_db)` so dialogue sits above background.

    The dialogue level is the reference and the background is turned down to
    sit `dialogue_over_bg_db` below it. The background is never boosted: if it
    is already at or below the target it is left exactly as it is, because
    raising a quiet music/effects bed mostly raises its noise. It is also never
    pushed more than `max_bg_reduction_db` down, so a loud bed is not silenced
    outright.
    """
    d = active_dbfs(dialogue)
    b = active_dbfs(background)
    if b <= -199.0 or d <= -199.0:
        return 0.0, 0.0
    desired_bg = d - dialogue_over_bg_db
    # Only ever pull the background down, never push it up.
    bg_gain = min(0.0, desired_bg - b)
    bg_gain = max(bg_gain, -abs(max_bg_reduction_db))
    return 0.0, float(bg_gain)
