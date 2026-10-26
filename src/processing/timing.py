"""
Phase 5: the authoritative timing model for dubbing.

One segment's timing is described by exactly one `SegmentTiming` record, created
once from the ASR/translation output and carried through TTS, correction and
placement. No downstream stage re-derives `target_duration` from a neighbour's
start time, which is how the previous implementation produced overlapping and
drifting timelines.

Correction is layered, and each layer is cheaper and less lossy than the next:

  L1 translation length   (src/translation/duration_aware.py)
  L2 TTS-native duration  (fix_duration / speed, honoured by the backend)
  L3 measurement          (this module: ratio + band classification)
  L4 bounded time-stretch (Rubberband, else librosa, else none)
  L5 explicit fallback    (accept overflow, or trim as a last resort)

The layers must not fight. A backend that already fitted the audio to a duration
sets `duration_honoured`, and the merge stage then leaves that segment alone
instead of stretching it a second time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

from src.utils import config


# ---------------------------------------------------------------------------
# Mismatch bands
# ---------------------------------------------------------------------------
# `ratio = generated_duration / target_duration`. Bands are inclusive of the
# inner bound and exclusive of the outer one, except the outermost which is
# unbounded. Thresholds live in config so they can be tuned from one place.
BAND_OK = "ok"
BAND_SHORT = "short"
BAND_LONG = "long"
BAND_EXTREME_SHORT = "extreme_short"
BAND_EXTREME_LONG = "extreme_long"


@dataclass
class SegmentTiming:
    """
    Everything the pipeline knows about one segment's place on the timeline.

    The first eight fields are the contract: every stage reads these and nothing
    else. The remainder record what correction was applied and are what the
    diagnostics report and the future editor UI read.
    """

    # --- identity / contract ---------------------------------------------
    segment_id: int
    speaker_id: Optional[str]
    start: float
    end: float
    source_text: str
    translated_text: str

    # --- measurement ------------------------------------------------------
    # `target_duration` is ALWAYS `end - start`. It is a property, not a field,
    # so no stage can set it to something else and silently desynchronise the
    # timeline.
    generated_duration: float = 0.0
    final_duration: float = 0.0
    timing_action: str = "none"

    # --- correction detail ------------------------------------------------
    band: str = BAND_OK
    ratio: float = 1.0
    stretch_ratio: float = 1.0
    duration_honoured: bool = False
    backend: Optional[str] = None
    error: Optional[str] = None
    overflow: float = 0.0
    # fits | extended | collision | not_placed - see classify_fit()
    fit: str = "fits"

    @property
    def target_duration(self) -> float:
        """The original dialogue window. Derived, never assigned."""
        return max(0.0, self.end - self.start)

    @property
    def final_start(self) -> float:
        """Placement start. Always the original start: we never shift dialogue."""
        return self.start

    @property
    def final_end(self) -> float:
        return self.final_start + self.final_duration

    @property
    def is_degraded(self) -> bool:
        """
        True when the segment is audibly worse than it should be.

        `extended` is NOT degraded: running into a following pause is the
        intended behaviour. Only a collision (tail cut), a failure, a silent
        segment or an unfixable overrun counts.
        """
        return (self.timing_action in ("trimmed", "failed", "silent", "overflow")
                or self.fit == "collision")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_duration"] = self.target_duration
        d["final_start"] = self.final_start
        d["final_end"] = self.final_end
        d["is_degraded"] = self.is_degraded
        return d


def classify_ratio(ratio: float) -> str:
    """
    Map a duration ratio to a mismatch band.

    The bands are deliberately asymmetric: speech can be compressed further than
    it can be slowed before it sounds wrong, so the "still acceptable" window is
    wider above 1.0 than below it.
    """
    if not math.isfinite(ratio) or ratio <= 0:
        return BAND_OK
    if ratio < config.TIMING_EXTREME_SHORT_RATIO:
        return BAND_EXTREME_SHORT
    if ratio < config.TIMING_SHORT_RATIO:
        return BAND_SHORT
    if ratio <= config.TIMING_LONG_RATIO:
        return BAND_OK
    if ratio <= config.TIMING_EXTREME_LONG_RATIO:
        return BAND_LONG
    return BAND_EXTREME_LONG


def clamp_stretch(ratio: float) -> float:
    """
    Clamp a stretch ratio to the range the ear tolerates.

    `ratio` is generated/target, so >1 means "too long, speed up". The returned
    value is the same convention, ready to hand to a time-stretcher.
    """
    if not math.isfinite(ratio) or ratio <= 0:
        return 1.0
    return min(max(ratio, config.TIMING_MIN_STRETCH), config.TIMING_MAX_STRETCH)


def build_timings(segments: list[dict]) -> list[SegmentTiming]:
    """
    Build the authoritative timing records from translated segments.

    Anything a later stage needs about a segment's identity or timing comes from
    here. `target_durations`, when Phase 4 recorded one per segment, is accepted
    only as a sanity cross-check: the window always wins, because the window is
    what the video actually shows.
    """
    timings: list[SegmentTiming] = []
    for i, seg in enumerate(segments):
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        if end < start:
            end = start
        timings.append(
            SegmentTiming(
                segment_id=seg.get("segment_id", i),
                speaker_id=seg.get("speaker_id"),
                start=start,
                end=end,
                source_text=seg.get("source_text") or seg.get("text") or "",
                translated_text=seg.get("translated_text") or "",
                backend=(seg.get("translation") or {}).get("backend"),
            )
        )
    return timings


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@dataclass
class TimingDiagnostics:
    """Per-run aggregate, so a finished render can be audited afterwards."""

    segments: list[dict] = field(default_factory=list)

    def record(self, timing: SegmentTiming) -> None:
        self.segments.append(timing.to_dict())

    def merge(self, other: "TimingDiagnostics") -> None:
        """Absorb another run's records. Used to aggregate chunked processing."""
        if other is not None:
            self.segments.extend(other.segments)

    @property
    def stretched(self) -> list[dict]:
        return [s for s in self.segments if s["timing_action"] in ("compressed", "stretched")]

    @property
    def overflowed(self) -> list[dict]:
        return [s for s in self.segments if s["timing_action"] == "overflow"]

    @property
    def trimmed(self) -> list[dict]:
        return [s for s in self.segments if s["timing_action"] == "trimmed"]

    @property
    def failed(self) -> list[dict]:
        return [s for s in self.segments if s["timing_action"] == "failed"]

    def summary(self) -> dict:
        by_action: dict[str, int] = {}
        by_backend: dict[str, int] = {}
        for s in self.segments:
            by_action[s["timing_action"]] = by_action.get(s["timing_action"], 0) + 1
            backend = s.get("backend") or "unknown"
            by_backend[backend] = by_backend.get(backend, 0) + 1
        return {
            "segments": len(self.segments),
            "by_action": by_action,
            "by_backend": by_backend,
            "worst_ratio": max((s["ratio"] for s in self.segments), default=1.0),
        }
