"""
Phase 5 tests: the timing model, mismatch bands, and the correction decision.

These pin the invariants the old pipeline violated: a target duration is always
derived from the window, an overrun is corrected but an underrun is left as a
pause, and a segment gets at most one lossy transform.
"""

import pytest

from src.processing.timing import (
    BAND_EXTREME_LONG,
    BAND_EXTREME_SHORT,
    BAND_LONG,
    BAND_OK,
    BAND_SHORT,
    SegmentTiming,
    TimingDiagnostics,
    build_timings,
    clamp_stretch,
    classify_ratio,
)
from src.processing.timing_correction import (
    TimingCorrector,
    allowed_end,
    classify_fit,
)
from src.utils import config


def make_timing(start=0.0, end=1.0, text="hello", **kw):
    return SegmentTiming(
        segment_id=kw.pop("segment_id", 0),
        speaker_id=kw.pop("speaker_id", "S0"),
        start=start, end=end,
        source_text=kw.pop("source_text", "hola"),
        translated_text=text, **kw,
    )


class TestSegmentTimingContract:
    def test_target_duration_is_derived_from_window(self):
        t = make_timing(start=2.0, end=5.0)
        assert t.target_duration == 3.0

    def test_target_duration_cannot_be_assigned(self):
        """The window is the single source of truth; no stage may override it."""
        t = make_timing(start=2.0, end=5.0)
        with pytest.raises(AttributeError):
            t.target_duration = 99.0

    def test_placement_start_is_always_the_original_start(self):
        """Dialogue is never shifted, so intentional overlap survives."""
        t = make_timing(start=3.5, end=4.0)
        assert t.final_start == 3.5

    def test_final_end_uses_final_duration(self):
        t = make_timing(start=1.0, end=2.0)
        t.final_duration = 0.5
        assert t.final_end == 1.5

    def test_negative_window_is_clamped_not_raised(self):
        t = make_timing(start=5.0, end=4.0)
        assert t.target_duration == 0.0

    def test_extended_is_not_degraded(self):
        """Running into a following pause is intended, not a defect."""
        t = make_timing()
        t.timing_action = "compressed"
        t.fit = "extended"
        assert t.is_degraded is False

    def test_collision_is_degraded(self):
        t = make_timing()
        t.fit = "collision"
        assert t.is_degraded is True

    def test_failed_is_degraded(self):
        t = make_timing()
        t.timing_action = "failed"
        assert t.is_degraded is True

    def test_to_dict_exposes_derived_fields(self):
        t = make_timing(start=1.0, end=3.0)
        t.final_duration = 1.5
        d = t.to_dict()
        assert d["target_duration"] == 2.0
        assert d["final_start"] == 1.0
        assert d["final_end"] == 2.5
        assert "is_degraded" in d


class TestBands:
    def test_ratio_one_is_ok(self):
        assert classify_ratio(1.0) == BAND_OK

    def test_short_and_long(self):
        assert classify_ratio(0.6) == BAND_SHORT
        assert classify_ratio(1.5) == BAND_LONG

    def test_extremes(self):
        assert classify_ratio(0.3) == BAND_EXTREME_SHORT
        assert classify_ratio(2.5) == BAND_EXTREME_LONG

    def test_degenerate_ratios_do_not_raise(self):
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            assert classify_ratio(bad) == BAND_OK

    def test_band_boundaries_follow_config(self):
        assert classify_ratio(config.TIMING_SHORT_RATIO) == BAND_OK
        assert classify_ratio(config.TIMING_LONG_RATIO) == BAND_OK
        assert classify_ratio(config.TIMING_SHORT_RATIO - 0.01) == BAND_SHORT
        assert classify_ratio(config.TIMING_LONG_RATIO + 0.01) == BAND_LONG


class TestClampStretch:
    def test_within_band_is_unchanged(self):
        assert clamp_stretch(1.1) == pytest.approx(1.1)

    def test_above_max_is_clamped(self):
        assert clamp_stretch(3.0) == config.TIMING_MAX_STRETCH

    def test_below_min_is_clamped(self):
        assert clamp_stretch(0.2) == config.TIMING_MIN_STRETCH

    def test_degenerate_is_neutral(self):
        assert clamp_stretch(0.0) == 1.0
        assert clamp_stretch(float("nan")) == 1.0


class TestBuildTimings:
    def test_builds_one_record_per_segment(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "a", "translated_text": "b"},
            {"start": 1.5, "end": 3.0, "text": "c", "translated_text": "d"},
        ]
        timings = build_timings(segs)
        assert [t.target_duration for t in timings] == [1.0, 1.5]

    def test_source_text_prefers_explicit_field(self):
        segs = [{"start": 0, "end": 1, "text": "orig", "source_text": "explicit",
                 "translated_text": "t"}]
        assert build_timings(segs)[0].source_text == "explicit"

    def test_missing_end_defaults_to_start(self):
        segs = [{"start": 2.0, "translated_text": "t"}]
        t = build_timings(segs)[0]
        assert t.end == 2.0
        assert t.target_duration == 0.0

    def test_backend_carried_from_translation_metadata(self):
        segs = [{"start": 0, "end": 1, "translated_text": "t",
                 "translation": {"backend": "nllb"}}]
        assert build_timings(segs)[0].backend == "nllb"


class TestDiagnostics:
    def test_summary_counts_actions_and_backends(self):
        diag = TimingDiagnostics()
        a = make_timing(); a.timing_action = "compressed"; a.backend = "nllb"
        b = make_timing(segment_id=1); b.timing_action = "none"; b.backend = "nllb"
        c = make_timing(segment_id=2); c.timing_action = "failed"; c.error = "x"
        for t in (a, b, c):
            diag.record(t)
        s = diag.summary()
        assert s["segments"] == 3
        assert s["by_action"]["compressed"] == 1
        assert s["by_backend"]["nllb"] == 2
        assert s["by_backend"]["unknown"] == 1

    def test_degraded_buckets(self):
        diag = TimingDiagnostics()
        over = make_timing(); over.timing_action = "overflow"
        cut = make_timing(segment_id=1); cut.timing_action = "trimmed"
        bad = make_timing(segment_id=2); bad.timing_action = "failed"
        ok = make_timing(segment_id=3); ok.timing_action = "none"; ok.fit = "extended"
        for t in (over, cut, bad, ok):
            diag.record(t)
        assert len(diag.overflowed) == 1
        assert len(diag.trimmed) == 1
        assert len(diag.failed) == 1
        assert len(diag.stretched) == 0

    def test_worst_ratio(self):
        diag = TimingDiagnostics()
        for r in (1.0, 1.8, 0.9):
            t = make_timing(); t.ratio = r
            diag.record(t)
        assert diag.summary()["worst_ratio"] == 1.8


class TestAllowedEnd:
    def test_sequential_uses_next_start(self):
        cur = make_timing(start=0.0, end=1.0)
        nxt = make_timing(start=1.2, end=2.0, segment_id=1)
        assert allowed_end(cur, nxt, 5.0) == 1.2

    def test_intentional_overlap_is_preserved(self):
        """Next speaker started before this one ended: keep the overlap."""
        cur = make_timing(start=0.0, end=2.0)
        nxt = make_timing(start=1.0, end=3.0, segment_id=1)
        assert allowed_end(cur, nxt, 5.0) == 2.0

    def test_last_segment_may_use_the_tail(self):
        cur = make_timing(start=0.0, end=1.0)
        assert allowed_end(cur, None, 5.0) == 5.0


class TestClassifyFit:
    def test_within_window(self):
        assert classify_fit(1.0, 0.9, 1.5) == "fits"

    def test_into_pause_is_extended(self):
        assert classify_fit(1.0, 1.2, 1.5) == "extended"

    def test_past_the_limit_is_collision(self):
        assert classify_fit(1.0, 1.6, 1.5) == "collision"


class TestCorrector:
    def test_ok_ratio_needs_no_change(self):
        t = make_timing()
        action, stretch, dur = TimingCorrector().correct(t, 1.05)
        assert action == "none" and stretch == 1.0

    def test_native_fit_is_never_restretched(self):
        """A backend that already fitted the audio must not be stretched again."""
        t = make_timing()
        t.duration_honoured = True
        action, stretch, dur = TimingCorrector().correct(t, 1.9)
        assert action == "native_fit"
        assert stretch == 1.0
        assert dur == 1.9

    def test_overrun_is_compressed_within_bounds(self):
        t = make_timing()
        action, stretch, dur = TimingCorrector().correct(t, 1.5)
        assert action == "compressed"
        assert stretch <= config.TIMING_MAX_STRETCH
        assert dur < 1.5

    def test_extreme_overrun_still_respects_the_clamp(self):
        t = make_timing()
        action, stretch, dur = TimingCorrector().correct(t, 5.0)
        assert action == "compressed"
        assert stretch == config.TIMING_MAX_STRETCH

    def test_underrun_is_left_alone(self):
        """Never stretch to fill a window: a pause is the natural result."""
        t = make_timing()
        action, stretch, dur = TimingCorrector().correct(t, 0.3)
        assert action == "none"
        assert stretch == 1.0
        assert dur == 0.3

    def test_stretch_disabled_reports_uncorrected(self):
        t = make_timing()
        action, stretch, dur = TimingCorrector(enable_stretch=False).correct(t, 3.0)
        assert action == "uncorrected"
        assert stretch == 1.0

    def test_zero_window_and_empty_audio(self):
        t = make_timing(start=1.0, end=1.0)
        assert TimingCorrector().correct(t, 1.0)[0] == "zero_window"
        t2 = make_timing()
        assert TimingCorrector().correct(t2, 0.0)[0] == "failed"
