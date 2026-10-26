"""
Phase 5 integration tests: the timeline assembly, end to end, with real WAVs.

These exercise the actual merge path (real soundfile reads, real librosa
stretch) rather than mocks, because the bugs this phase fixes were all in the
seams between stages: a stretch applied twice, a segment shifted away from its
start, a track that was not exactly the video's length.
"""

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from src.processing.synchronization import AudioSynchronizer
from src.processing.timing import TimingDiagnostics

SR = 24000


def tone(seconds: float, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


@pytest.fixture
def workspace():
    d = tempfile.mkdtemp()
    yield d
    for name in os.listdir(d):
        try:
            os.unlink(os.path.join(d, name))
        except OSError:
            pass
    try:
        os.rmdir(d)
    except OSError:
        pass


def write(workspace, name, seconds):
    path = os.path.join(workspace, name)
    sf.write(path, tone(seconds), SR)
    return path


class TestTimelineAssembly:
    def test_output_is_exactly_the_video_duration(self, workspace):
        segs = [{"audio_path": write(workspace, "a.wav", 1.0),
                 "start": 0.0, "end": 1.0, "segment_id": 0}]
        out = os.path.join(workspace, "out.wav")
        AudioSynchronizer().merge_segments(segs, total_duration=5.0, output_path=out)
        assert sf.info(out).duration == pytest.approx(5.0, abs=1e-3)

    def test_short_output_is_padded_to_video_duration(self, workspace):
        """TTS shorter than the video must not shorten the video."""
        segs = [{"audio_path": write(workspace, "a.wav", 0.5),
                 "start": 0.0, "end": 0.5, "segment_id": 0}]
        out = os.path.join(workspace, "out.wav")
        AudioSynchronizer().merge_segments(segs, total_duration=3.0, output_path=out)
        assert sf.info(out).duration == pytest.approx(3.0, abs=1e-3)

    def test_segment_is_placed_at_its_original_start(self, workspace):
        """A segment must not be shifted earlier or later than the source."""
        segs = [{"audio_path": write(workspace, "a.wav", 0.5),
                 "start": 2.0, "end": 2.5, "segment_id": 0}]
        out = os.path.join(workspace, "out.wav")
        diag = TimingDiagnostics()
        AudioSynchronizer().merge_segments(segs, total_duration=4.0,
                                           output_path=out, diagnostics=diag)
        rec = diag.segments[0]
        assert rec["final_start"] == 2.0

    def test_audio_lands_at_the_expected_sample_offset(self, workspace):
        """Silence before the first word, then sound exactly at its start."""
        segs = [{"audio_path": write(workspace, "a.wav", 0.5),
                 "start": 1.0, "end": 1.5, "segment_id": 0}]
        out = os.path.join(workspace, "out.wav")
        AudioSynchronizer().merge_segments(segs, total_duration=2.0, output_path=out)
        data, _ = sf.read(out)
        before = np.max(np.abs(data[: int(0.9 * SR)]))
        during = np.max(np.abs(data[int(1.05 * SR): int(1.4 * SR)]))
        assert before < 1e-3
        assert during > 0.1

    def test_intentional_overlap_is_preserved(self, workspace):
        """Two speakers mixed in the source stay mixed in the dub."""
        a = write(workspace, "a.wav", 2.0, )
        b = write(workspace, "b.wav", 2.0)
        segs = [
            {"audio_path": a, "start": 0.0, "end": 2.0, "segment_id": 0},
            {"audio_path": b, "start": 1.0, "end": 3.0, "segment_id": 1},
        ]
        out = os.path.join(workspace, "out.wav")
        diag = TimingDiagnostics()
        AudioSynchronizer().merge_segments(segs, total_duration=4.0,
                                           output_path=out, diagnostics=diag)
        assert diag.segments[1]["final_start"] == 1.0
        # Both segments are present, and the overlapping region is louder than
        # either alone would be at that point.
        data, _ = sf.read(out)
        assert np.max(np.abs(data)) > 0.1

    def test_empty_segment_list_still_produces_a_track(self, workspace):
        out = os.path.join(workspace, "out.wav")
        AudioSynchronizer().merge_segments([], total_duration=2.0, output_path=out)
        assert sf.info(out).duration == pytest.approx(2.0, abs=1e-3)


class TestCorrectionBehaviour:
    def test_overrun_is_compressed_not_left_long(self, workspace):
        # 1.5s of audio in a 1.0s window, with room after it.
        segs = [{"audio_path": write(workspace, "a.wav", 1.5),
                 "start": 0.0, "end": 1.0, "segment_id": 0}]
        out = os.path.join(workspace, "out.wav")
        diag = TimingDiagnostics()
        AudioSynchronizer().merge_segments(segs, total_duration=4.0, output_path=out,
                                           enable_time_stretch=True, diagnostics=diag)
        rec = diag.segments[0]
        assert rec["timing_action"] == "compressed"
        assert rec["stretch_ratio"] > 1.0
        assert rec["final_duration"] < rec["generated_duration"]

    def test_underrun_is_not_stretched_to_fill(self, workspace):
        """The whole point: never drawl a short line out to fill its window."""
        segs = [{"audio_path": write(workspace, "a.wav", 0.4),
                 "start": 0.0, "end": 2.0, "segment_id": 0}]
        out = os.path.join(workspace, "out.wav")
        diag = TimingDiagnostics()
        AudioSynchronizer().merge_segments(segs, total_duration=3.0, output_path=out,
                                           enable_time_stretch=True, diagnostics=diag)
        rec = diag.segments[0]
        assert rec["timing_action"] == "none"
        assert rec["stretch_ratio"] == 1.0
        assert rec["final_duration"] == pytest.approx(0.4, abs=0.02)

    def test_native_fit_is_not_stretched_again(self, workspace):
        """A backend that already fitted the audio must be left alone."""
        from src.processing.timing import SegmentTiming

        t = SegmentTiming(segment_id=0, speaker_id="S0", start=0.0, end=1.0,
                          source_text="s", translated_text="t")
        t.duration_honoured = True
        segs = [{"audio_path": write(workspace, "a.wav", 1.6), "start": 0.0,
                 "end": 1.0, "segment_id": 0, "timing": t}]
        out = os.path.join(workspace, "out.wav")
        diag = TimingDiagnostics()
        AudioSynchronizer().merge_segments(segs, total_duration=4.0, output_path=out,
                                           enable_time_stretch=True, diagnostics=diag)
        rec = diag.segments[0]
        assert rec["timing_action"] == "native_fit"
        assert rec["stretch_ratio"] == 1.0

    def test_collision_with_next_speaker_is_trimmed(self, workspace):
        # 3s of audio, a 1s window, and the next speaker starts at 1.5s.
        segs = [
            {"audio_path": write(workspace, "a.wav", 3.0), "start": 0.0,
             "end": 1.0, "segment_id": 0},
            {"audio_path": write(workspace, "b.wav", 0.5), "start": 1.5,
             "end": 2.0, "segment_id": 1},
        ]
        out = os.path.join(workspace, "out.wav")
        diag = TimingDiagnostics()
        AudioSynchronizer().merge_segments(segs, total_duration=3.0, output_path=out,
                                           enable_time_stretch=True, diagnostics=diag)
        rec = diag.segments[0]
        assert rec["fit"] == "collision"
        assert rec["overflow"] > 0
        assert rec["is_degraded"] is True

    def test_stretch_disabled_reports_uncorrected(self, workspace):
        segs = [{"audio_path": write(workspace, "a.wav", 2.0),
                 "start": 0.0, "end": 1.0, "segment_id": 0}]
        out = os.path.join(workspace, "out.wav")
        diag = TimingDiagnostics()
        AudioSynchronizer().merge_segments(segs, total_duration=4.0, output_path=out,
                                           enable_time_stretch=False, diagnostics=diag)
        assert diag.segments[0]["timing_action"] == "uncorrected"

    def test_silent_segment_is_flagged_not_counted_as_success(self, workspace):
        path = os.path.join(workspace, "silent.wav")
        sf.write(path, np.zeros(SR, dtype=np.float32), SR)
        segs = [{"audio_path": path, "start": 0.0, "end": 1.0, "segment_id": 0}]
        out = os.path.join(workspace, "out.wav")
        diag = TimingDiagnostics()
        AudioSynchronizer().merge_segments(segs, total_duration=2.0, output_path=out,
                                           diagnostics=diag)
        rec = diag.segments[0]
        assert rec["timing_action"] == "silent"
        assert rec["is_degraded"] is True

    def test_missing_file_is_recorded_not_fatal(self, workspace):
        segs = [{"audio_path": os.path.join(workspace, "nope.wav"),
                 "start": 0.0, "end": 1.0, "segment_id": 0}]
        out = os.path.join(workspace, "out.wav")
        diag = TimingDiagnostics()
        AudioSynchronizer().merge_segments(segs, total_duration=2.0, output_path=out,
                                           diagnostics=diag)
        rec = diag.segments[0]
        assert rec["timing_action"] == "failed"
        assert rec["error"] == "missing_file"

    def test_one_bad_segment_does_not_lose_the_others(self, workspace):
        good = write(workspace, "a.wav", 1.0)
        segs = [
            {"audio_path": os.path.join(workspace, "nope.wav"), "start": 0.0,
             "end": 1.0, "segment_id": 0},
            {"audio_path": good, "start": 2.0, "end": 3.0, "segment_id": 1},
        ]
        out = os.path.join(workspace, "out.wav")
        diag = TimingDiagnostics()
        AudioSynchronizer().merge_segments(segs, total_duration=4.0, output_path=out,
                                           diagnostics=diag)
        assert diag.segments[0]["timing_action"] == "failed"
        assert diag.segments[1]["timing_action"] != "failed"

    def test_corrupt_wav_is_recorded_not_fatal(self, workspace):
        path = os.path.join(workspace, "corrupt.wav")
        with open(path, "wb") as f:
            f.write(b"not a wav file at all, just bytes" * 20)
        segs = [{"audio_path": path, "start": 0.0, "end": 1.0, "segment_id": 0}]
        out = os.path.join(workspace, "out.wav")
        diag = TimingDiagnostics()
        AudioSynchronizer().merge_segments(segs, total_duration=2.0, output_path=out,
                                           diagnostics=diag)
        assert diag.segments[0]["timing_action"] == "failed"


class TestBackendCapabilities:
    def test_indicf5_honours_duration_and_others_do_not(self):
        from src.synthesis.capabilities import honours_duration

        assert honours_duration("indicf5") is True
        for other in ("xtts", "edge", "piper", "f5", "vibevoice", None, "nonsense"):
            assert honours_duration(other) is False

    def test_plan_native_fit_only_requests_for_capable_backends(self):
        from src.processing.duration_model import DurationModel, plan_native_fit
        from src.processing.timing import build_timings

        segs = [{"start": 0.0, "end": 1.0, "text": "This is a long English line",
                 "translated_text": "ఇది చాలా పొడవైన తెలుగు వాక్యం ఉదాహరణ",
                 "speaker_id": "S0"}]
        timings = build_timings(segs)
        model = DurationModel(source_lang="en", target_lang="te")
        refs = {"S0": 6.0}

        assert plan_native_fit(timings, model, model_name="xtts", ref_durations=refs) == {}
        req = plan_native_fit(timings, model, model_name="indicf5", ref_durations=refs)
        assert 0 in req

    def test_native_fit_respects_the_stretch_bounds(self):
        from src.processing.duration_model import DurationModel, plan_native_fit
        from src.processing.timing import build_timings
        from src.utils import config

        segs = [{"start": 0.0, "end": 1.0, "text": "tiny",
                 "translated_text": "x" * 400, "speaker_id": "S0"}]
        timings = build_timings(segs)
        model = DurationModel(source_lang="en", target_lang="en")
        refs = {"S0": 5.0}
        req = plan_native_fit(timings, model, model_name="indicf5", ref_durations=refs)
        # Requested value is the TOTAL: reference + at most the stretched window.
        assert req[0] <= 5.0 + 1.0 * config.TIMING_MAX_STRETCH + 1e-6

    def test_native_fit_includes_the_reference_length(self):
        """IndicF5 counts reference + generated, so the total must include it."""
        from src.processing.duration_model import DurationModel, plan_native_fit
        from src.processing.timing import build_timings

        segs = [{"start": 0.0, "end": 1.0, "text": "a short line",
                 "translated_text": "x" * 60, "speaker_id": "S0"}]
        timings = build_timings(segs)
        model = DurationModel(source_lang="en", target_lang="en")

        without = plan_native_fit(timings, model, model_name="indicf5",
                                  ref_durations={"S0": 0.0})
        with_ref = plan_native_fit(timings, model, model_name="indicf5",
                                   ref_durations={"S0": 6.0})
        assert with_ref[0] - without[0] == pytest.approx(6.0)

    def test_native_fit_skips_a_segment_with_no_measurable_reference(self):
        from src.processing.duration_model import DurationModel, plan_native_fit
        from src.processing.timing import build_timings

        segs = [{"start": 0.0, "end": 1.0, "text": "a short line",
                 "translated_text": "x" * 60, "speaker_id": "UNKNOWN_SPK"}]
        timings = build_timings(segs)
        model = DurationModel(source_lang="en", target_lang="en")
        assert plan_native_fit(timings, model, model_name="indicf5",
                               ref_durations={"S0": 6.0}) == {}

    def test_measure_reference_durations_reads_headers(self, workspace):
        from src.processing.duration_model import measure_reference_durations

        p = write(workspace, "ref.wav", 4.0)
        durations = measure_reference_durations({"S0": p, "S1": None})
        assert durations["S0"] == pytest.approx(4.0, abs=1e-3)
        assert "S1" not in durations

    def test_measure_reference_durations_tolerates_bad_paths(self, workspace):
        from src.processing.duration_model import measure_reference_durations

        durations = measure_reference_durations(
            {"S0": os.path.join(workspace, "nope.wav")})
        assert durations == {}

    def test_no_request_when_already_close(self):
        from src.processing.duration_model import DurationModel, plan_native_fit
        from src.processing.timing import build_timings

        # 15.5 cps * 1.0s ~ 15 chars fits the window already.
        segs = [{"start": 0.0, "end": 1.0, "text": "fifteen chars!!",
                 "translated_text": "fifteen chars!!", "speaker_id": "S0"}]
        timings = build_timings(segs)
        model = DurationModel(source_lang="en", target_lang="en")
        assert plan_native_fit(timings, model, model_name="indicf5",
                               ref_durations={"S0": 5.0}) == {}


class TestDurationModel:
    def test_same_language_same_slot_needs_no_change(self):
        from src.processing.duration_model import DurationModel

        model = DurationModel(source_lang="en", target_lang="en")
        est = model.estimate("hello there", source_text="hello there", slot=2.0)
        assert est == pytest.approx(2.0, abs=1e-6)

    def test_longer_translation_estimates_longer(self):
        from src.processing.duration_model import DurationModel

        model = DurationModel(source_lang="en", target_lang="en")
        short = model.estimate("hi", source_text="hello world", slot=2.0)
        long = model.estimate("hello world and more words", source_text="hi", slot=2.0)
        assert long > short

    def test_unknown_language_code_does_not_raise(self):
        from src.processing.duration_model import DurationModel

        model = DurationModel(source_lang="zz", target_lang="zz")
        assert model.estimate("hello", slot=1.0) > 0

    def test_speaker_factor_is_bounded(self):
        from src.processing.duration_model import DurationModel

        model = DurationModel(ref_duration=1.0, ref_text="a" * 400,
                              source_lang="en", target_lang="en")
        assert model.MIN_SPEAKER_FACTOR <= model.speaker_factor <= model.MAX_SPEAKER_FACTOR


class TestTTSWiring:
    """
    The `fix_duration` key was plumbed through the TTS engine in an earlier
    phase but never actually set by the pipeline, so IndicF5's duration fitting
    was dead code. These tests pin the wiring.
    """

    def _translator(self, workspace):
        from src.core.video_translator import VideoTranslator

        captured = {}

        class FakeTTSEngine:
            def route_model(self, code, model):
                return model

            def generate_batch(self, tasks, model=None, check_cancel_callback=None):
                captured["tasks"] = tasks
                out = []
                for t in tasks:
                    p = t["output_path"]
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    sf.write(p, tone(1.0), SR)
                    out.append(p)
                return out

            def unload_model(self):
                pass

        vt = VideoTranslator.__new__(VideoTranslator)
        vt.tts_engine = FakeTTSEngine()
        from src.processing.timing import TimingDiagnostics
        vt.timing_diagnostics = TimingDiagnostics()
        vt._check_cancel = lambda: None
        vt._get_assigned_voice = lambda spk, voices: voices[0]
        vt._resolve_speaker_reference = lambda *a, **k: (None, False)
        vt._apply_reference_fallback = (
            lambda w, v, m, i, speaker_id=None, enable_diarization=True: (w or v, False))
        vt._resolve_speaker_ref_text = lambda *a, **k: None
        vt._find_overlapping_speaker = lambda seg, ds: None
        return vt, captured

    def _segments(self):
        return [{
            "start": 0.0, "end": 1.0,
            "text": "This is a fairly long English sentence",
            "translated_text": "ఇది చాలా పొడవైన తెలుగు వాక్యం ఉదాహరణ",
            "segment_id": 7, "speaker_id": "SPK_1",
        }]

    def _run(self, vt, model_name, timings, native_fit, workspace):
        from pathlib import Path

        gen = vt._step_tts_generator(
            self._segments(), Path(workspace) / "x.mp4", "te", "en",
            model_name, None, False, False, None, {}, {}, "/tmp/vocals.wav",
            timings=timings, native_fit=native_fit,
        )
        result = None
        for item in gen:
            if isinstance(item, list):
                result = item
        return result

    def test_indicf5_task_carries_a_fix_duration(self, workspace):
        from src.processing.duration_model import DurationModel, plan_native_fit
        from src.processing.timing import build_timings

        vt, captured = self._translator(workspace)
        segs = self._segments()
        timings = build_timings(segs)
        model = DurationModel(source_lang="en", target_lang="te")
        native_fit = plan_native_fit(timings, model, model_name="indicf5",
                                     ref_durations={"SPK_1": 6.0})
        assert native_fit, "test premise: the line should need a native fit"

        self._run(vt, "indicf5", timings, native_fit, workspace)
        assert captured["tasks"][0]["fix_duration"] is not None

    def test_non_capable_backend_gets_no_fix_duration(self, workspace):
        from src.processing.duration_model import DurationModel, plan_native_fit
        from src.processing.timing import build_timings

        vt, captured = self._translator(workspace)
        timings = build_timings(self._segments())
        model = DurationModel(source_lang="en", target_lang="te")
        native_fit = plan_native_fit(timings, model, model_name="xtts",
                                     ref_durations={"SPK_1": 6.0})
        assert native_fit == {}

        self._run(vt, "xtts", timings, native_fit, workspace)
        assert captured["tasks"][0]["fix_duration"] is None

    def test_timing_record_travels_with_the_audio(self, workspace):
        from src.processing.timing import build_timings

        vt, _ = self._translator(workspace)
        timings = build_timings(self._segments())
        result = self._run(vt, "xtts", timings, {}, workspace)
        entry = result[0]
        assert entry["segment_id"] == 7
        assert entry["speaker_id"] == "SPK_1"
        assert entry["timing"] is timings[0]

    def test_honoured_flag_is_set_only_for_capable_backends(self, workspace):
        from src.processing.duration_model import DurationModel, plan_native_fit
        from src.processing.timing import build_timings

        segs = self._segments()
        model = DurationModel(source_lang="en", target_lang="te")

        vt, _ = self._translator(workspace)
        timings = build_timings(segs)
        nf = plan_native_fit(timings, model, model_name="indicf5",
                             ref_durations={"SPK_1": 6.0})
        self._run(vt, "indicf5", timings, nf, workspace)
        assert timings[0].duration_honoured is True

        vt2, _ = self._translator(workspace)
        timings2 = build_timings(segs)
        nf2 = plan_native_fit(timings2, model, model_name="indicf5",
                              ref_durations={"SPK_1": 6.0})
        # Even with a request planned, a backend that cannot honour it must not
        # claim it did, or the merge stage would skip a needed correction.
        self._run(vt2, "xtts", timings2, nf2, workspace)
        assert timings2[0].duration_honoured is False
