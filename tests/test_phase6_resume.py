"""
Phase 6: resume behaviour of the pipeline itself.

The unit tests in `test_phase6_jobs.py` prove the pieces work. These prove the
promise: run a stage, stop, run it again, and the expensive work is not repeated
while the output is still correct.

The TTS stage is the one under test here. It is the most expensive per-segment
work in the pipeline, it is the stage a long render spends most of its wall
clock in, and it is where a wrong resume is most damaging (a re-used voice from
a previous run's speaker, or a hole where a segment should be).
"""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from src.core import jobs
from src.core.cache import ArtifactCache
from src.core.manifest import (
    Manifest, STAGE_COMPLETE, STAGE_FAILED, STAGE_PENDING,
)
from src.core.video_translator import VideoTranslator
from src.utils import config

RATE = 16000


def _tone(seconds=0.5, freq=220.0):
    t = np.linspace(0, seconds, int(RATE * seconds), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class CountingTTS:
    """
    Stands in for TTSEngine, counting how many lines it was actually asked to
    synthesize. That count is the measurement: a resume that re-synthesizes
    everything has not resumed anything.
    """

    def __init__(self, fail_indices=()):
        self.batches = []
        self.fail_indices = set(fail_indices)

    def route_model(self, target_lang_code, requested_model=None):
        return requested_model or "edge"

    def validate_reference(self, wav_path, model_name=None):
        # The pipeline validates a reference before conditioning on it. A real
        # file that exists is good enough for this stand-in.
        return bool(wav_path) and __import__("pathlib").Path(str(wav_path)).exists()

    def get_available_voices(self, model, language_code):
        return ["Auto"]

    def generate_batch(self, tasks, model="edge", check_cancel_callback=None):
        self.batches.append([t["text"] for t in tasks])
        results = []
        for i, t in enumerate(tasks):
            path = t["output_path"]
            if i in self.fail_indices:
                results.append(None)  # a segment the backend could not produce
                continue
            sf.write(str(path), _tone(), RATE)
            results.append(str(path))
        return results

    def unload_all(self):
        pass

    @property

    def synthesized(self):
        return sum(len(b) for b in self.batches)


def _segments(n=4, speaker="SPEAKER_00"):
    return [
        {
            "start": float(i * 2),
            "end": float(i * 2 + 1.5),
            "text": f"line {i}",
            "translated_text": f"translated line {i}",
            "segment_id": i,
            "speaker_id": speaker,
        }
        for i in range(n)
    ]


def _translator(tmp_path, monkeypatch, tts=None):
    """A translator with a real job, real cache, real manifest, fake TTS."""
    jobs_dir = tmp_path / "jobs"
    monkeypatch.setattr(config, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path / "temp")
    (tmp_path / "temp").mkdir(parents=True, exist_ok=True)

    vt = VideoTranslator.__new__(VideoTranslator)
    from src.core.session import SessionContext
    vt.session = SessionContext()
    vt.cancel_requested = False
    vt.tts_engine = tts or CountingTTS()
    vt.device = "cpu"
    vt.timing_diagnostics = None

    # A real vocals file: the fallback chain reads it when no speaker profile
    # matches, so the stage cannot run without one. It is created once and left
    # alone, because the reference identity is derived from its mtime and
    # rewriting it would (correctly) invalidate every cached segment.
    vocals = tmp_path / "vocals.wav"
    if not vocals.exists():
        sf.write(str(vocals), _tone(2.0), RATE)
    vt._test_vocals = str(vocals)

    vt.job = jobs.Job.create(source_path=str(tmp_path / "movie.mp4"))
    vt.manifest = Manifest.create(
        vt.job, source_path=str(tmp_path / "movie.mp4"),
        source_lang="en", target_lang="te",
        run_config={"tts_model_name": "edge"})
    vt.cache = ArtifactCache(vt.job)
    return vt


def _run_tts(vt, segments, tmp_path):
    """Drive `_step_tts_generator` to completion and return its segment list."""
    gen = vt._step_tts_generator(
        segments, tmp_path / "movie.mp4", "te", "en",
        "edge", None, False, False, None, {}, {}, vt._test_vocals,
    )
    result = None
    for item in gen:
        if isinstance(item, list):
            result = item
    return result


class TestTTSResume:

    def test_a_second_run_reuses_every_cached_segment(self, tmp_path, monkeypatch):
        tts = CountingTTS()
        vt = _translator(tmp_path, monkeypatch, tts)
        segments = _segments(4)

        first = _run_tts(vt, segments, tmp_path)
        assert tts.synthesized == 4
        assert len(first) == 4

        # A fresh process: same job id, new objects, empty in-memory state.
        resumed = _translator(tmp_path, monkeypatch, tts)
        resumed.job = jobs.Job.load(vt.job.job_id, jobs_dir=tmp_path / "jobs")
        resumed.manifest = Manifest.load(resumed.job)
        resumed.cache = ArtifactCache(resumed.job)
        tts.batches.clear()

        second = _run_tts(resumed, segments, tmp_path)

        assert tts.synthesized == 0, "a resumed run must not re-synthesize"
        assert len(second) == 4
        # The audio is the same audio, not merely the same number of entries.
        assert [e["audio_path"] for e in second] == [e["audio_path"] for e in first]
        assert [e["start"] for e in second] == [e["start"] for e in first]

    def test_changing_one_line_re_synthesizes_only_that_line(self, tmp_path, monkeypatch):
        tts = CountingTTS()
        vt = _translator(tmp_path, monkeypatch, tts)
        segments = _segments(4)
        _run_tts(vt, segments, tmp_path)

        edited = _segments(4)
        edited[2]["translated_text"] = "a completely different line"

        resumed = _translator(tmp_path, monkeypatch, tts)
        resumed.job = jobs.Job.load(vt.job.job_id, jobs_dir=tmp_path / "jobs")
        resumed.manifest = Manifest.load(resumed.job)
        resumed.cache = ArtifactCache(resumed.job)
        tts.batches.clear()

        result = _run_tts(resumed, edited, tmp_path)

        assert tts.synthesized == 1, "only the edited line should be redone"
        assert tts.batches[0] == ["a completely different line"]
        assert len(result) == 4

    def test_changing_the_model_invalidates_the_cache(self, tmp_path, monkeypatch):
        tts = CountingTTS()
        vt = _translator(tmp_path, monkeypatch, tts)
        _run_tts(vt, _segments(3), tmp_path)

        resumed = _translator(tmp_path, monkeypatch, tts)
        resumed.job = jobs.Job.load(vt.job.job_id, jobs_dir=tmp_path / "jobs")
        resumed.manifest = Manifest.load(resumed.job)
        resumed.cache = ArtifactCache(resumed.job)
        tts.batches.clear()

        gen = resumed._step_tts_generator(
            _segments(3), tmp_path / "movie.mp4", "te", "en",
            "xtts", None, False, False, None, {}, {}, resumed._test_vocals)
        for _ in gen:
            pass

        # A different voice model must not reuse the previous model's audio.
        assert tts.synthesized == 3

    def test_changing_the_speaker_reference_invalidates_that_speaker(self, tmp_path, monkeypatch):
        tts = CountingTTS()
        vt = _translator(tmp_path, monkeypatch, tts)

        ref = tmp_path / "ref.wav"
        sf.write(str(ref), _tone(), RATE)
        diar = [{"start": 0.0, "end": 100.0, "speaker": "SPEAKER_00"}]
        profiles = {"SPEAKER_00": str(ref)}
        ref_audio = {"SPEAKER_00": str(ref)}
        ref_texts = {"SPEAKER_00": "hello"}

        def run(vt):
            gen = vt._step_tts_generator(
                _segments(2), tmp_path / "movie.mp4", "te", "en",
                "edge", None, False, True, diar,
                {"SPEAKER_00": "Female"}, profiles, vt._test_vocals,
                speaker_ref_texts=ref_texts, speaker_ref_audio=ref_audio)
            for _ in gen:
                pass

        run(vt)
        assert tts.synthesized == 2

        # Re-extract the reference: same path, different audio, so a different
        # identity. Reusing the old audio would put the previous voice in.
        sf.write(str(ref), _tone(freq=440.0), RATE)

        resumed = _translator(tmp_path, monkeypatch, tts)
        resumed.job = jobs.Job.load(vt.job.job_id, jobs_dir=tmp_path / "jobs")
        resumed.manifest = Manifest.load(resumed.job)
        resumed.cache = ArtifactCache(resumed.job)
        tts.batches.clear()

        run(resumed)

        assert tts.synthesized == 2, "a new voice must not reuse the old audio"


class TestTTSFailureHandling:

    def test_one_failed_segment_does_not_sink_the_run(self, tmp_path, monkeypatch):
        tts = CountingTTS(fail_indices={1})
        vt = _translator(tmp_path, monkeypatch, tts)

        result = _run_tts(vt, _segments(3), tmp_path)

        # The other two survive; the failed one is simply absent.
        assert len(result) == 2
        assert vt.manifest.segments[1].status == STAGE_FAILED
        assert vt.manifest.segments[0].status == STAGE_COMPLETE
        # The failure is recorded, not swallowed.
        assert any("tts" == e.get("stage") for e in vt.manifest.errors)
        # TTS is partial, not complete, so a resume will try the gap again.
        assert vt.manifest.stages["tts"].status == "PARTIAL"

    def test_a_total_failure_raises_rather_than_dubbing_silence(self, tmp_path, monkeypatch):
        tts = CountingTTS(fail_indices={0, 1, 2})
        vt = _translator(tmp_path, monkeypatch, tts)

        with pytest.raises(RuntimeError, match="failed to synthesize"):
            _run_tts(vt, _segments(3), tmp_path)

    def test_a_failed_segment_is_retried_on_resume(self, tmp_path, monkeypatch):
        # CountingTTS fails by index *within the batch*, and a resumed run only
        # sends the missing line, so a fresh healthy backend is what "recovers"
        # means here.
        failing = CountingTTS(fail_indices={1})
        vt = _translator(tmp_path, monkeypatch, failing)
        _run_tts(vt, _segments(3), tmp_path)
        assert vt.manifest.segments[1].status == STAGE_FAILED
        assert failing.synthesized == 3  # all three were attempted

        healthy = CountingTTS()
        resumed = _translator(tmp_path, monkeypatch, healthy)
        resumed.job = jobs.Job.load(vt.job.job_id, jobs_dir=tmp_path / "jobs")
        resumed.manifest = Manifest.load(resumed.job)
        resumed.cache = ArtifactCache(resumed.job)

        result = _run_tts(resumed, _segments(3), tmp_path)

        assert healthy.synthesized == 1, "only the missing line is regenerated"
        assert len(result) == 3
        assert resumed.manifest.segments[1].status == STAGE_COMPLETE

    def test_silent_output_is_not_cached_as_success(self, tmp_path, monkeypatch):
        class SilentTTS(CountingTTS):
            def generate_batch(self, tasks, model="edge", check_cancel_callback=None):
                self.batches.append([t["text"] for t in tasks])
                results = []
                for t in tasks:
                    path = t["output_path"]
                    sf.write(str(path), np.zeros(int(RATE * 0.5), dtype=np.float32), RATE)
                    results.append(str(path))
                return results

        tts = SilentTTS()
        vt = _translator(tmp_path, monkeypatch, tts)
        _run_tts(vt, _segments(2), tmp_path)

        # Silence produced audio paths, but nothing usable was cached.
        assert vt.cache.stats()["tts_entries"] == 0


class TestManifestProgress:

    def test_manifest_records_the_render_as_complete(self, tmp_path, monkeypatch):
        vt = _translator(tmp_path, monkeypatch, CountingTTS())
        _run_tts(vt, _segments(2), tmp_path)

        vt.manifest.save()
        reloaded = Manifest.load(vt.job)
        counts = reloaded.segment_counts()
        assert counts["complete"] == 2
        assert reloaded.stages["tts"].status == STAGE_COMPLETE

    def test_cached_entries_are_recorded_with_their_cache_key(self, tmp_path, monkeypatch):
        vt = _translator(tmp_path, monkeypatch, CountingTTS())
        _run_tts(vt, _segments(2), tmp_path)

        for index in (0, 1):
            rec = vt.manifest.segments[index]
            assert rec.cache_key, "the cache key must be recorded for resume"
            assert rec.audio_path
            assert rec.translated_text == f"translated line {index}"

    def test_progress_text_reflects_recorded_state(self, tmp_path, monkeypatch):
        vt = _translator(tmp_path, monkeypatch, CountingTTS())
        _run_tts(vt, _segments(2), tmp_path)
        text = vt.manifest.render_progress_text()
        assert isinstance(text, str) and text


class TestChunkNamespacing:

    def test_chunks_do_not_share_stage_output(self, tmp_path, monkeypatch):
        vt = _translator(tmp_path, monkeypatch, CountingTTS())

        vt._chunk_index = 0
        first = vt._stage_output("transcription", "segments.json")
        vt._chunk_index = 1
        second = vt._stage_output("transcription", "segments.json")

        assert first != second
        assert "chunk_000" in str(first) and "chunk_001" in str(second)

    def test_chunks_produce_different_fingerprints(self, tmp_path, monkeypatch):
        vt = _translator(tmp_path, monkeypatch, CountingTTS())
        vt._chunk_index = 0
        first = vt._stage_fingerprint("transcription")
        vt._chunk_index = 1
        assert vt._stage_fingerprint("transcription") != first

    def test_without_a_chunk_index_there_is_no_chunk_subdirectory(self, tmp_path, monkeypatch):
        vt = _translator(tmp_path, monkeypatch, CountingTTS())
        out = vt._stage_output("transcription", "segments.json")
        # No chunk subdirectory: the stage directory is used directly.
        assert out.parent == vt.job.stage_dir("transcription")

    def test_a_chunked_run_nests_under_the_stage(self, tmp_path, monkeypatch):
        vt = _translator(tmp_path, monkeypatch, CountingTTS())
        vt._chunk_index = 2
        out = vt._stage_output("transcription", "segments.json")
        assert out.parent.parent == vt.job.stage_dir("transcription")

    def test_a_completed_chunk_is_not_reprocessed_after_a_restart(
            self, tmp_path, monkeypatch):
        """
        chunk COMPLETE -> restart -> valid artifact found -> chunk not reprocessed.

        This drives the real chunk loop with a recording pipeline, so it measures
        the behaviour rather than the source order. The regression it guards:
        the loop used to reset a chunk to PENDING before the reuse check, so the
        check never saw the COMPLETE record and every chunk was redone on every
        resume while the code still looked like it supported chunk resume.
        """
        vt = _translator(tmp_path, monkeypatch, CountingTTS())

        # validate_video is a real binary check; stub only that so the test's
        # synthetic placeholder file is accepted regardless of whether ffprobe
        # is installed in the environment. The reuse path itself stays real.
        monkeypatch.setattr(jobs, "validate_video", lambda p: Path(str(p)).exists())

        # A chunk from a previous run: a valid file plus a COMPLETE record.
        chunk_dir = vt.job.stage_dir("final")
        chunk_dir.mkdir(parents=True, exist_ok=True)
        durable = chunk_dir / "chunk_000.mp4"
        durable.write_bytes(b"\x00" * 2048)
        vt.manifest.upsert_chunk(0, status=STAGE_COMPLETE)
        vt.manifest.save()

        assert vt._chunk_done(0, 3), "the finished chunk must read as done"
        assert vt._valid_artifact("final", "chunk_000.mp4", kind="video") is not None

        calls = []

        class _Chunker:
            max_duration = 1

            def should_chunk(self, path):
                return True

            def split_video(self, path):
                return [tmp_path / "part0.mp4", tmp_path / "part1.mp4"]

            def merge_videos(self, parts, out):
                Path(str(out)).write_bytes(b"\x00" * 2048)
                return str(out)

        def _fake_pipeline(**kw):
            calls.append(kw.get("video_path"))
            out = tmp_path / "chunk_out.mp4"
            out.write_bytes(b"\x00" * 2048)
            yield ("result", str(out))

        monkeypatch.setattr(vt, "_process_pipeline", _fake_pipeline)

        from src.core.video_translator import VideoChunker
        monkeypatch.setattr(VideoChunker, "should_chunk", lambda self, p: True)
        monkeypatch.setattr(VideoChunker, "split_video", lambda self, p: _Chunker().split_video(p))
        monkeypatch.setattr(vt, "unload_all_models", lambda: None)

        list(vt._process_chunked(
            tmp_path / "movie.mp4", _Chunker(),
            source_lang="en", target_lang="te",
            audio_model_name="x", tts_model_name="edge",
            translation_model_name="google", context_model_name="x",
            transcription_model_name="x", optimize_translation=False,
            enable_diarization=False, diarization_model="x",
            min_speakers=1, max_speakers=2, enable_time_stretch=False,
            enable_vad=False, vad_min_silence_duration_ms=250,
            enable_lipsync=False, lipsync_model_name="x", live_portrait_mode="x",
            enable_visual_translation=False, ocr_model_name="x", tts_voice=None,
            transcription_beam_size=1, tts_enable_cfg=False,
            enable_audio_enhancement=False, hf_token=None))

        # Chunk 0 was reused, so the pipeline ran for chunk 1 only.
        assert len(calls) == 1, f"the finished chunk must not be reprocessed; calls={calls}"

    def test_redo_reset_still_happens_for_a_chunk_that_is_not_done(
            self, tmp_path, monkeypatch):
        """A chunk with no recorded completion is reset, then redone."""
        vt = _translator(tmp_path, monkeypatch, CountingTTS())
        assert not vt._chunk_done(0, 12)
        vt.manifest.upsert_chunk(0, status=STAGE_PENDING)
        assert vt.manifest.chunks[0].status == "PENDING"


class TestNoJobBehavesAsBefore:

    def test_without_a_job_nothing_is_cached(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "TEMP_DIR", tmp_path / "temp")
        (tmp_path / "temp").mkdir(parents=True, exist_ok=True)

        tts = CountingTTS()
        vt = VideoTranslator.__new__(VideoTranslator)
        from src.core.session import SessionContext
        vt.session = SessionContext()
        vt.cancel_requested = False
        vt.tts_engine = tts
        vt.device = "cpu"
        # No job, no manifest, no cache: the pre-Phase-6 behaviour.
        vt.job = vt.manifest = vt.cache = None
        vocals = tmp_path / "vocals.wav"
        sf.write(str(vocals), _tone(2.0), RATE)
        vt._test_vocals = str(vocals)

        segments = _segments(3)
        first = _run_tts(vt, segments, tmp_path)
        second = _run_tts(vt, segments, tmp_path)

        assert len(first) == 3 and len(second) == 3
        # Without durability, every run synthesizes: that is the old contract.
        assert tts.synthesized == 6

    def test_stage_output_falls_back_to_the_legacy_path(self, tmp_path):
        vt = VideoTranslator.__new__(VideoTranslator)
        vt.job = vt.manifest = vt.cache = None
        legacy = tmp_path / "legacy"
        assert vt._cached_path("tts", "segments", legacy) == legacy
