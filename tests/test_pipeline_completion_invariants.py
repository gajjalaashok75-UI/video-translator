"""
A render must not be recorded COMPLETE unless it actually dubbed something.

The bug these tests pin down: a translation backend that fails every line (an
outage, a rate limit, an unsupported pair) returns a full-length list of empty
strings rather than raising. The pipeline wrote that list to disk, marked the
translation stage complete, synthesized nothing, muxed the original video with
an accompaniment-only track, and reported `status = COMPLETE`. The user got a
"successful" render with no translated dialogue and no dubbed speech.

Each stage now has an explicit success condition, checked before the stage is
recorded complete:

  * translation: source speech present => at least one usable translation
  * TTS:         translated text present => at least one synthesized segment
  * final:       source speech present => at least one synthesized segment

The final check is defence in depth: the earlier stages normally fail first, but
a job must never be marked complete on a silent dub even if one of them is
bypassed. A legitimately speechless source is not an error at any stage.
"""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from unittest.mock import MagicMock

from src.core import jobs as _jobs
from src.core.cache import ArtifactCache
from src.core.manifest import Manifest
from src.core.session import SessionContext
from src.core.video_translator import VideoTranslator
from src.utils import config


# ---------------------------------------------------------------------------
# Pure invariant functions
# ---------------------------------------------------------------------------

def _seg(text="", translated="", **extra):
    seg = {"start": 0.0, "end": 1.0, "text": text,
           "translated_text": translated}
    seg.update(extra)
    return seg


class TestInvariantFunctions:

    def test_translation_ok_when_every_line_has_text(self):
        src = [_seg(text="hello"), _seg(text="world")]
        out = [_seg(text="hello", translated="hola"),
               _seg(text="world", translated="mundo")]
        assert VideoTranslator._translation_invariant_error(src, out) is None

    def test_translation_ok_when_some_lines_succeed(self):
        # One failure among otherwise usable lines is tolerated: a partial dub is
        # still a dub, and the per-segment failure is recorded elsewhere.
        src = [_seg(text="hello"), _seg(text="world")]
        out = [_seg(text="hello", translated="hola"), _seg(text="world", translated="")]
        assert VideoTranslator._translation_invariant_error(src, out) is None

    def test_translation_fails_when_every_line_is_empty(self):
        src = [_seg(text="hello"), _seg(text="world")]
        out = [_seg(text="hello", translated=""), _seg(text="world", translated="")]
        err = VideoTranslator._translation_invariant_error(src, out)
        assert err is not None
        assert "2 speech segment(s)" in err

    def test_translation_treats_whitespace_as_empty(self):
        src = [_seg(text="hello")]
        out = [_seg(text="hello", translated="   \n\t ")]
        assert VideoTranslator._translation_invariant_error(src, out) is not None

    def test_translation_passes_when_source_has_no_speech(self):
        # Nothing to translate is not a translation failure.
        src = [_seg(text=""), _seg(text="   ")]
        out = [_seg(text="", translated=""), _seg(text="", translated="")]
        assert VideoTranslator._translation_invariant_error(src, out) is None

    def test_translation_ignores_non_dict_entries(self):
        src = [_seg(text="hello"), "not-a-dict"]
        out = [_seg(text="hello", translated="hola"), None]
        assert VideoTranslator._translation_invariant_error(src, out) is None

    def test_tts_fails_when_dialogue_yields_no_speech(self):
        translated = [_seg(text="hello", translated="hola"),
                      _seg(text="world", translated="mundo")]
        err = VideoTranslator._tts_invariant_error(translated, [])
        assert err is not None
        assert "2 translated segment(s)" in err

    def test_tts_passes_when_speech_was_generated(self):
        translated = [_seg(text="hello", translated="hola")]
        assert VideoTranslator._tts_invariant_error(
            translated, [{"audio_path": "seg0.wav"}]) is None

    def test_tts_passes_when_there_is_no_dialogue(self):
        translated = [_seg(text="", translated="")]
        assert VideoTranslator._tts_invariant_error(translated, []) is None

    def test_final_fails_on_silent_dub(self):
        src = [_seg(text="hello")]
        err = VideoTranslator._final_invariant_error(src, [])
        assert err is not None
        assert "1 speech segment(s)" in err

    def test_final_passes_with_speech_out(self):
        src = [_seg(text="hello")]
        assert VideoTranslator._final_invariant_error(
            src, [{"audio_path": "seg0.wav"}]) is None

    def test_final_passes_for_speechless_source(self):
        src = [_seg(text="")]
        assert VideoTranslator._final_invariant_error(src, []) is None


# ---------------------------------------------------------------------------
# The pipeline end to end, with only the model collaborators faked
# ---------------------------------------------------------------------------

def _kwargs(**overrides):
    base = dict(
        video_path="movie.mp4", source_lang="en", target_lang="te",
        audio_model_name="demucs", tts_model_name="edge",
        translation_model_name="google", context_model_name="m",
        transcription_model_name="large-v3", optimize_translation=False,
        enable_diarization=False, diarization_model="sb",
        min_speakers=1, max_speakers=2, enable_time_stretch=False,
        enable_vad=False, vad_min_silence_duration_ms=1000,
        enable_lipsync=False, lipsync_model_name=None,
        live_portrait_mode="ort", enable_visual_translation=False,
        ocr_model_name="PaddleOCR", tts_voice=None, transcription_beam_size=1,
        tts_enable_cfg=False, enable_audio_enhancement=False, hf_token=None)
    base.update(overrides)
    return base


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """
    A real `VideoTranslator` wired to real durable state and fake collaborators.

    Every stage boundary is exercised; only the heavy models are replaced. The
    translation and TTS collaborators are the ones under test, so they are left
    for each test to configure.
    """
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path / "temp")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    for d in ("temp", "out"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_jobs, "validate_video", lambda p: Path(str(p)).exists())
    monkeypatch.setattr(_jobs, "validate_wav",
                        lambda p, min_duration=0.05: Path(str(p)).exists())

    src = tmp_path / "movie.mp4"
    src.write_bytes(b"\x00" * 4096)

    vt = VideoTranslator.__new__(VideoTranslator)
    vt.session = SessionContext()
    vt.cancel_requested = False
    vt.device = "cpu"
    vt.timing_diagnostics = None
    vt.job = _jobs.Job.create(source_path=str(src))
    vt.manifest = Manifest.create(
        vt.job, source_path=str(src), source_lang="en", target_lang="te",
        run_config={"tts_model_name": "edge"})
    vt.cache = ArtifactCache(vt.job)
    vt.load_model = lambda *a, **k: None
    vt.unload_all_models = lambda: None

    def _write(name, content=b"\x00" * 4096):
        p = tmp_path / "temp" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return str(p)

    def _write_wav(name, seconds=0.1):
        p = tmp_path / "temp" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), np.zeros(int(24000 * seconds), dtype=np.float32), 24000)
        return str(p)

    vt._step_extraction = lambda *a, **k: _write_wav("full.wav")
    vt._step_separation = lambda *a, **k: (_write_wav("vocals.wav"),
                                           _write_wav("bg.wav"))
    vt._step_transcription = lambda *a, **k: (
        [{"text": "hello", "start": 0.0, "end": 1.0}], "en")
    vt._step_duration_aware_text = lambda segs, *a, **k: segs

    vt.transcriber = MagicMock()
    vt.transcriber.merge_short_segments.side_effect = lambda segs, **k: segs
    vt.diarizer = MagicMock()
    vt.diarizer.diarize.return_value = []
    vt.lipsyncer = MagicMock()
    vt.synchronizer = MagicMock()

    def fake_merge_mix(segments, video_path, *a, **k):
        return _write_wav("merged.wav"), _write_wav("mix.wav")

    vt._step_merge_mix = fake_merge_mix

    def fake_replace(video_path, audio_path, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"\x02" * 4096)
        return str(output_path)

    vt.processor = MagicMock()
    vt.processor.replace_audio.side_effect = fake_replace

    return vt, tmp_path, src


def _tts_producing(n):
    """A TTS stand-in that produces `n` real audio files for the segments."""
    def _gen(segments, video_path, target_code, source_code, *a, **k):
        produced = []
        for i, seg in enumerate(segments[:n]):
            path = Path(video_path).parent / f"seg{i}.wav"
            sf.write(str(path), np.zeros(2400, dtype=np.float32), 24000)
            entry = dict(seg)
            entry["audio_path"] = str(path)
            entry.setdefault("segment_id", i)
            produced.append(entry)
        yield produced
    return _gen


class TestPipelineCompletion:

    def test_all_empty_translations_fail_the_render(self, pipeline):
        """
        The reported bug, reproduced: the backend returns empties, nothing is
        dubbed, and the job must FAIL rather than report COMPLETE.
        """
        vt, tmp_path, src = pipeline

        def empty_translation(segments, *a, **k):
            return [dict(s, translated_text="") for s in segments]

        vt.translator = MagicMock()
        vt.translator.translate_segments.side_effect = empty_translation
        vt.translator.last_failures = []
        vt._step_tts_generator = _tts_producing(0)

        with pytest.raises(RuntimeError) as excinfo:
            list(vt._process_pipeline(**_kwargs(video_path=str(src))))

        assert "Translation failed" in str(excinfo.value)
        assert vt.manifest.status != "COMPLETE"
        assert vt.manifest.stages["translation"].status == "FAILED"

    def test_usable_translation_but_no_speech_fails_the_render(self, pipeline):
        """Translation succeeded; synthesis produced nothing. Still not a dub."""
        vt, tmp_path, src = pipeline

        def good_translation(segments, *a, **k):
            return [dict(s, translated_text="hola") for s in segments]

        vt.translator = MagicMock()
        vt.translator.translate_segments.side_effect = good_translation
        vt.translator.last_failures = []
        vt._step_tts_generator = _tts_producing(0)

        with pytest.raises(RuntimeError) as excinfo:
            list(vt._process_pipeline(**_kwargs(video_path=str(src))))

        assert "Speech synthesis failed" in str(excinfo.value)
        assert vt.manifest.status != "COMPLETE"

    def test_final_guard_blocks_complete_on_a_silent_dub(self, pipeline,
                                                         monkeypatch):
        """
        Defence in depth: even with translation and TTS checks bypassed, the
        job must not be recorded COMPLETE with no dubbed speech.
        """
        vt, tmp_path, src = pipeline

        def good_translation(segments, *a, **k):
            return [dict(s, translated_text="hola") for s in segments]

        vt.translator = MagicMock()
        vt.translator.translate_segments.side_effect = good_translation
        vt.translator.last_failures = []
        # Empty TTS output, and the TTS invariant is neutralised so the test
        # reaches the final guard the way a future refactor might.
        vt._step_tts_generator = _tts_producing(0)
        monkeypatch.setattr(VideoTranslator, "_tts_invariant_error",
                            lambda self, *a, **k: None)

        with pytest.raises(RuntimeError) as excinfo:
            list(vt._process_pipeline(**_kwargs(video_path=str(src))))

        assert "Refusing to mark the job complete" in str(excinfo.value)
        assert vt.manifest.status != "COMPLETE"

    def test_healthy_render_still_completes(self, pipeline):
        """Zero regressions: a normal render is untouched by the guards."""
        vt, tmp_path, src = pipeline

        def good_translation(segments, *a, **k):
            return [dict(s, translated_text="hola") for s in segments]

        vt.translator = MagicMock()
        vt.translator.translate_segments.side_effect = good_translation
        vt.translator.last_failures = []
        vt._step_tts_generator = _tts_producing(1)

        events = list(vt._process_pipeline(**_kwargs(video_path=str(src))))

        assert vt.manifest.status == "COMPLETE"
        assert any(e[0] == "result" for e in events if isinstance(e, tuple))
        assert vt.manifest.stages["translation"].status == "COMPLETE"

    def test_speechless_source_completes_without_error(self, pipeline):
        """A video with no speech is not a failure at any stage."""
        vt, tmp_path, src = pipeline

        vt._step_transcription = lambda *a, **k: (
            [{"text": "", "start": 0.0, "end": 1.0}], "en")

        def empty_translation(segments, *a, **k):
            return [dict(s, translated_text="") for s in segments]

        vt.translator = MagicMock()
        vt.translator.translate_segments.side_effect = empty_translation
        vt.translator.last_failures = []
        vt._step_tts_generator = _tts_producing(0)

        list(vt._process_pipeline(**_kwargs(video_path=str(src))))

        assert vt.manifest.status == "COMPLETE"

    def test_a_failed_translation_is_retried_not_reused_on_resume(self, pipeline):
        """
        The failed stage must stay FAILED, so a second run re-runs translation
        instead of reading an empty translation back as cached work.
        """
        vt, tmp_path, src = pipeline
        kwargs = _kwargs(video_path=str(src))

        def empty_translation(segments, *a, **k):
            return [dict(s, translated_text="") for s in segments]

        vt.translator = MagicMock()
        vt.translator.translate_segments.side_effect = empty_translation
        vt.translator.last_failures = []
        vt._step_tts_generator = _tts_producing(0)

        with pytest.raises(RuntimeError):
            list(vt._process_pipeline(**kwargs))
        assert vt.manifest.stages["translation"].status == "FAILED"

        # The backend recovers; a second run over the same job must translate
        # again rather than reuse the failed attempt.
        calls = []

        def good_translation(segments, *a, **k):
            calls.append(len(segments))
            return [dict(s, translated_text="hola") for s in segments]

        vt.translator = MagicMock()
        vt.translator.translate_segments.side_effect = good_translation
        vt.translator.last_failures = []
        vt._step_tts_generator = _tts_producing(1)

        list(vt._process_pipeline(**kwargs))

        assert calls, "translation must be retried on the second run"
        assert vt.manifest.status == "COMPLETE"
