"""
Phase 8: subtitle output modes.

Three layers are covered here, each at the boundary that can actually fail:

  * the VTT writer, against the exact text a player must parse;
  * `Manifest.skip_stage`, because "no subtitles" is a decision that has to
    survive a restart rather than a missing file;
  * `VideoProcessor.add_subtitles`, against real ffmpeg output, because the
    burn-in path is the one place where an escaping mistake produces a video
    with no visible subtitles and no error;
  * the pipeline's subtitle stage and final mux, driven with a fake processor
    so the mode -> behaviour contract is checked without a render.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from unittest.mock import MagicMock

import numpy as np
import soundfile as sf

from src.core.manifest import Manifest, STAGE_SKIPPED, STAGE_COMPLETE
from src.core.cache import ArtifactCache
from src.core import jobs
from src.core.session import SessionContext
from src.core.video_translator import VideoTranslator
from src.processing.video import VideoProcessor
from src.utils import config, srt_generator

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not installed")


# ---------------------------------------------------------------------------
# VTT writer
# ---------------------------------------------------------------------------
class TestVttGenerator:

    def test_header_and_cue_shape(self, tmp_path):
        segments = [
            {"start": 1.0, "end": 2.5, "translated_text": "Hello world."},
            {"start": 3.0, "end": 4.0, "translated_text": "Second line."},
        ]
        out = tmp_path / "s.vtt"
        path = srt_generator.generate_vtt(segments, out)

        text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
        assert text.startswith("WEBVTT")
        # WebVTT uses a dot before milliseconds, unlike SRT's comma.
        assert "00:00:01.000 --> 00:00:02.500" in text
        assert "00:00:03.000 --> 00:00:04.000" in text
        assert "Hello world." in text
        assert "," not in text.split("WEBVTT", 1)[1].split("-->")[0]

    def test_timestamps_use_dots_not_commas(self):
        assert srt_generator.format_vtt_timestamp(0.0) == "00:00:00.000"
        assert srt_generator.format_vtt_timestamp(3661.234) == "01:01:01.234"

    def test_negative_times_are_clamped(self):
        """A bad segment start must not emit a negative cue time."""
        assert srt_generator.format_vtt_timestamp(-1.0) == "00:00:00.000"
        assert srt_generator.format_timestamp(-1.0) == "00:00:00,000"

    def test_empty_text_segments_are_skipped(self, tmp_path):
        segments = [
            {"start": 0.0, "end": 1.0, "translated_text": "   "},
            {"start": 1.0, "end": 2.0, "translated_text": "real"},
        ]
        path = srt_generator.generate_vtt(segments, tmp_path / "s.vtt")
        text = Path(path).read_text(encoding="utf-8")
        assert "real" in text
        assert text.count("-->") == 1

    def test_srt_output_is_unchanged_by_the_vtt_addition(self, tmp_path):
        """The existing SRT format must keep its comma timestamps."""
        segments = [{"start": 1.0, "end": 2.5, "translated_text": "Hi"}]
        path = srt_generator.generate_srt(segments, tmp_path / "s.srt")
        text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
        assert "00:00:01,000 --> 00:00:02,500" in text


# ---------------------------------------------------------------------------
# Manifest: a skipped stage is terminal, but is not a result
# ---------------------------------------------------------------------------
class TestSkippedStage:

    @pytest.fixture
    def job(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
        (tmp_path / "jobs").mkdir(parents=True, exist_ok=True)
        return jobs.Job.create(source_path="/tmp/movie.mp4")

    def test_skip_records_a_terminal_status(self, job):
        m = Manifest.create(job, source_path="/tmp/m.mp4", source_lang="en",
                            target_lang="te", run_config={})
        m.start_stage("subtitles", "fp1")
        m.skip_stage("subtitles", "subtitle_mode=none")

        assert m.stages["subtitles"].status == STAGE_SKIPPED
        assert m.stages["subtitles"].detail["skipped"] == "subtitle_mode=none"
        # Terminal-OK: a resume must not re-run it.
        assert m.stage_is_valid("subtitles", "fp1")

    def test_skip_is_not_completion(self, job):
        """SKIPPED and COMPLETE are both terminal but not the same claim."""
        m = Manifest.create(job, source_path="/tmp/m.mp4", source_lang="en",
                            target_lang="te", run_config={})
        m.start_stage("subtitles", "fp1")
        m.skip_stage("subtitles", "no text")
        assert m.stages["subtitles"].status != STAGE_COMPLETE

    def test_a_skip_survives_a_round_trip(self, job):
        m = Manifest.create(job, source_path="/tmp/m.mp4", source_lang="en",
                            target_lang="te", run_config={})
        m.start_stage("subtitles", "fp1")
        m.skip_stage("subtitles", "subtitle_mode=none")
        m.save()

        reloaded = Manifest.load(job)
        assert reloaded.stages["subtitles"].status == STAGE_SKIPPED
        assert reloaded.stage_is_valid("subtitles", "fp1")


# ---------------------------------------------------------------------------
# Subtitle modes registry
# ---------------------------------------------------------------------------
class TestSubtitleModeConfig:

    def test_default_is_a_known_mode(self):
        assert config.DEFAULT_SUBTITLE_MODE in config.SUBTITLE_MODES

    def test_external_is_the_default(self):
        assert config.DEFAULT_SUBTITLE_MODE == "external"

    def test_modes_are_unique(self):
        assert len(set(config.SUBTITLE_MODES)) == len(config.SUBTITLE_MODES)

    def test_none_is_offered(self):
        """`none` must be selectable, not merely absent by default."""
        assert "none" in config.SUBTITLE_MODES


# ---------------------------------------------------------------------------
# Real ffmpeg: soft mux and burn-in
# ---------------------------------------------------------------------------
def _make_clip(path: Path, seconds: float = 2.0) -> Path:
    """A tiny h264/aac clip with a solid colour, so a burned cue changes it."""
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c=navy:s=160x120:d={seconds}:r=10",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(path)],
        check=True, capture_output=True,
    )
    return path


def _subtitle_streams(path: Path) -> list[str]:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return [line for line in out.stdout.split() if line]


@needs_ffmpeg
class TestAddSubtitles:

    @pytest.fixture
    def clip(self, tmp_path):
        return _make_clip(tmp_path / "src.mp4")

    @pytest.fixture
    def srt(self, tmp_path):
        path = tmp_path / "s.srt"
        srt_generator.generate_srt(
            [{"start": 0.2, "end": 1.8, "translated_text": "Burned cue"}], path)
        return path

    def test_soft_mux_adds_a_selectable_track(self, clip, srt, tmp_path):
        out = tmp_path / "soft.mp4"
        VideoProcessor().add_subtitles(clip, srt, out, burn=False)
        assert out.exists() and out.stat().st_size > 0
        # A soft track is present as a stream the player can toggle.
        assert _subtitle_streams(out) == ["mov_text"]

    def test_burn_in_produces_a_playable_video(self, clip, srt, tmp_path):
        out = tmp_path / "burned.mp4"
        VideoProcessor().add_subtitles(clip, srt, out, burn=True)
        assert out.exists() and out.stat().st_size > 0
        # Burn-in renders into the frames, so no subtitle stream remains.
        assert _subtitle_streams(out) == []

    def test_burn_in_actually_changes_the_picture(self, clip, srt, tmp_path):
        """
        The failure this guards against is a burn that "succeeds" silently.

        libass does not error on an unreadable subtitle file in every case, so
        the check is on the pixels: a cue drawn over a solid colour must differ
        from the source frame.
        """
        out = tmp_path / "burned.mp4"
        VideoProcessor().add_subtitles(clip, srt, out, burn=True)

        def frame_hash(video: Path, png: Path) -> str:
            subprocess.run(
                [FFMPEG, "-y", "-v", "error", "-ss", "1", "-i", str(video),
                 "-frames:v", "1", str(png)],
                check=True, capture_output=True)
            return __import__("hashlib").md5(png.read_bytes()).hexdigest()

        src_hash = frame_hash(clip, tmp_path / "src.png")
        out_hash = frame_hash(out, tmp_path / "out.png")
        assert src_hash != out_hash, "burn-in left the frame unchanged"

    def test_a_path_with_a_colon_is_handled(self, clip, tmp_path):
        """
        A colon is filtergraph syntax, so an unescaped path silently mis-parses.

        A colon can appear in a real path (a Windows drive letter, a directory
        named `a:b`), so this is not hypothetical.
        """
        srt = tmp_path / "a:b"
        srt.mkdir()
        sub = srt / "sub.srt"
        srt_generator.generate_srt(
            [{"start": 0.2, "end": 1.8, "translated_text": "Colon cue"}], sub)

        soft = tmp_path / "colon_soft.mp4"
        VideoProcessor().add_subtitles(clip, sub, soft, burn=False)
        assert _subtitle_streams(soft) == ["mov_text"]

        burned = tmp_path / "colon_burned.mp4"
        VideoProcessor().add_subtitles(clip, sub, burned, burn=True)
        assert burned.stat().st_size > 0

    def test_a_missing_subtitle_file_raises(self, clip, tmp_path):
        """A missing sidecar must fail loudly, not produce a trackless video."""
        import ffmpeg as ffmpeg_lib
        with pytest.raises(Exception):
            VideoProcessor().add_subtitles(
                clip, tmp_path / "absent.srt", tmp_path / "x.mp4", burn=False)

    def test_escape_quotes_a_plain_path(self, tmp_path):
        escaped = VideoProcessor()._escape_subtitle_path(tmp_path / "s.srt")
        assert escaped.startswith("'") and escaped.endswith("'")

    def test_escape_neutralises_a_colon(self):
        escaped = VideoProcessor()._escape_subtitle_path("/x/a:b/s.srt")
        assert "\\:" in escaped


# ---------------------------------------------------------------------------
# Pipeline wiring: mode -> stage behaviour and final mux
# ---------------------------------------------------------------------------
def _pipeline_kwargs(**overrides):
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
    A real `VideoTranslator` with real durable state and fake heavy models.

    The subtitle stage and the final mux are the code under test, so they run
    for real; only extraction, separation, ASR, translation and TTS are faked.
    """
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path / "temp")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    for d in ("temp", "out"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(jobs, "validate_video", lambda p: Path(str(p)).exists())
    monkeypatch.setattr(jobs, "validate_wav",
                        lambda p, min_duration=0.05: Path(str(p)).exists())

    src = tmp_path / "movie.mp4"
    src.write_bytes(b"\x00" * 4096)

    vt = VideoTranslator.__new__(VideoTranslator)
    vt.session = SessionContext()
    vt.cancel_requested = False
    vt.device = "cpu"
    vt.timing_diagnostics = None
    vt._pending_subtitles = {}
    vt.job = jobs.Job.create(source_path=str(src))
    vt.manifest = Manifest.create(
        vt.job, source_path=str(src), source_lang="en", target_lang="te",
        run_config={"tts_model_name": "edge", "subtitle_mode": "external"})
    vt.cache = ArtifactCache(vt.job)
    vt.load_model = lambda *a, **k: None
    vt.unload_all_models = lambda: None

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

    def fake_translation(segments, *a, **k):
        return [dict(s, translated_text="hola", segment_id=i)
                for i, s in enumerate(segments)]

    vt.translator = MagicMock()
    vt.translator.translate_segments.side_effect = fake_translation
    vt.translator.last_failures = []

    def fake_tts(segments, video_path, target_code, source_code, *a, **k):
        produced = []
        for i, seg in enumerate(segments):
            path = Path(video_path).parent / f"seg{i}.wav"
            sf.write(str(path), np.zeros(2400, dtype=np.float32), 24000)
            entry = dict(seg)
            entry["audio_path"] = str(path)
            entry.setdefault("segment_id", i)
            produced.append(entry)
        yield produced

    vt._step_tts_generator = fake_tts
    return vt, tmp_path, src


class TestSubtitleStageWiring:

    def test_external_mode_writes_sidecar_files(self, pipeline):
        vt, tmp_path, src = pipeline
        list(vt._process_pipeline(
            **_pipeline_kwargs(video_path=str(src), subtitle_mode="external")))

        assert vt.manifest.stages["subtitles"].status == STAGE_COMPLETE
        srt = Path(vt.manifest.artifacts["subtitles_srt"])
        vtt = Path(vt.manifest.artifacts["subtitles_vtt"])
        assert srt.exists() and srt.suffix == ".srt"
        assert vtt.exists() and vtt.suffix == ".vtt"

    def test_none_mode_writes_nothing_and_is_recorded_skipped(self, pipeline):
        vt, tmp_path, src = pipeline
        list(vt._process_pipeline(
            **_pipeline_kwargs(video_path=str(src), subtitle_mode="none")))

        assert vt.manifest.stages["subtitles"].status == STAGE_SKIPPED
        assert vt.manifest.stages["subtitles"].detail["skipped"] == "subtitle_mode=none"
        assert "subtitles_srt" not in vt.manifest.artifacts
        assert not (vt.job.stage_dir("subtitles")).exists() or not list(
            vt.job.stage_dir("subtitles").glob("*.srt"))

    def test_an_unknown_mode_falls_back_to_external(self, pipeline):
        """A typo must still produce subtitles, not a silent no-subtitle job."""
        vt, tmp_path, src = pipeline
        list(vt._process_pipeline(
            **_pipeline_kwargs(video_path=str(src), subtitle_mode="bogus")))
        assert vt.manifest.stages["subtitles"].status == STAGE_COMPLETE

    def test_none_mode_survives_a_resume(self, pipeline):
        """A resumed job must not re-add subtitles the request excluded."""
        vt, tmp_path, src = pipeline
        kwargs = _pipeline_kwargs(video_path=str(src), subtitle_mode="none")
        list(vt._process_pipeline(**kwargs))
        assert vt.manifest.stages["subtitles"].status == STAGE_SKIPPED

        list(vt._process_pipeline(**kwargs))
        assert vt.manifest.stages["subtitles"].status == STAGE_SKIPPED
        assert "subtitles_srt" not in vt.manifest.artifacts

    def test_embedded_mode_calls_the_processor(self, pipeline):
        vt, tmp_path, src = pipeline
        list(vt._process_pipeline(
            **_pipeline_kwargs(video_path=str(src), subtitle_mode="embedded")))

        vt.processor.add_subtitles.assert_called_once()
        args, kwargs = vt.processor.add_subtitles.call_args
        assert kwargs.get("burn") is False
        assert str(args[1]).endswith(".srt")

    def test_burned_mode_asks_for_a_burn(self, pipeline):
        vt, tmp_path, src = pipeline
        list(vt._process_pipeline(
            **_pipeline_kwargs(video_path=str(src), subtitle_mode="burned")))
        _, kwargs = vt.processor.add_subtitles.call_args
        assert kwargs.get("burn") is True

    def test_external_mode_does_not_touch_the_video(self, pipeline):
        """External means sidecar only: the mux must be left alone."""
        vt, tmp_path, src = pipeline
        list(vt._process_pipeline(
            **_pipeline_kwargs(video_path=str(src), subtitle_mode="external")))
        vt.processor.add_subtitles.assert_not_called()

    def test_a_failed_embed_keeps_the_plain_video(self, pipeline):
        """
        A subtitle failure must not fail a render whose picture and sound are
        already correct.
        """
        vt, tmp_path, src = pipeline
        vt.processor.add_subtitles.side_effect = RuntimeError("libass exploded")
        list(vt._process_pipeline(
            **_pipeline_kwargs(video_path=str(src), subtitle_mode="embedded")))

        assert vt.manifest.status == "COMPLETE"
        final = Path(vt.manifest.artifacts["final_output"])
        assert final.exists()

    def test_the_subtitled_video_is_recorded_as_the_output(self, pipeline):
        vt, tmp_path, src = pipeline

        def fake_add(video, subs, out, burn=False):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"\x03" * 4096)
            return out

        vt.processor.add_subtitles.side_effect = fake_add
        list(vt._process_pipeline(
            **_pipeline_kwargs(video_path=str(src), subtitle_mode="embedded")))

        final = Path(vt.manifest.artifacts["final_output"])
        assert final.exists()
        assert "subtitled" in final.name


class TestAudioArtifacts:
    """
    The audio deliverables are produced by the mix stage, not by subtitles.

    These are the A8 requirements: the final mix and the dialogue-only track
    must be durable, first-class artifacts, and the background/accompaniment
    track must be exposed under an accurate name. Crucially they must exist
    regardless of the subtitle mode, because the mix is already computed by the
    time the subtitle choice is applied.
    """

    @pytest.mark.parametrize("mode", ["none", "external", "embedded", "burned"])
    def test_the_final_mix_is_persisted_for_every_subtitle_mode(self, pipeline, mode):
        vt, tmp_path, src = pipeline
        list(vt._process_pipeline(
            **_pipeline_kwargs(video_path=str(src), subtitle_mode=mode)))

        mixed = Path(vt.manifest.artifacts["audio_mixed"])
        assert mixed.exists(), f"audio_mixed missing for mode={mode}"
        assert mixed.parent == vt.job.stage_dir("final")

    @pytest.mark.parametrize("mode", ["none", "external", "embedded", "burned"])
    def test_the_dialogue_track_is_persisted_for_every_subtitle_mode(self, pipeline, mode):
        vt, tmp_path, src = pipeline
        list(vt._process_pipeline(
            **_pipeline_kwargs(video_path=str(src), subtitle_mode=mode)))

        dialogue = Path(vt.manifest.artifacts["audio_dialogue"])
        assert dialogue.exists(), f"audio_dialogue missing for mode={mode}"

    def test_the_background_track_is_persisted(self, pipeline):
        vt, tmp_path, src = pipeline
        list(vt._process_pipeline(
            **_pipeline_kwargs(video_path=str(src), subtitle_mode="external")))

        background = Path(vt.manifest.artifacts["audio_background"])
        assert background.exists()
        assert background.name.endswith("_audio_background.wav")

    def test_the_audio_artifacts_are_the_files_the_mix_produced(self, pipeline):
        """
        The artifact must be a copy of the real mix, not a second mixdown: the
        same bytes on disk, so re-encoding cannot silently change the audio.
        """
        vt, tmp_path, src = pipeline
        list(vt._process_pipeline(
            **_pipeline_kwargs(video_path=str(src), subtitle_mode="external")))

        mixed = Path(vt.manifest.artifacts["audio_mixed"])
        source = tmp_path / "temp" / "mix.wav"
        assert mixed.read_bytes() == source.read_bytes()

    def test_the_audio_artifacts_survive_a_restart(self, pipeline):
        """A re-read manifest must still resolve the same durable files."""
        vt, tmp_path, src = pipeline
        list(vt._process_pipeline(
            **_pipeline_kwargs(video_path=str(src), subtitle_mode="external")))
        recorded = dict(vt.manifest.artifacts)

        reloaded = Manifest.load(vt.job)
        assert reloaded.artifacts["audio_mixed"] == recorded["audio_mixed"]
        assert Path(reloaded.artifacts["audio_mixed"]).exists()
