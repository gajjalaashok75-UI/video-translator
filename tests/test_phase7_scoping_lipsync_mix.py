"""
Phase 7 focused tests: scoped invalidation, lip-sync checkpointing, the
multi-face safe policy, and the dialogue/background mix levels.

These exercise the real code paths (real manifest, real renderer, real mixer);
only the ML models are replaced, since they need weights this environment does
not have.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from src.audio import mix_loudness
from src.core import jobs as joblib
from src.core.manifest import Manifest
from src.processing.synchronization import AudioSynchronizer
from src.processing.wav2lip import Wav2LipSyncer
from src.utils import config


def _run_config(**overrides):
    cfg = {
        "source_lang": "en", "target_lang": "te", "audio_model_name": "demucs",
        "tts_model_name": "edge", "translation_model_name": "nllb",
        "context_model_name": "m", "transcription_model_name": "large-v3",
        "optimize_translation": True, "enable_diarization": True,
        "diarization_model": "sb", "min_speakers": 1, "max_speakers": 10,
        "enable_time_stretch": True, "enable_vad": True,
        "vad_min_silence_duration_ms": 1000, "enable_lipsync": False,
        "lipsync_model_name": None, "live_portrait_mode": None,
        "enable_visual_translation": False, "ocr_model_name": "PaddleOCR",
        "tts_voice": None, "transcription_beam_size": 5,
        "tts_enable_cfg": False, "enable_audio_enhancement": False,
        "chunk_duration": 60,
    }
    cfg.update(overrides)
    return cfg


class _Fingerprinter:
    """Borrows the real `_stage_fingerprint` without building a translator."""

    def __init__(self, manifest):
        from src.core.video_translator import VideoTranslator
        self.manifest = manifest
        self.job = None
        self._chunk_index = None
        self._stage_fingerprint = VideoTranslator._stage_fingerprint.__get__(self)


@pytest.fixture
def job_manifest(tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"\x00" * 4096)
    job = joblib.Job.open_or_create(None)
    manifest = Manifest.load_or_create(
        job, source_path=src, source_lang="en", target_lang="te",
        run_config=_run_config(), software={"app": "test"})
    manifest.job_id = job.job_id
    manifest.save()
    return job, manifest, src


def _fingerprints(manifest):
    fp = _Fingerprinter(manifest)
    return {s: fp._stage_fingerprint(s)
            for s in config.JOB_STAGES if s in config.STAGE_CONFIG_KEYS}


def test_lipsync_only_change_preserves_upstream_work(job_manifest):
    """Flipping lip-sync must invalidate lip-sync and the mux, nothing else."""
    _, manifest, _ = job_manifest
    before = _fingerprints(manifest)

    changed = _run_config(enable_lipsync=True, lipsync_model_name="wav2lip")
    drifted = manifest.differs_from(changed)
    assert set(drifted) <= set(config.RE_RENDERABLE_CONFIG_KEYS)

    report = manifest.reconcile(changed, reason="test")
    assert set(report["invalidated"]) == {"lipsync", "final"}

    after = _fingerprints(manifest)
    for stage in ("extraction", "separation", "transcription", "diarization",
                  "references", "translation", "tts", "sync", "subtitles"):
        assert before[stage] == after[stage], f"{stage} should be preserved"
    assert before["lipsync"] != after["lipsync"]
    assert before["final"] != after["final"]


def test_translation_change_is_not_re_renderable(job_manifest):
    """A change that affects cached audio must not be treated as re-renderable."""
    _, manifest, _ = job_manifest
    drifted = manifest.differs_from(_run_config(translation_model_name="other"))
    assert "translation_model_name" in drifted
    assert not set(drifted) <= set(config.RE_RENDERABLE_CONFIG_KEYS)


def test_upstream_change_does_not_invalidate_downstream_fingerprints(job_manifest):
    """Fingerprints are cumulative: a change to `tts` must not alter `separation`."""
    _, manifest, _ = job_manifest
    before = _fingerprints(manifest)
    manifest.reconcile(_run_config(tts_model_name="indicf5"), reason="test")
    after = _fingerprints(manifest)
    assert before["separation"] == after["separation"]
    assert before["transcription"] == after["transcription"]
    assert before["tts"] != after["tts"]


def test_manifest_marker_blocks_under_invalidation(job_manifest):
    """A manifest written under a different fingerprint scheme must not match.

    The marker is folded into every stage fingerprint, so work recorded by an
    older scheme (or by hand, without the marker) can never be mistaken for
    current work.
    """
    _, manifest, _ = job_manifest
    assert manifest.source.get("fingerprint_scope") == config.FINGERPRINT_SCOPE

    stale_job = joblib.Job.open_or_create(None)
    stale_manifest = Manifest.create(
        stale_job, source_path=manifest.source["path"], source_lang="en",
        target_lang="te", run_config=_run_config(), software={"app": "test"})
    # A manifest written before Phase 7 has no scope marker at all.
    stale_manifest.source.pop("fingerprint_scope", None)

    fresh_fp = _Fingerprinter(manifest)._stage_fingerprint("separation")
    stale_fp = _Fingerprinter(stale_manifest)._stage_fingerprint("separation")
    assert fresh_fp != stale_fp, "a different fingerprint scheme must not match"


# --- Lip-sync checkpointing ---

def _write_tiny_video(path):
    """A real (tiny) mp4, so existence/validation checks run on a real file."""
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 32))
    for _ in range(5):
        w.write(np.zeros((32, 32, 3), dtype=np.uint8))
    w.release()


def test_lipsync_checkpoint_is_reused_when_complete(job_manifest, monkeypatch):
    """A completed lip-sync stage with a valid artifact must be reused."""
    from src.core.video_translator import VideoTranslator
    job, manifest, _ = job_manifest
    # ffprobe is not guaranteed in this environment; the artifact is a real
    # file and its existence is what the reuse decision needs here.
    monkeypatch.setattr(joblib, "validate_video", lambda p: True)

    vt = VideoTranslator()
    vt.job = job
    vt.manifest = manifest
    durable = vt._stage_output("lipsync", "lipsynced.mp4")
    _write_tiny_video(durable)

    fp = vt._begin_stage("lipsync")
    vt._finish_stage("lipsync", {"output": str(durable)})

    assert vt._valid_artifact("lipsync", "lipsynced.mp4", kind="video") == durable
    assert vt._stage_done("lipsync") is True
    assert manifest.stage("lipsync").detail["output"] == str(durable)


def test_lipsync_output_missing_is_not_a_checkpoint(job_manifest, monkeypatch):
    """A completed record whose file is gone must not count as a checkpoint."""
    from src.core.video_translator import VideoTranslator
    job, manifest, _ = job_manifest
    monkeypatch.setattr(joblib, "validate_video", lambda p: True)

    vt = VideoTranslator()
    vt.job = job
    vt.manifest = manifest
    vt._begin_stage("lipsync")
    vt._finish_stage("lipsync", {"output": str(job.stage_dir("lipsync") / "gone.mp4")})

    assert vt._valid_artifact("lipsync", "lipsynced.mp4", kind="video") is None


def test_lipsync_failed_stage_is_not_a_checkpoint(job_manifest, monkeypatch):
    """A failed stage must not be treated as a cache hit even if a file exists."""
    from src.core.video_translator import VideoTranslator
    job, manifest, _ = job_manifest
    monkeypatch.setattr(joblib, "validate_video", lambda p: True)

    vt = VideoTranslator()
    vt.job = job
    vt.manifest = manifest
    durable = vt._stage_output("lipsync", "lipsynced.mp4")
    _write_tiny_video(durable)

    vt._begin_stage("lipsync")
    vt._fail_stage("lipsync", "model crashed")

    assert vt._valid_artifact("lipsync", "lipsynced.mp4", kind="video") == durable
    assert vt._stage_done("lipsync") is False, "a failed stage is not done"


# --- Multi-face policy ---

class _Detector:
    def __init__(self, n_faces):
        self.n = n_faces

    def get_landmarks(self, img):
        h, w = img.shape[:2]
        out = []
        for k in range(self.n):
            pts = np.random.RandomState(k).rand(68, 2) * [w * 0.25, h * 0.25]
            pts[:, 0] += w * (0.1 + 0.5 * k)
            pts[:, 1] += h * 0.3
            out.append(pts)
        return out


def _syncer(n_faces):
    s = Wav2LipSyncer()
    s.detector = _Detector(n_faces)
    return s


def test_single_face_is_selected():
    boxes, chosen = _syncer(1)._detect_frame_boxes(np.zeros((120, 160, 3), np.uint8))
    assert len(boxes) == 1
    assert isinstance(chosen, list) and len(chosen) == 4


def test_two_faces_are_ambiguous():
    boxes, chosen = _syncer(2)._detect_frame_boxes(np.zeros((120, 160, 3), np.uint8))
    assert len(boxes) == 2
    assert chosen is None, "must refuse to guess between two actors"


def test_no_faces_yields_nothing():
    boxes, chosen = _syncer(0)._detect_frame_boxes(np.zeros((120, 160, 3), np.uint8))
    assert boxes == [] and chosen is None


def test_face_count_smoothing_filters_stray_and_keeps_edges():
    # A single stray multi-face detection in a single-face shot is removed.
    assert Wav2LipSyncer._smooth_face_counts([1, 1, 1, 2, 1]) == [1, 1, 1, 1, 1]
    # A genuine two-face stretch keeps its edges after smoothing.
    assert all(c >= 2 for c in Wav2LipSyncer._smooth_face_counts([0, 2, 2, 2, 0]))


# --- Mix levels ---

def _tone(freq, secs, level_db, sr=24000):
    t = np.linspace(0, secs, int(secs * sr), endpoint=False)
    x = np.sin(2 * np.pi * freq * t)
    return (x * (10 ** (level_db / 20.0))).astype(np.float32)


def test_background_is_never_boosted():
    """A quiet music bed must be left alone, not raised (which raises its noise)."""
    d = _tone(300, 2, -20)
    b = _tone(120, 2, -60)
    d_gain, b_gain = mix_loudness.estimate_mix_gains(d, b)
    assert d_gain == 0.0
    assert b_gain == 0.0, "background far below the dialogue target must not be boosted"


def test_loud_background_is_reduced_but_not_silenced():
    d = _tone(300, 2, -20)
    b = _tone(120, 2, -5)
    _, b_gain = mix_loudness.estimate_mix_gains(d, b)
    assert b_gain < 0, "a bed louder than the dialogue must come down"
    assert b_gain >= -24.0, "but never be silenced outright"


def test_dialogue_gain_is_bounded():
    """A very quiet line is lifted, but only so far, so dynamics survive."""
    quiet, loud = -45.0, -20.0
    g_quiet = mix_loudness.bounded_gain_db(quiet, loud)
    g_loud = mix_loudness.bounded_gain_db(loud, loud)
    assert g_quiet == mix_loudness.DEFAULT_MAX_DIALOGUE_ADJUST_DB
    assert g_loud == 0.0


def test_render_timeline_reduces_loudness_spread(tmp_path):
    """Two lines at very different levels come out closer together."""
    sr = 24000
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    sf.write(str(a), _tone(300, 1, -35), sr)
    sf.write(str(b), _tone(400, 1, -12), sr)

    segs = [
        {"audio_path": str(a), "start": 0.0, "end": 1.0, "segment_id": 0},
        {"audio_path": str(b), "start": 1.5, "end": 2.5, "segment_id": 1},
    ]
    out = tmp_path / "out.wav"
    AudioSynchronizer().merge_segments(segs, total_duration=3.0, output_path=str(out))

    y, _ = sf.read(str(out))
    l0 = mix_loudness.active_dbfs(y[:sr])
    l1 = mix_loudness.active_dbfs(y[int(1.5 * sr):int(2.5 * sr)])
    assert abs(l0 - l1) < abs(-35 - -12), "spread must shrink"
    assert np.max(np.abs(y)) <= mix_loudness.DEFAULT_CEILING + 1e-6


def test_render_timeline_preserves_overlap(tmp_path):
    """Two intentionally overlapping speakers must both be audible."""
    sr = 24000
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    sf.write(str(a), _tone(300, 4, -15), sr)
    sf.write(str(b), _tone(900, 3, -15), sr)
    segs = [
        {"audio_path": str(a), "start": 10.0, "end": 14.0, "segment_id": 0},
        {"audio_path": str(b), "start": 12.0, "end": 15.0, "segment_id": 1},
    ]
    out = tmp_path / "ov.wav"
    AudioSynchronizer().merge_segments(segs, total_duration=20.0, output_path=str(out))
    y, _ = sf.read(str(out))

    def lvl(s, e):
        return mix_loudness.active_dbfs(y[int(s * sr):int(e * sr)])

    assert lvl(12, 14) > lvl(10, 12) + 2, "overlap region must be louder than either line alone"
    assert lvl(12, 14) > lvl(14, 15) + 2
    assert lvl(15, 20) < -80, "after the last segment the track is silent"


def test_render_timeline_has_no_click_at_onsets(tmp_path):
    """The fades must remove the discontinuity a hard cut would leave."""
    sr = 24000
    a = tmp_path / "a.wav"
    sf.write(str(a), _tone(300, 0.5, -12), sr)
    segs = [{"audio_path": str(a), "start": 1.0, "end": 1.5, "segment_id": 0}]
    out = tmp_path / "c.wav"
    AudioSynchronizer().merge_segments(segs, total_duration=2.0, output_path=str(out))
    y, _ = sf.read(str(out))
    onset = int(1.0 * sr)
    assert abs(y[onset] - y[onset - 1]) < 0.05


# --- End-to-end: a second run must reuse the checkpointed lip-sync ---

def test_lipsync_checkpoint_short_circuits_the_pipeline(tmp_path, monkeypatch):
    """Driving the real pipeline twice must run the model only on the first pass."""
    from src.core.video_translator import VideoTranslator
    from src.core import jobs as _jobs
    from src.core.manifest import Manifest as _Manifest
    from src.core.session import SessionContext

    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path / "temp")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    for d in ("temp", "out"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    # ffprobe may be absent; the artifacts here are real files, which is what
    # the reuse decision needs.
    monkeypatch.setattr(_jobs, "validate_video", lambda p: Path(str(p)).exists())
    monkeypatch.setattr(_jobs, "validate_wav", lambda p, min_duration=0.05: Path(str(p)).exists())

    src = tmp_path / "movie.mp4"
    src.write_bytes(b"\x00" * 4096)

    vt = VideoTranslator.__new__(VideoTranslator)
    vt.session = SessionContext()
    vt.cancel_requested = False
    vt.device = "cpu"
    vt.timing_diagnostics = None
    vt.job = _jobs.Job.create(source_path=str(src))
    vt.manifest = _Manifest.create(
        vt.job, source_path=str(src), source_lang="en", target_lang="te",
        run_config=_run_config(enable_lipsync=True, lipsync_model_name="wav2lip"))
    from src.core.cache import ArtifactCache
    vt.cache = ArtifactCache(vt.job)
    vt.load_model = lambda *a, **k: None
    vt.unload_all_models = lambda: None

    calls = {"lipsync": 0, "sync": 0}

    def fake_artifact(name, content=b"\x00" * 4096):
        def _step(*a, **k):
            p = tmp_path / "temp" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
            return str(p)
        return _step

    # The pipeline calls these collaborators directly; each returns something
    # the next stage can consume, so the wiring is exercised rather than stubbed
    # at the top level.
    vt.transcriber = MagicMock()
    vt.transcriber.merge_short_segments.side_effect = lambda segs, **k: segs
    vt.transcriber.apply_vad.side_effect = lambda segs, *a, **k: segs
    vt.translator = MagicMock()
    # A real translation backend adds `translated_text` to each segment; the
    # stub must do the same, or the pipeline (correctly) refuses to dub nothing.
    vt.translator.translate_segments.side_effect = lambda segs, *a, **k: [
        dict(s, translated_text=s.get("translated_text") or s.get("text", ""))
        for s in segs]
    vt.diarizer = MagicMock()
    vt.diarizer.diarize.return_value = []
    vt.synchronizer = MagicMock()
    vt.lipsyncer = MagicMock()

    vt._step_extraction = fake_artifact("full.wav")
    vt._step_separation = lambda *a, **k: (str(fake_artifact("vocals.wav")()),
                                           str(fake_artifact("bg.wav")()))
    vt._step_transcription = lambda *a, **k: ([{"text": "hi", "start": 0.0, "end": 1.0}], "en")
    vt._step_duration_aware_text = lambda segs, *a, **k: segs

    def fake_tts(*a, **k):
        segs = a[0]
        for i, seg in enumerate(segs):
            p = tmp_path / "temp" / f"seg{i}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), np.zeros(2400, dtype=np.float32), 24000)
            seg["audio_path"] = str(p)
            seg.setdefault("segment_id", i)
        yield segs

    vt._step_tts_generator = fake_tts

    def fake_merge_mix(segments, video_path, *a, **k):
        calls["sync"] += 1
        merged = tmp_path / "temp" / "merged.wav"
        mix = tmp_path / "temp" / "mix.wav"
        sf.write(str(merged), np.zeros(24000, dtype=np.float32), 24000)
        sf.write(str(mix), np.zeros(24000, dtype=np.float32), 24000)
        return str(merged), str(mix)

    vt._step_merge_mix = fake_merge_mix

    def fake_lipsync(video_path, audio_path, model_name):
        calls["lipsync"] += 1
        out = config.TEMP_DIR / f"{Path(video_path).stem}_lipsync.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x01" * 4096)
        return str(out)

    vt._step_lipsync = fake_lipsync

    def fake_replace(video_path, audio_path, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"\x02" * 4096)
        return str(output_path)

    vt.processor = MagicMock()
    vt.processor.replace_audio.side_effect = fake_replace

    kwargs = dict(
        video_path=str(src), source_lang="en", target_lang="te",
        audio_model_name="demucs", tts_model_name="edge",
        translation_model_name="nllb", context_model_name="m",
        transcription_model_name="large-v3", optimize_translation=False,
        enable_diarization=False, diarization_model="sb",
        min_speakers=1, max_speakers=2, enable_time_stretch=True,
        enable_vad=False, vad_min_silence_duration_ms=1000,
        enable_lipsync=True, lipsync_model_name="wav2lip",
        live_portrait_mode="ort", enable_visual_translation=False,
        ocr_model_name="PaddleOCR", tts_voice=None, transcription_beam_size=1,
        tts_enable_cfg=False, enable_audio_enhancement=False, hf_token=None)

    list(vt._process_pipeline(**kwargs))
    assert calls["lipsync"] == 1, "first run must invoke the model"
    assert vt._stage_done("lipsync"), "lip-sync must be checkpointed"

    # Second run on the same job: the checkpoint must be reused.
    list(vt._process_pipeline(**kwargs))
    assert calls["lipsync"] == 1, (
        f"second run must reuse the checkpoint, not re-run the model; calls={calls}")


# --- Quality presets (P5) ---

def test_presets_are_ordered_by_quality():
    fast = config.preset_config("FAST")
    balanced = config.preset_config("BALANCED")
    high = config.preset_config("HIGH")

    # FAST: no lip-sync, no enhancement — the cheapest core dub.
    assert fast["enable_lipsync"] is False
    assert fast["enable_audio_enhancement"] is False

    # BALANCED: standard lip-sync, enhancement still opt-in.
    assert balanced["enable_lipsync"] is True
    assert balanced["enable_audio_enhancement"] is False

    # HIGH: best lip-sync plus enhancement.
    assert high["enable_lipsync"] is True
    assert high["enable_audio_enhancement"] is True
    assert high["lipsync_model_name"] != balanced["lipsync_model_name"]
    assert set(fast.values()) != set(high.values())


def test_presets_only_touch_quality_keys():
    """A preset must not change what is said, nor whose voice says it."""
    forbidden = {
        "source_lang", "target_lang", "translation_model_name",
        "context_model_name", "transcription_model_name",
        "transcription_beam_size", "enable_diarization", "diarization_model",
        "tts_model_name", "tts_voice", "tts_enable_cfg", "enable_time_stretch",
        "min_speakers", "max_speakers", "optimize_translation",
    }
    for name in config.QUALITY_PRESET_NAMES:
        cfg = config.preset_config(name)
        assert not (set(cfg) & forbidden), f"{name} changes non-quality settings"
        assert set(cfg) <= config.PRESET_ALLOWED_KEYS


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        config.preset_config("ULTRA")
    with pytest.raises(ValueError):
        config.preset_config("")


def test_preset_change_invalidates_only_downstream_quality_stages(job_manifest):
    """FAST -> HIGH must not invalidate ASR, diarization, translation or TTS."""
    _, manifest, _ = job_manifest
    fast = _run_config(**config.preset_config("FAST"))
    manifest.reconcile(fast, reason="preset FAST")

    high = _run_config(**config.preset_config("HIGH"))
    drifted = manifest.differs_from(high)
    assert drifted, "HIGH must differ from FAST"
    assert set(drifted) <= set(config.RE_RENDERABLE_CONFIG_KEYS)

    report = manifest.reconcile(high, reason="preset HIGH")
    invalidated = set(report["invalidated"])
    assert "lipsync" in invalidated and "final" in invalidated
    for preserved in ("transcription", "diarization", "references",
                      "translation", "tts"):
        assert preserved not in invalidated, f"{preserved} must stay valid"


def test_process_video_applies_the_preset_over_the_arguments(monkeypatch):
    """An explicit preset wins over the individual quality arguments."""
    from src.core.video_translator import VideoTranslator

    captured = {}

    def fake_pipeline(self, **kwargs):
        captured.update(kwargs)
        return iter(())

    monkeypatch.setattr(VideoTranslator, "_process_pipeline", fake_pipeline)
    monkeypatch.setattr(VideoTranslator, "_run_config", staticmethod(lambda lv: {}))
    monkeypatch.setattr("src.core.video_translator.config.validate_path",
                        lambda p, **k: Path(p))

    vt = VideoTranslator()
    vt.job = None  # no durable state, so no manifest/diff path
    vt.manifest = None

    list(vt.process_video(
        "m.mp4", enable_lipsync=False, lipsync_model_name="Wav2Lip-GAN (Low Quality - Fast)",
        enable_audio_enhancement=False, quality_preset="HIGH"))

    assert captured["enable_lipsync"] is True
    assert captured["lipsync_model_name"] == "LivePortrait (High Quality - Slow)"
    assert captured["enable_audio_enhancement"] is True


def test_process_video_without_a_preset_leaves_arguments_alone(monkeypatch):
    from src.core.video_translator import VideoTranslator

    captured = {}

    def fake_pipeline(self, **kwargs):
        captured.update(kwargs)
        return iter(())

    monkeypatch.setattr(VideoTranslator, "_process_pipeline", fake_pipeline)
    monkeypatch.setattr(VideoTranslator, "_run_config", staticmethod(lambda lv: {}))
    monkeypatch.setattr("src.core.video_translator.config.validate_path",
                        lambda p, **k: Path(p))

    vt = VideoTranslator()
    vt.job = None
    vt.manifest = None

    list(vt.process_video("m.mp4", enable_lipsync=True,
                          lipsync_model_name="Wav2Lip-GAN (Low Quality - Fast)",
                          enable_audio_enhancement=True))

    assert captured["enable_lipsync"] is True
    assert captured["lipsync_model_name"] == "Wav2Lip-GAN (Low Quality - Fast)"
    assert captured["enable_audio_enhancement"] is True
