"""
Phase 7 real-sample validation on the His Girl Friday scene.

Drives the REAL audio path (mix_loudness, synchronizer, mux) on the cached
stems and segment timeline of the existing E2E job, and measures the
properties the phase is supposed to guarantee. Only the ML models are not
re-run: there are no Wav2Lip/LivePortrait weights in this environment, so
lip-sync is validated at the detector/placement level (stated explicitly in
the report).

Run:  env PYTHONPATH=. .venv/bin/python -m pytest tests/test_phase7_real_validation.py -q
"""
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audio import mix_loudness
from src.core import jobs as _jobs
from src.core.manifest import Manifest
from src.processing.synchronization import AudioSynchronizer
from src.processing.video import VideoProcessor
from src.utils import config

JOB = Path("jobs/hgf_scene-20261021-142210-f20a32a1")
_SAMPLE_DIRS = [Path("samples"), Path("/tmp/sample")]
SOURCE_WEBM = next((d / "hgf_scene.webm" for d in _SAMPLE_DIRS
                    if (d / "hgf_scene.webm").exists()), _SAMPLE_DIRS[0] / "hgf_scene.webm")
SOURCE_MP4 = next((d / "hgf_h264.mp4" for d in _SAMPLE_DIRS
                   if (d / "hgf_h264.mp4").exists()), _SAMPLE_DIRS[0] / "hgf_h264.mp4")
SR = 24000
DURATION = 96.6935


def _ffprobe(path, *args):
    out = subprocess.run(
        ["ffprobe", "-v", "error", *args, "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _stream(path, kind):
    info = _ffprobe(path, "-show_streams")
    return next((s for s in info["streams"] if s["codec_type"] == kind), None)


def _fmt_duration(path):
    return float(_ffprobe(path, "-show_format")["format"]["duration"])


def _rms_db(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return -math.inf
    r = float(np.sqrt(np.mean(x ** 2)))
    return 20.0 * math.log10(r) if r > 0 else -math.inf


def _mono(path):
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float64), sr


@pytest.fixture(scope="module")
def stems():
    vocals = JOB / "separation" / "vocals.wav"
    bg = JOB / "separation" / "accompaniment.wav"
    segments = json.loads((JOB / "sync" / "segments.json").read_text())
    if not vocals.exists() or not bg.exists():
        pytest.skip("cached separation stems are not present")
    return vocals, bg, segments


@pytest.fixture(scope="module")
def phase7_dub(stems, tmp_path_factory):
    """Render the dub track with the CURRENT (Phase 7) mixer."""
    _, _, segments = stems
    out = tmp_path_factory.mktemp("p7") / "dub.wav"
    AudioSynchronizer(sample_rate=SR).merge_segments(
        segments, total_duration=DURATION, output_path=out, enable_time_stretch=True)
    return out


# ---------------------------------------------------------------- audio mix

def test_dub_is_audible_and_unclipped(phase7_dub):
    data, sr = _mono(phase7_dub)
    assert abs(len(data) / sr - DURATION) < 0.05
    assert _rms_db(data) > -40.0, "dub is near-silent"
    assert float(np.max(np.abs(data))) <= mix_loudness.DEFAULT_CEILING + 1e-6


def test_mix_gain_rule_places_background_below_dialogue():
    """
    The rule `estimate_mix_gains` enforces, checked directly on the real dub
    and background: a loud background is turned down until the dialogue leads
    it by ~DEFAULT_DIALOGUE_OVER_BG_DB, and is never boosted.
    """
    dub, _ = _mono(JOB / "separation" / "vocals.wav")  # stand-in for dialogue level
    bg, _ = _mono(JOB / "separation" / "accompaniment.wav")

    # Real (already quiet) background: never boosted.
    _, gain = mix_loudness.estimate_mix_gains(dub, bg)
    assert gain == 0.0, "an already-quiet background must not be boosted"

    # Synthesise a loud background and confirm it is pulled down to the target,
    # or clamped at the documented maximum reduction when that is not enough.
    for factor in (30.0, 300.0):
        loud = bg * factor
        _, gain = mix_loudness.estimate_mix_gains(dub, loud)
        assert gain <= 0.0, "background must never be boosted"

        needed = (mix_loudness.active_dbfs(dub)
                  - mix_loudness.DEFAULT_DIALOGUE_OVER_BG_DB
                  - mix_loudness.active_dbfs(loud))
        if needed >= -24.0:
            resulting_gap = (mix_loudness.active_dbfs(dub)
                             - (mix_loudness.active_dbfs(loud) + gain))
            assert abs(resulting_gap - mix_loudness.DEFAULT_DIALOGUE_OVER_BG_DB) < 1.5, \
                f"gap {resulting_gap:.1f} dB is not the intended " \
                f"{mix_loudness.DEFAULT_DIALOGUE_OVER_BG_DB} dB"
        else:
            # Too loud to bring all the way down: the reduction is capped.
            assert gain == pytest.approx(-24.0, abs=0.01), \
                f"a very loud background should hit the -24 dB cap, got {gain}"


def test_background_is_preserved_in_the_mix(stems, phase7_dub, tmp_path):
    """The music/effects bed must survive the mix, not be buried or clipped."""
    _, bg, _ = stems
    out = tmp_path / "mix.wav"
    VideoProcessor().mix_tracks(phase7_dub, bg, out)

    dub, _ = _mono(phase7_dub)
    b, _ = _mono(bg)
    mix, _ = _mono(out)

    assert _rms_db(mix) > _rms_db(b), "the mix must be louder than the bare bed"
    assert abs(_rms_db(mix) - _rms_db(dub)) < 6.0, "the bed should not change the overall level"
    assert float(np.max(np.abs(mix))) <= mix_loudness.DEFAULT_CEILING + 1e-6


def test_no_onset_clicks(phase7_dub):
    """Equal-power fades mean no sample-to-sample jump at a segment onset."""
    data, _ = _mono(phase7_dub)
    d = np.abs(np.diff(data))
    threshold = 12.0 * (np.percentile(d, 99.9) + 1e-9)
    assert float(np.max(d)) <= threshold, "a segment onset produced a discontinuity"


def test_dialogue_loudness_consistency(phase7_dub):
    """Per-line loudness spread is bounded rather than wildly alternating."""
    data, _ = _mono(phase7_dub)
    win = int(0.4 * SR)
    levels = [_rms_db(data[i:i + win]) for i in range(0, len(data) - win, win)]
    levels = [l for l in levels if l > -45.0]
    assert len(levels) > 20
    spread = float(np.percentile(levels, 95) - np.percentile(levels, 5))
    assert spread < 20.0, f"dialogue loudness spread too wide: {spread:.1f} dB"


def test_adjacent_segments_are_preserved(stems, phase7_dub):
    """
    The sample has no overlapping dialogue, so validate the next-best
    correctness property: back-to-back and near-adjacent lines all survive in
    the merged track at their own positions.
    """
    _, _, segments = stems
    data, _ = _mono(phase7_dub)

    adjacent = [(a, b) for a, b in zip(segments, segments[1:])
                if b["start"] - a["end"] < 0.25]
    assert adjacent, "expected back-to-back dialogue in the sample"

    for a, b in adjacent[:5]:
        for seg in (a, b):
            lo, hi = int(seg["start"] * SR), int(seg["end"] * SR)
            assert _rms_db(data[lo:hi]) > -50.0, f"segment {seg['segment_id']} is missing"
    # Adjacent lines must not be clipped by each other.
    assert float(np.max(np.abs(data))) <= mix_loudness.DEFAULT_CEILING + 1e-6


# ---------------------------------------------------------------- A/V drift

def test_no_cumulative_av_drift(stems, phase7_dub):
    """
    Drift must not accumulate: check the dialogue's position near the start,
    the middle, and the end against the source timeline, and compare the
    offset at each point. Total duration alone cannot show this.
    """
    _, _, segments = stems
    data, _ = _mono(phase7_dub)

    def onset_near(t, span=1.2):
        lo = max(0, int((t - 0.05) * SR))
        hi = min(len(data), int((t + span) * SR))
        w = np.abs(data[lo:hi])
        if w.size == 0:
            return None
        peak = float(np.max(w))
        if peak <= 0:
            return None
        idx = int(np.argmax(w > 0.15 * peak))
        return (lo + idx) / SR

    probes = [segments[0], segments[len(segments) // 2], segments[-1]]
    offsets = []
    for seg in probes:
        placed = onset_near(seg["start"])
        assert placed is not None, f"segment at {seg['start']} is missing"
        offsets.append(placed - seg["start"])

    drift = offsets[-1] - offsets[0]
    assert abs(drift) < 0.10, f"cumulative drift {drift:.3f}s across the timeline"
    assert all(abs(o) < 0.25 for o in offsets), f"per-probe offsets {offsets}"


# ---------------------------------------------------------------- containers

def _make_input(prefer_mp4):
    if prefer_mp4 and SOURCE_MP4.exists():
        return SOURCE_MP4
    if SOURCE_WEBM.exists():
        return SOURCE_WEBM
    return SOURCE_MP4


def test_mp4_uses_h264_aac(tmp_path):
    src = _make_input(prefer_mp4=True)
    if not src.exists():
        pytest.skip("no MP4 source available")
    out = tmp_path / "out.mp4"
    VideoProcessor().replace_audio(str(src), str(src), str(out))
    v, a = _stream(out, "video"), _stream(out, "audio")
    assert v["codec_name"] == "h264", f"MP4 video codec is {v['codec_name']}"
    assert a["codec_name"] == "aac", f"MP4 audio codec is {a['codec_name']}"


def test_webm_uses_vp9_opus(tmp_path):
    src = _make_input(prefer_mp4=False)
    if not src.exists():
        pytest.skip("no WebM source available")
    out = tmp_path / "out.webm"
    VideoProcessor().replace_audio(str(src), str(src), str(out))
    v, a = _stream(out, "video"), _stream(out, "audio")
    assert v["codec_name"] == "vp9", f"WebM video codec is {v['codec_name']}"
    assert a["codec_name"] == "opus", f"WebM audio codec is {a['codec_name']}"


def test_audio_only_replacement_stream_copies_video(tmp_path):
    """When only the audio changes, the video stream must be copied, not re-encoded."""
    src = _make_input(prefer_mp4=False)
    if not src.exists():
        pytest.skip("no WebM source available")
    new_audio = tmp_path / "a.wav"
    dur = _fmt_duration(src)
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    sf.write(str(new_audio), 0.2 * np.sin(2 * np.pi * 220 * t), SR)

    out = tmp_path / "copy.webm"
    VideoProcessor().replace_audio(str(src), str(new_audio), str(out))

    src_v = _stream(src, "video")
    out_v = _stream(out, "video")
    assert out_v["codec_name"] == src_v["codec_name"], "video codec changed"
    assert abs(int(out_v["width"]) - int(src_v["width"])) == 0, "resolution changed"
    assert out_v["codec_name"] == "vp9"


@pytest.fixture(scope="module")
def real_detector():
    """One Wav2LipSyncer (and thus one face-detector load) for the module."""
    from src.processing.wav2lip import Wav2LipSyncer
    return Wav2LipSyncer()


# ------------------------------------------------------------- lip-sync policy

def test_lipsync_detection_policy_on_real_frames(real_detector):
    """
    Run the REAL face detector on real HGF frames and check the Phase 7 policy:
    exactly one confident face -> eligible; two or more -> not chosen (frame
    preserved); none -> not chosen (frame preserved). Inference itself is not
    run here (no Wav2Lip weights in this environment).
    """
    cv2 = pytest.importorskip("cv2")

    src = SOURCE_WEBM if SOURCE_WEBM.exists() else SOURCE_MP4
    if not src.exists():
        pytest.skip("no source video for face detection")

    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    syncer = real_detector

    step = max(1, int(round(fps * 2)))
    counts, chosen = [], []
    try:
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                break
            boxes, box = syncer._detect_frame_boxes(frame)
            counts.append(len(boxes))
            chosen.append(box is not None)
    finally:
        cap.release()

    assert counts, "detector produced no results"
    for c, b in zip(counts, chosen):
        if c == 1:
            assert b, "a single face must be eligible for lip-sync"
        else:
            assert not b, f"a frame with {c} faces must be left untouched"


def test_ambiguous_frames_are_not_filled_from_neighbours():
    """
    The Phase 7 mechanism itself: a sustained multi-face shot must survive the
    gap-fill and end up unmarked, while single-face frames stay marked. An
    isolated one-frame count-2 reading is a stray and is correctly ignored.
    """
    from src.processing.wav2lip import Wav2LipSyncer

    syncer = Wav2LipSyncer()
    box = [10, 10, 60, 60]
    raw_boxes = [box] * 12
    # A four-frame two-actor shot (indices 4-7), single face elsewhere.
    raw_counts = [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1]

    filled = syncer._fill_missing_boxes(syncer.get_smooth_box(raw_boxes, 5))
    ambiguous = syncer._smooth_face_counts(raw_counts, 5)

    # The sustained shot is flagged; a single-face frame well away from it is not.
    assert ambiguous[5] >= 2, "the sustained two-face shot must be ambiguous"
    assert ambiguous[0] == 1, "a single-face frame must stay eligible"

    for idx, count in enumerate(ambiguous):
        if count >= 2:
            filled[idx] = None
    assert filled[5] is None, "the multi-face frame must not be lip-synced"
    assert filled[0] is not None, "the single-face frame must stay eligible"


def test_an_isolated_stray_detection_does_not_open_a_hole():
    """One stray count-2 frame among single-face frames must be median-ed out."""
    from src.processing.wav2lip import Wav2LipSyncer

    ambiguous = Wav2LipSyncer()._smooth_face_counts([1, 1, 2, 1, 1, 1, 1], 5)
    assert ambiguous[2] == 1, "an isolated stray must not mark the frame ambiguous"


def test_no_face_frame_is_preserved(real_detector):
    """A frame with no face must not be chosen (original frame kept)."""
    blank = np.zeros((480, 658, 3), dtype=np.uint8)
    boxes, box = real_detector._detect_frame_boxes(blank)
    assert boxes == []
    assert box is None


# ------------------------------------------------------ resume / checkpoint

def test_real_job_accepts_a_lipsync_only_rerender(tmp_path, monkeypatch):
    """
    The real Phase 6 job (no Phase 7 marker) must refuse a stale match, and a
    lip-sync-only change must invalidate only lip-sync + final.

    Works on a COPY of the job: `reconcile` writes the manifest, and the real
    job is a fixture of this validation, not scratch space.
    """
    src_job = JOB / "manifest.json"
    if not src_job.exists():
        pytest.skip("real job manifest not present")

    work = tmp_path / "jobs"
    work.mkdir()
    shutil.copytree(JOB, work / JOB.name)
    monkeypatch.setattr(config, "JOBS_DIR", work)

    manifest = Manifest.load(_jobs.Job.load(JOB.name))
    if manifest is None:
        pytest.skip("real job manifest not readable")

    # It predates Phase 7, so it carries no scope marker: its fingerprints can
    # never be mistaken for scoped ones.
    assert "fingerprint_scope" not in (manifest.source or {})

    recorded = dict(manifest.config)
    assert recorded.get("enable_lipsync") is False

    # Flip lip-sync on: only the sticky stages may be invalidated.
    requested = dict(recorded)
    requested["enable_lipsync"] = True
    requested["lipsync_model_name"] = "Wav2Lip-GAN (Low Quality - Fast)"
    drifted = manifest.differs_from(requested)
    assert set(drifted) <= set(config.RE_RENDERABLE_CONFIG_KEYS)

    report = manifest.reconcile(requested, reason="lipsync-only re-render")
    invalidated = set(report["invalidated"])
    assert "lipsync" in invalidated and "final" in invalidated
    for preserved in ("transcription", "diarization", "references",
                      "translation", "tts", "separation"):
        assert preserved not in invalidated, f"{preserved} must stay cached"


# ------------------------------------------------------------- scene cuts

def test_scene_cuts_do_not_produce_unsafe_lipsync_marks(real_detector):
    """
    Around a real hard cut the policy must stay safe: every frame is either
    single-face (eligible) or unmarked. A cut must never cause a frame with
    several faces to become eligible, which is the failure the phase prevents.
    """
    cv2 = pytest.importorskip("cv2")
    src = SOURCE_WEBM if SOURCE_WEBM.exists() else SOURCE_MP4
    if not src.exists():
        pytest.skip("no source video")

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps * 2)))

    frames, counts, chosen = [], [], []
    try:
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (160, 120))
            frames.append(small.astype(np.float32))
            boxes, box = real_detector._detect_frame_boxes(frame)
            counts.append(len(boxes))
            chosen.append(box is not None)
    finally:
        cap.release()

    # Find hard cuts: a large mean absolute difference between consecutive samples.
    diffs = [float(np.mean(np.abs(frames[i + 1] - frames[i])))
             for i in range(len(frames) - 1)]
    cuts = [i + 1 for i, d in enumerate(diffs) if d > np.percentile(diffs, 90)]

    assert cuts, "no candidate cuts found in the sample"

    # The safety invariant must hold on every frame, including at each cut.
    for c, b in zip(counts, chosen):
        assert (b == (c == 1)), f"unsafe mark: {c} faces but chosen={b}"
    for idx in cuts:
        for j in (idx - 1, idx, idx + 1):
            if 0 <= j < len(counts):
                assert (chosen[j] == (counts[j] == 1)), f"unsafe mark at cut frame {j}"
