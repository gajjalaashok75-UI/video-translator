"""
Tests for VideoProcessor.replace_audio container/codec compatibility.

Regression: the mux used to hardcode h264/aac regardless of the output
extension. That is invalid inside a WebM container, so a .webm source failed
with "Nothing was written into output file". These tests exercise real ffmpeg
on real files; nothing is mocked.
"""

import json
import subprocess

import pytest

from src.processing.video import VideoProcessor


def _run(args):
    subprocess.run(args, check=True, stderr=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL)


def _streams(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return {s["codec_type"]: s["codec_name"] for s in json.loads(out)["streams"]}


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    """One silent video per container plus a WAV to mux in."""
    d = tmp_path_factory.mktemp("replace_audio")
    mp4, webm, wav = d / "src.mp4", d / "src.webm", d / "dub.wav"
    common = ["-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=15",
              "-f", "lavfi", "-i", "sine=frequency=440:duration=2"]
    try:
        _run(["ffmpeg", "-y", *common, "-c:v", "libx264", "-c:a", "aac", str(mp4)])
        _run(["ffmpeg", "-y", *common, "-c:v", "libvpx-vp9", "-c:a", "libopus", str(webm)])
        _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=880:duration=2",
              "-c:a", "pcm_s16le", str(wav)])
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("FFmpeg not available or failed to create test media")
    return mp4, webm, wav


@pytest.mark.requires_ffmpeg
def test_mp4_output_keeps_mp4_codecs(media, tmp_path):
    mp4, _, wav = media
    out = tmp_path / "out.mp4"
    result = VideoProcessor().replace_audio(str(mp4), str(wav), str(out))
    assert result == str(out) and out.exists() and out.stat().st_size > 0
    assert _streams(out) == {"video": "h264", "audio": "aac"}


@pytest.mark.requires_ffmpeg
def test_webm_output_uses_webm_codecs(media, tmp_path):
    """The regression: h264/aac cannot be muxed into WebM."""
    _, webm, wav = media
    out = tmp_path / "out.webm"
    result = VideoProcessor().replace_audio(str(webm), str(wav), str(out))
    assert result == str(out) and out.exists() and out.stat().st_size > 0
    assert _streams(out) == {"video": "vp9", "audio": "opus"}


@pytest.mark.requires_ffmpeg
def test_mux_failure_raises_instead_of_returning_none(media, tmp_path):
    """A None return would be recorded as a completed final artifact."""
    _, _, wav = media
    with pytest.raises(Exception):
        VideoProcessor().replace_audio(
            str(tmp_path / "missing.mp4"), str(wav), str(tmp_path / "out.mp4"))
