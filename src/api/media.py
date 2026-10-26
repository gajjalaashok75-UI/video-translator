"""
Read-only media probing for uploaded files.

`src/core/jobs.validate_video` already answers "is this a readable video?", and
that is the check the pipeline itself trusts, so it is reused here rather than
reimplemented. What it does not provide is the *detail* an upload response
should carry (duration, resolution, codecs, container), which needs one ffprobe
call.

Nothing in this module writes to the media, mutates it, or changes pipeline
behaviour; it runs ffprobe and reads the JSON it prints. When ffprobe is
unavailable the detail is empty and only the existing `validate_video` result is
reported, so the endpoint still works on a machine without FFmpeg.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from src.core import jobs as joblib

logger = logging.getLogger(__name__)

# ffprobe on a large file can take a moment on a cold cache; the pipeline's own
# validation uses 30s, so this matches it rather than inventing a new bound.
_PROBE_TIMEOUT = 30


def ffprobe_path() -> Optional[str]:
    """The ffprobe binary, or None when it is not installed."""
    return shutil.which("ffprobe")


def _run_ffprobe(args: list[str]) -> Optional[dict]:
    probe = ffprobe_path()
    if not probe:
        return None
    try:
        out = subprocess.run(
            [probe, "-v", "error", *args, "-of", "json"],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("ffprobe failed: %s", exc)
        return None
    if out.returncode != 0:
        logger.info("ffprobe returned %s: %s", out.returncode,
                    (out.stderr or "").strip()[:300])
        return None
    try:
        return json.loads(out.stdout or "{}")
    except ValueError:
        return None


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe_media(path: Path) -> dict:
    """
    A best-effort summary of an uploaded file's container and streams.

    Returns a dict with any of `duration`, `width`, `height`, `video_codec`,
    `audio_codec`, `container`, `has_video`, `has_audio`. Missing values are
    simply absent, so a caller can distinguish "not probed" from "zero".

    The result is informational: the authoritative check on whether the file is
    usable is `jobs.validate_video`, which the pipeline also uses.
    """
    path = Path(path)
    info: dict = {}

    data = _run_ffprobe([
        "-show_entries",
        "format=format_name,duration",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,duration",
        str(path),
    ])
    if not data:
        return info

    fmt = data.get("format") or {}
    container = fmt.get("format_name")
    if container:
        # ffprobe reports a comma-separated list of candidate formats; the first
        # is the most specific.
        info["container"] = str(container).split(",")[0]
    duration = _to_float(fmt.get("duration"))

    video_streams = []
    audio_streams = []
    for stream in data.get("streams") or []:
        kind = stream.get("codec_type")
        if kind == "video":
            video_streams.append(stream)
        elif kind == "audio":
            audio_streams.append(stream)

    if video_streams:
        first = video_streams[0]
        info["has_video"] = True
        info["video_codec"] = first.get("codec_name")
        info["width"] = int(first.get("width") or 0) or None
        info["height"] = int(first.get("height") or 0) or None
        # Container duration is preferred; a stream duration is the fallback for
        # formats that do not record it at the format level.
        if duration is None:
            duration = _to_float(first.get("duration"))
    else:
        info["has_video"] = False

    if audio_streams:
        info["has_audio"] = True
        info["audio_codec"] = audio_streams[0].get("codec_name")
        if duration is None:
            duration = _to_float(audio_streams[0].get("duration"))
    else:
        info["has_audio"] = False

    if duration is not None:
        info["duration"] = round(duration, 3)

    # Drop the None placeholders so the payload only carries known values.
    return {k: v for k, v in info.items() if v is not None}


def validate_upload(path: Path) -> tuple[bool, str]:
    """
    Whether an uploaded file is a usable video, and why not when it is not.

    Delegates to the pipeline's own `jobs.validate_video` (which prefers ffprobe
    and falls back to a size check) and then requires that a video stream is
    actually present, because a bare audio file or a text file renamed to `.mp4`
    passes the duration check but cannot be dubbed.
    """
    path = Path(path)
    if not path.is_file():
        return False, "The uploaded file was not written to disk."
    if not joblib.validate_video(path):
        return False, ("The uploaded file is not a readable video, or is "
                       "truncated or empty.")

    info = probe_media(path)
    # When ffprobe is unavailable there is no stream information; the
    # validate_video result is then the only evidence, and it is what the
    # pipeline relies on, so the upload is accepted.
    if "has_video" in info and not info["has_video"]:
        return False, "The uploaded file contains no video stream."
    if info.get("duration") is not None and info["duration"] <= 0:
        return False, "The uploaded file has zero duration."
    return True, ""
