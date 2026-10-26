import datetime
from pathlib import Path


def _clamp(seconds: float) -> float:
    """Timestamps are never negative; a negative one would format as garbage."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return 0.0
    return value if value > 0.0 else 0.0


def format_timestamp(seconds: float) -> str:
    """
    Converts seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds (float)

    Returns:
        String formatted as HH:MM:SS,mmm
    """
    seconds = _clamp(seconds)
    # Create a timedelta object
    td = datetime.timedelta(seconds=seconds)

    # Get total seconds
    total_seconds = int(td.total_seconds())

    # Calculate hours, minutes, seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    # Calculate milliseconds
    milliseconds = int(round((seconds - total_seconds) * 1000))
    if milliseconds == 1000:
        # Rounding can carry into the next second; without this a cue at
        # 00:00:01.9996 formats as 00:00:01,1000, which no player accepts.
        total_seconds += 1
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        milliseconds = 0

    # Format the string
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def generate_srt(segments: list, output_path: str | Path) -> str:
    """
    Creates an SRT file from segments with 'start', 'end', and 'translated_text' keys.

    Args:
        segments: List of segment dictionaries
        output_path: Path to write the SRT file

    Returns:
        Path to the generated SRT file
    """
    srt_content = []

    for i, seg in enumerate(segments, start=1):
        start_time = format_timestamp(seg.get('start', 0.0))
        end_time = format_timestamp(seg.get('end', 0.0))
        text = (seg.get('translated_text') or '').strip()

        if not text:
            continue

        srt_content.append(f"{i}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text)
        srt_content.append("")  # Empty line after each block

    # Join with newlines
    content = "\n".join(srt_content)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return output_path


def format_vtt_timestamp(seconds: float) -> str:
    """Converts seconds to WebVTT timestamp format (HH:MM:SS.mmm)."""
    return format_timestamp(seconds).replace(",", ".")


def generate_vtt(segments: list, output_path: str | Path) -> str:
    """
    Creates a WebVTT file from segments with 'start', 'end' and
    'translated_text' keys.

    WebVTT is the format a browser's native ``<track>`` element renders without a
    shim, which is why it is offered alongside SRT: SRT is what an editor
    imports, VTT is what a page plays. Both are generated from the same
    segments, so they cannot disagree.

    Args:
        segments: List of segment dictionaries.
        output_path: Path to write the VTT file.

    Returns:
        Path to the generated VTT file.
    """
    lines = ["WEBVTT", ""]

    for seg in segments:
        start = _clamp(seg.get('start', 0.0))
        end = _clamp(seg.get('end', 0.0))
        text = (seg.get('translated_text') or '').strip()
        if not text:
            continue
        if end <= start:
            # A zero-length cue is not renderable; give it the minimum a player
            # will display rather than dropping a line the user expects to see.
            end = start + 0.5
        lines.append(f"{format_vtt_timestamp(start)} --> {format_vtt_timestamp(end)}")
        lines.append(text)
        lines.append("")

    content = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return output_path
