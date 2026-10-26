"""
Reading a job's durable log.

The log is `jobs/<id>/logs/job.log`, appended to by the pipeline through
`Job.log` — a plain text file, one line per event, which is the format an
operator reads with `tail -f`. It is *not* the SSE buffer: that is a bounded
ring of recent events in memory, it dies with the process, and it is not a log
store. A client that wants the log after a restart needs the file, which is what
this module reads.

Two constraints shape the reader.

**Bounded.** A long render's log is small (a line per stage), but nothing
enforces that, and an API that returns whatever is on disk is one bad render away
from streaming megabytes to a browser. The default is a tail; a client asks for
more explicitly and gets a `has_more` flag telling it whether it is seeing
everything.

**Filtered.** The lines are written for an operator and contain absolute paths.
They go through `redact` before leaving, so the response is safe to hand to a
browser while the file on disk keeps its full detail.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.api import redact
from src.core import jobs as joblib

logger = logging.getLogger(__name__)

# A line longer than this is truncated before it is returned. A single
# pathological line (a dumped array, a stack trace on one line) should not be
# able to make one response enormous.
MAX_LINE_CHARS = 4000

# The most lines one request may return, whatever the client asks for. Chosen so
# a full log of a normal render arrives in one response, while a runaway log is
# still capped.
MAX_TAIL = 5000

DEFAULT_TAIL = 200


def log_path(job_id: str) -> Optional[Path]:
    """The job's log file, if the job exists and has one."""
    if not joblib.is_valid_job_id(job_id):
        return None
    try:
        job = joblib.Job.load(job_id)
    except joblib.JobError:
        return None
    path = job.paths.logs
    if not path.is_file():
        return None
    return path


def _read_tail(path: Path, tail: int) -> tuple[list[str], bool]:
    """
    The last `tail` lines, and whether earlier lines were left behind.

    Read from the end in chunks rather than with `readlines()`: the point of a
    bounded API is not to load the whole file to serve part of it, and a log that
    grew large is exactly the case where that matters.
    """
    if tail <= 0:
        return [], path.stat().st_size > 0

    chunk_size = 8192
    collected: list[bytes] = []
    newlines = 0
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        while position > 0 and newlines <= tail:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            block = handle.read(read_size)
            collected.append(block)
            newlines += block.count(b"\n")

    data = b"".join(reversed(collected))
    # `splitlines` handles a trailing newline without inventing an empty last
    # line, and a file written on Windows without a final newline.
    lines = data.decode("utf-8", errors="replace").splitlines()
    has_more = len(lines) > tail
    if has_more:
        lines = lines[-tail:]
    return lines, has_more


def read(job_id: str, tail: int = DEFAULT_TAIL,
         offset: Optional[int] = None,
         limit: Optional[int] = None) -> Optional[dict]:
    """
    A job's log, filtered and bounded, or None when there is no log.

    Two ways to bound it, and they compose:

      * `tail=N` — the last N lines. This is what a "show me the end of the log"
        button wants, and the default.
      * `offset` / `limit` — a window from the start, for a client that pages
        through the log. `offset` counts lines from the beginning.

    The response always reports `total_lines`, so a client paging through knows
    when to stop, and `has_more` for the tail case.
    """
    path = log_path(job_id)
    if path is None:
        return None

    # The cap is applied to the *tail* request; a paged read is bounded by
    # `limit` instead.
    tail = max(1, min(int(tail), MAX_TAIL)) if tail else DEFAULT_TAIL
    try:
        lines, has_more = _read_tail(path, tail)
        total = _count_lines(path)
    except OSError as exc:
        logger.warning("Could not read log for job %s: %s", job_id, exc)
        return None

    filtered = [redact.redact_text(line)[:MAX_LINE_CHARS] for line in lines]

    if offset is not None:
        start = max(0, int(offset))
        end = None if limit is None else start + max(0, int(limit))
        filtered = filtered[start:end]
        has_more = end is not None and end < len(lines)

    return {
        "job_id": job_id,
        "lines": filtered,
        "returned_lines": len(filtered),
        "total_lines": total,
        "has_more": has_more,
        "truncated_lines": any(len(redact.redact_text(line)) > MAX_LINE_CHARS
                               for line in lines),
        "redacted": True,
    }


def _count_lines(path: Path) -> int:
    """
    The number of lines in the file.

    Counted by streaming in binary rather than by reading lines, so a large log
    costs a scan and not a list of strings in memory.
    """
    count = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            count += block.count(b"\n")
    # A file that does not end in a newline has one more line than it has
    # newlines; `Job.log` always writes a trailing newline, so this is a guard
    # for a hand-edited file rather than the normal path.
    try:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                count += 1
    except OSError:
        pass
    return count
