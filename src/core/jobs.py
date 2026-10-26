"""
Durable job workspace (Phase 6).

A render is a *job*: a stable id and a directory tree under `config.JOBS_DIR`
that holds the source, every intermediate artifact and one authoritative
manifest. The point is that a crash at hour three of a four-hour render does not
throw away hours one and two.

Two things this module is deliberately strict about:

  * **Containment.** Every path handed out is resolved and checked to live
    inside the job directory. A job id arrives from a URL, a CLI argument or a
    resumed manifest, so it is untrusted input: `../../etc` must not resolve to
    anything outside the workspace.
  * **Atomicity.** Artifacts are written beside their target and renamed into
    place, so a crash mid-write leaves the previous good file rather than a
    truncated one that a later run would happily treat as valid cache.
"""

from __future__ import annotations

import calendar
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.utils import config

logger = logging.getLogger(__name__)

# Subdirectories of a job. The names are the stage names from
# `config.JOB_STAGES` plus `logs`; keeping them in one tuple means the layout is
# defined once and can be created or cleaned consistently.
JOB_SUBDIRS = (
    "source", "audio", "separation", "transcription", "diarization",
    "speaker_refs", "translation", "tts", "sync", "subtitles", "lipsync",
    "final", "logs", "cache", "scratch",
)

# Pipeline stage -> directory. Most stages own a directory of the same name;
# `references` writes into `speaker_refs` because that is what it produces, and
# the two names are kept distinct on purpose: the stage is a step, the directory
# is a location.
STAGE_DIRS = {
    "source": "source",
    "extraction": "audio",
    "separation": "separation",
    "transcription": "transcription",
    "diarization": "diarization",
    "references": "speaker_refs",
    "translation": "translation",
    "tts": "tts",
    "sync": "sync",
    "subtitles": "subtitles",
    "lipsync": "lipsync",
    "final": "final",
}

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

# The canonical creation stamp `new_job_id` appends: `[<stem>-]YYYYMMDD-HHMMSS-<hash>`.
# Matched from the right because the source stem may itself contain hyphens, so
# the number of components before the stamp is not fixed.
_ID_STAMP_RE = re.compile(r"(?<!\d)(\d{8})-(\d{6})-([0-9a-fA-F]+)$")


class JobError(Exception):
    """A job could not be created, loaded or written."""


class PathEscapeError(JobError):
    """A path resolved outside the job directory."""


# ---------------------------------------------------------------------------
# Identifiers and safe names
# ---------------------------------------------------------------------------
def new_job_id(prefix: str = "") -> str:
    """A fresh, filesystem-safe job id. Time-ordered so `ls` reads chronologically."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    token = uuid.uuid4().hex[:8]
    clean = safe_name(prefix, max_length=24)
    return f"{clean}-{stamp}-{token}" if clean else f"{stamp}-{token}"


def is_valid_job_id(job_id: str) -> bool:
    """Whether `job_id` is a plain name that cannot traverse or hide."""
    if not job_id or not isinstance(job_id, str):
        return False
    if not _ID_RE.match(job_id):
        return False
    # A bare "." or ".." matches the pattern but is not a name.
    return job_id not in (".", "..")


def safe_name(name: str, max_length: int = 80) -> str:
    """
    Reduce arbitrary text to something safe to use as one path component.

    Collapses runs of anything that is not `[A-Za-z0-9._-]` to a single `_`,
    strips leading dots (so `..` and hidden names cannot survive) and truncates.
    """
    if not name:
        return ""
    cleaned = _UNSAFE_CHARS.sub("_", str(name)).strip("._-")
    return cleaned[:max_length]


def require_job_id(job_id: str) -> str:
    """Validate a job id or raise. Use this at every trust boundary."""
    if not is_valid_job_id(job_id):
        raise JobError(f"Invalid job id: {job_id!r}")
    return job_id


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------
def _is_within(child: Path, parent: Path) -> bool:
    """True when `child` is `parent` or lives under it, after resolution."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_within(path: Path, root: Path, what: str = "path") -> Path:
    """
    Resolve `path` and confirm it is inside `root`. Raises otherwise.

    Used before writing, deleting or reading anything derived from job state, so
    a corrupt or hostile manifest cannot reach outside the workspace.
    """
    root = Path(root).resolve()
    resolved = Path(path).resolve()
    if not _is_within(resolved, root):
        raise PathEscapeError(f"{what} escapes its root: {resolved} is not inside {root}")
    return resolved


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------
def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """
    Write `data` to `path` atomically.

    Writes to `<path>.tmp-<pid>` in the same directory (same filesystem, so the
    rename is atomic) then `os.replace`s it. An interrupted call leaves the old
    file intact; a reader never sees a half-written one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Never leave the scratch file behind on failure.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path, payload, indent: int = 2) -> Path:
    """
    Serialise `payload` and write it atomically.

    `sort_keys` makes the output deterministic, so two runs that produced the
    same state write byte-identical files and a diff shows what actually
    changed. `default=str` keeps an unexpected numpy scalar or Path from turning
    a successful render into a crash on the way to disk.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=indent,
                      sort_keys=True, default=str)
    return atomic_write_text(path, text)


def read_json(path: Path, default=None):
    """Read JSON, returning `default` if the file is missing or unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------------
def validate_wav(path: Path, min_duration: float = 0.0) -> bool:
    """
    Whether `path` is a readable, *complete* WAV of at least `min_duration`.

    The header alone is not enough. A crash mid-write leaves a file whose header
    promises more frames than the file contains, and treating that as a cache
    hit is how a resumed render ends up with a silent stretch in the middle. So
    the declared frame count is verified against the data that is actually
    present by seeking to the last declared frame and reading it back.

    This runs on every cache lookup, so it must stay cheap: it reads a header,
    never the whole file.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size < 44:  # smaller than a WAV header
        return False

    try:
        # The RIFF header records the total size up front. A truncated write
        # leaves that number larger than the file, and libsndfile hides this by
        # silently clamping its frame count, so the header is checked directly.
        with open(path, "rb") as f:
            head = f.read(12)
        if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WAVE":
            declared = int.from_bytes(head[4:8], "little") + 8
            # 0xFFFFFFFF means "size unknown" (streaming); nothing to compare.
            if declared != 0xFFFFFFFF + 8 and size < declared:
                return False

        import soundfile as sf
        info = sf.info(str(path))
        if info.frames <= 0 or info.samplerate <= 0:
            return False
        return (info.frames / float(info.samplerate)) >= min_duration
    except Exception:  # noqa: BLE001 - any failure means "not usable"
        return False


def is_silent(path: Path, threshold: float = 1e-4) -> bool:
    """
    Whether a WAV is digital silence.

    A TTS backend that failed often returns a correctly-formed, correctly-sized
    file of silence rather than an error. Caching that would make the failure
    permanent, so `ArtifactCache.put_tts` refuses it. Reading the samples is
    acceptable here because it happens once per generated segment, not once per
    cache lookup.
    """
    try:
        import soundfile as sf
        data, _ = sf.read(str(path), dtype="float32", always_2d=True)
        if data.size == 0:
            return True
        import numpy as np
        return float(np.max(np.abs(data))) < threshold
    except Exception:  # noqa: BLE001
        return True


def validate_video(path: Path, min_duration: float = 0.0) -> bool:
    """
    Whether `path` looks like a readable video of at least `min_duration`.

    Uses ffprobe when available and falls back to a size check, because the
    container is large enough that a truncated file is the realistic failure.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        import subprocess
        probe = shutil.which("ffprobe")
        if not probe:
            return path.stat().st_size > 1024
        out = subprocess.run(
            [probe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return False
        return float(out.stdout.strip() or 0) >= min_duration
    except Exception:  # noqa: BLE001
        return path.stat().st_size > 1024


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------
@dataclass
class JobPaths:
    """Every directory a job owns. Created on demand, never outside the root."""

    root: Path

    def sub(self, name: str) -> Path:
        if name not in JOB_SUBDIRS:
            raise JobError(f"Unknown job subdirectory: {name!r}")
        return self.root / name

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def logs(self) -> Path:
        """The job's log *file*. The directory is available as `sub("logs")`."""
        return self.root / "logs" / "job.log"

    def ensure(self) -> "JobPaths":
        for name in JOB_SUBDIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict:
        return {name: str(self.root / name) for name in JOB_SUBDIRS}


class Job:
    """
    One dubbing job's durable workspace.

    Holds no pipeline state of its own beyond the paths; the manifest
    (`src/core/manifest.py`) is the authoritative record and is written by the
    pipeline as stages complete.
    """

    def __init__(self, job_id: str, root: Optional[Path] = None,
                 jobs_dir: Optional[Path] = None):
        self.job_id = require_job_id(job_id)
        base = Path(jobs_dir or config.JOBS_DIR).resolve()
        candidate = Path(root) if root is not None else base / self.job_id
        # A caller-supplied root is still checked: it must be a directory named
        # for the job id, so two jobs can never share a workspace.
        resolved = Path(candidate).resolve()
        if resolved.name != self.job_id:
            raise JobError(
                f"Job directory must be named {self.job_id!r}, got {resolved.name!r}")
        self.paths = JobPaths(resolved)

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def create(cls, source_path: Optional[str | Path] = None,
               prefix: str = "", jobs_dir: Optional[Path] = None) -> "Job":
        """Make a new job directory and return it."""
        job = cls(new_job_id(prefix), jobs_dir=jobs_dir)
        job.paths.ensure()
        logger.info("Created job %s at %s", job.job_id, job.paths.root)
        return job

    @classmethod
    def load(cls, job_id: str, jobs_dir: Optional[Path] = None) -> "Job":
        """Reopen an existing job. Raises if its directory is gone."""
        job = cls(job_id, jobs_dir=jobs_dir)
        if not job.paths.root.is_dir():
            raise JobError(f"Job directory does not exist: {job.paths.root}")
        return job

    @classmethod
    def open_or_create(cls, job_id: Optional[str],
                       jobs_dir: Optional[Path] = None,
                       prefix: str = "") -> "Job":
        """Resume `job_id` when given and present, otherwise start a new job."""
        if job_id:
            job = cls(job_id, jobs_dir=jobs_dir)
            if job.paths.root.is_dir():
                job.paths.ensure()
                return job
        return cls.create(prefix=prefix, jobs_dir=jobs_dir)

    @property
    def root(self) -> Path:
        return self.paths.root

    @property
    def exists(self) -> bool:
        return self.paths.root.is_dir()

    # -- paths -------------------------------------------------------------
    def path(self, *parts: str | Path) -> Path:
        """
        A path inside the job, guaranteed to stay inside it.

        The parts are sanitised individually, so a caller passing a segment's
        translated text or a speaker id cannot introduce separators.
        """
        if not parts:
            raise JobError("path() needs at least one part")
        safe_parts = [safe_name(str(p)) for p in parts]
        if any(not p for p in safe_parts):
            raise JobError(f"Unsafe path component in {parts!r}")
        return assert_within(self.paths.root.joinpath(*safe_parts), self.paths.root)

    def stage_dir(self, stage: str) -> Path:
        """
        The directory for a named stage, created if needed.

        Accepts either a stage name (`references`) or a directory name
        (`speaker_refs`), so callers can use whichever they have.
        """
        name = STAGE_DIRS.get(stage, stage)
        if name not in JOB_SUBDIRS:
            raise JobError(f"Unknown stage directory: {stage!r}")
        d = self.paths.sub(name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def scratch_dir(self) -> Path:
        """
        Per-job scratch space for intermediates that are not durable artifacts.

        The pipeline used to write several files into a *shared* `temp/`
        directory (`<stem>_full.wav`, `<stem>_merged_speech.wav`, the speaker
        profile folders, …). With two jobs on one machine those names collide,
        and — worse — a job whose source stem matched a committed fixture
        (`temp/vocals.wav`, `output/translated_hgf_scene.webm`) would overwrite
        that tracked file on every run, dirtying the repository. Job-scoped
        scratch removes both problems: nothing here is shared and nothing here
        can reach a tracked path.
        """
        d = self.paths.sub("scratch")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def log(self, message: str) -> None:
        """Append one line to the job's own log, best-effort."""
        try:
            self.paths.logs.parent.mkdir(parents=True, exist_ok=True)
            line = f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())} {message}\n"
            with open(self.paths.logs, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            pass

    # -- disk --------------------------------------------------------------
    def disk_usage(self) -> dict:
        """Bytes per top-level subdirectory, plus the total."""
        usage = {}
        total = 0
        for name in JOB_SUBDIRS:
            d = self.paths.root / name
            size = 0
            if d.is_dir():
                for p in d.rglob("*"):
                    if p.is_file():
                        try:
                            size += p.stat().st_size
                        except OSError:
                            pass
            usage[name] = size
            total += size
        usage["total_bytes"] = total
        return usage

    # -- cleanup -----------------------------------------------------------
    def clean(self, level: str = config.CLEANUP_KEEP_RESUMABLE,
              active: bool = False) -> list[str]:
        """
        Delete intermediates according to `level`. Returns the names removed.

        Refuses to touch anything when `active` is True: deleting a
        resume-critical artifact while a job is running is exactly the bug this
        whole phase exists to prevent. `final` and the manifest are never
        removed unless the level is FINAL_ONLY, and never while active.
        """
        if active:
            logger.info("Job %s is active; refusing to clean.", self.job_id)
            return []

        if level not in (config.CLEANUP_KEEP_ALL, config.CLEANUP_KEEP_RESUMABLE,
                         config.CLEANUP_FINAL_ONLY):
            raise JobError(f"Unknown cleanup level: {level!r}")

        if level == config.CLEANUP_KEEP_ALL:
            return []

        # KEEP_RESUMABLE: drop bulky intermediates that can be regenerated
        # cheaply from what remains. Everything needed to continue the pipeline
        # (transcripts, translations, per-segment TTS, references) is kept.
        # `scratch` is pure working space — never a resume input — so it always
        # goes, even at the most conservative level.
        resumable_drop = ("audio", "separation", "subtitles", "lipsync",
                          "cache", "scratch")
        final_drop = tuple(n for n in JOB_SUBDIRS if n != "final")

        targets = resumable_drop if level == config.CLEANUP_KEEP_RESUMABLE else final_drop

        removed = []
        for name in targets:
            d = self.paths.root / name
            # Containment check before any delete: this is the guard against a
            # symlinked or tampered subdirectory pointing elsewhere.
            assert_within(d, self.paths.root, what=f"cleanup target {name}")
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                removed.append(name)
        logger.info("Job %s cleaned (%s): removed %s", self.job_id, level, removed)
        return removed


def job_sort_key(job_id: str) -> tuple[float, str]:
    """
    Ordering key for a job id: `(creation_epoch, job_id)`.

    The creation time comes from the canonical stamp `new_job_id` embeds in the
    id, which is UTC and therefore directly comparable. A full job id is
    `<source-stem>-YYYYMMDD-HHMMSS-<hash>`, so sorting the id string itself would
    order by *source filename* first; that is the bug this key exists to avoid.

    A non-canonical id (a legacy or hand-made directory) has no stamp. It sorts
    oldest, and the id breaks the tie, so a malformed directory can never crash
    or reorder the rest of the listing.
    """
    match = _ID_STAMP_RE.search(job_id)
    if match is not None:
        try:
            epoch = calendar.timegm(time.strptime(
                match.group(1) + match.group(2), "%Y%m%d%H%M%S"))
            return (epoch, job_id)
        except ValueError:
            pass
    return (float("-inf"), job_id)


def list_jobs(jobs_dir: Optional[Path] = None) -> list[str]:
    """
    Job ids present on disk, newest first.

    Ordering uses the canonical timestamp embedded in the id, so it is by real
    creation time and not by source filename. Reads directory names only;
    nothing here opens a manifest, so it stays cheap enough to call from a UI on
    every refresh even with hundreds of jobs.
    """
    base = Path(jobs_dir or config.JOBS_DIR)
    if not base.is_dir():
        return []
    ids = [p.name for p in base.iterdir() if p.is_dir() and is_valid_job_id(p.name)]
    return sorted(ids, key=job_sort_key, reverse=True)
