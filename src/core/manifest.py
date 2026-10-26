"""
The job manifest: one authoritative, durable record of a render (Phase 6).

Everything needed to answer "what was this, what has been done, and what is
left" lives here and is written to `jobs/<job_id>/manifest.json` after every
stage and every batch of segments. A crash therefore loses at most the work in
flight, not the work already done.

Design notes:

  * **Versioned.** `schema_version` is checked on load. An unknown version is
    refused rather than half-understood, because a manifest that is misread is
    worse than one that is missing.
  * **Atomic.** Writes go through `jobs.atomic_write_json`, so an interrupted
    save leaves the previous manifest intact.
  * **Recoverable.** `Manifest.load` falls back to the `.bak` copy when the
    primary file is corrupt, and repairs the primary from it.
  * **Explicit stages.** A stage is one of `config.JOB_STAGES`; its status and
    the fingerprint of its inputs decide whether a resume reruns it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.core import jobs
from src.utils import config

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
BACKUP_SUFFIX = ".bak"

# Stage lifecycle.
STAGE_PENDING = "PENDING"
STAGE_RUNNING = "RUNNING"
STAGE_COMPLETE = "COMPLETE"
STAGE_PARTIAL = "PARTIAL"      # some work done, safely resumable
STAGE_FAILED = "FAILED"
STAGE_SKIPPED = "SKIPPED"

_TERMINAL_OK = (STAGE_COMPLETE, STAGE_SKIPPED)


def now() -> float:
    return time.time()


def fingerprint(payload: Any) -> str:
    """
    A stable short hash of whatever decides whether cached work is still valid.

    Keys are sorted and values are stringified, so two calls with the same
    meaning produce the same digest even if a dict was built in another order.
    """
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class StageRecord:
    """One pipeline stage's status and the fingerprint of its inputs."""

    name: str
    status: str = STAGE_PENDING
    fingerprint: str = ""
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    attempts: int = 0
    error: Optional[str] = None
    # Free-form per-stage detail (segment counts, output paths, timings).
    detail: dict = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.status in _TERMINAL_OK

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "fingerprint": self.fingerprint,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "attempts": self.attempts,
            "error": self.error,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StageRecord":
        return cls(
            name=data.get("name", ""),
            status=data.get("status", STAGE_PENDING),
            fingerprint=data.get("fingerprint", ""),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            attempts=int(data.get("attempts", 0) or 0),
            error=data.get("error"),
            detail=dict(data.get("detail") or {}),
        )


@dataclass
class SegmentRecord:
    """
    Per-segment durable state.

    `cache_key` is the content identity of the TTS output (and `translation_key`
    of the translation), so a resume can tell a reusable artifact from a stale
    one without regenerating anything.
    """

    index: int
    speaker_id: Optional[str] = None
    start: float = 0.0
    end: float = 0.0
    source_text: str = ""
    translated_text: str = ""
    status: str = STAGE_PENDING
    audio_path: Optional[str] = None
    translation_key: str = ""
    cache_key: str = ""
    attempts: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "speaker_id": self.speaker_id,
            "start": self.start,
            "end": self.end,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "status": self.status,
            "audio_path": self.audio_path,
            "translation_key": self.translation_key,
            "cache_key": self.cache_key,
            "attempts": self.attempts,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SegmentRecord":
        return cls(
            index=int(data.get("index", 0)),
            speaker_id=data.get("speaker_id"),
            start=float(data.get("start", 0.0) or 0.0),
            end=float(data.get("end", 0.0) or 0.0),
            source_text=data.get("source_text", "") or "",
            translated_text=data.get("translated_text", "") or "",
            status=data.get("status", STAGE_PENDING),
            audio_path=data.get("audio_path"),
            translation_key=data.get("translation_key", "") or "",
            cache_key=data.get("cache_key", "") or "",
            attempts=int(data.get("attempts", 0) or 0),
            error=data.get("error"),
        )


class Manifest:
    """
    The job's authoritative state, backed by `manifest.json`.

    The pipeline mutates this object and calls `save()`; nothing else writes the
    file, so there is exactly one writer and no merge problem.
    """

    def __init__(self, job: jobs.Job, data: Optional[dict] = None):
        self.job = job
        self.path = job.paths.manifest
        data = data or {}
        self.schema_version = int(data.get("schema_version", config.JOB_SCHEMA_VERSION))
        self.job_id = data.get("job_id", job.job_id)
        self.created_at = data.get("created_at", now())
        self.updated_at = data.get("updated_at", self.created_at)
        self.status = data.get("status", "PENDING")

        # Source identity and configuration.
        self.source = dict(data.get("source") or {})
        self.config = dict(data.get("config") or {})
        self.software = dict(data.get("software") or {})

        # Stage and segment state.
        self.stages: dict[str, StageRecord] = {
            name: StageRecord.from_dict(rec)
            for name, rec in (data.get("stages") or {}).items()
        }
        for name in config.JOB_STAGES:
            self.stages.setdefault(name, StageRecord(name=name))

        self.segments: dict[int, SegmentRecord] = {}
        for rec in (data.get("segments") or {}).values():
            try:
                sr = SegmentRecord.from_dict(rec)
                self.segments[sr.index] = sr
            except (TypeError, ValueError):
                continue

        # Chunked runs keep per-chunk progress separately from `segments`.
        # Chunks and speech segments are different things that happen to be
        # numbered from zero, and sharing one dict would let chunk 3 overwrite
        # the record for speech segment 3.
        self.chunks: dict[int, SegmentRecord] = {}
        for rec in (data.get("chunks") or {}).values():
            try:
                cr = SegmentRecord.from_dict(rec)
                self.chunks[cr.index] = cr
            except (TypeError, ValueError):
                continue

        # Cross-stage durable facts.
        self.speakers: dict = dict(data.get("speakers") or {})
        self.references: dict = dict(data.get("references") or {})
        self.artifacts: dict = dict(data.get("artifacts") or {})
        self.diagnostics: dict = dict(data.get("diagnostics") or {})
        self.errors: list = list(data.get("errors") or [])
        self.notes: list = list(data.get("notes") or [])

    # -- construction ------------------------------------------------------
    @classmethod
    def create(cls, job: jobs.Job, source_path: str | Path,
               source_lang: str, target_lang: str, run_config: dict,
               software: Optional[dict] = None) -> "Manifest":
        """A fresh manifest for a new render."""
        m = cls(job)
        m.source = {
            "path": str(source_path),
            "name": Path(str(source_path)).name,
            "size": _file_size(source_path),
            "language": source_lang,
            "target_language": target_lang,
            # Bumped when the fingerprint scheme changes. It is part of every
            # stage's fingerprint, so a manifest written before Phase 7 (whose
            # fingerprints hashed the whole config) cannot falsely match a
            # scoped one and hand back work made under different settings.
            "fingerprint_scope": config.FINGERPRINT_SCOPE,
        }
        m.config = dict(run_config)
        m.software = dict(software or software_versions())
        m.status = "RUNNING"
        return m

    @classmethod
    def load(cls, job: jobs.Job) -> Optional["Manifest"]:
        """
        Load a manifest, recovering from a corrupt primary file if possible.

        Returns None when there is nothing readable, which callers treat as "no
        checkpoint, start fresh" rather than as an error.
        """
        data = jobs.read_json(job.paths.manifest)
        if data is None:
            backup = _backup_path(job.paths.manifest)
            data = jobs.read_json(backup)
            if data is None:
                return None
            logger.warning("Job %s: primary manifest unreadable, recovered from %s",
                           job.job_id, backup.name)
            # Repair the primary from the backup so the next write is normal.
            try:
                jobs.atomic_write_json(job.paths.manifest, data)
            except OSError:
                pass

        version = int(data.get("schema_version", 0) or 0)
        if version != config.JOB_SCHEMA_VERSION:
            # Refuse rather than guess. The caller decides what to do; the
            # common choice is to start a new job and leave the old files alone.
            logger.error("Job %s: manifest schema %s, expected %s; refusing to load",
                         job.job_id, version, config.JOB_SCHEMA_VERSION)
            raise jobs.JobError(
                f"Manifest schema version {version} is not supported "
                f"(expected {config.JOB_SCHEMA_VERSION})")

        return cls(job, data)

    @classmethod
    def load_or_create(cls, job: jobs.Job, **create_kwargs) -> "Manifest":
        try:
            existing = cls.load(job)
        except jobs.JobError:
            existing = None
        return existing if existing is not None else cls.create(job, **create_kwargs)

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "source": self.source,
            "config": self.config,
            "software": self.software,
            "stages": {name: rec.to_dict() for name, rec in self.stages.items()},
            "segments": {str(i): rec.to_dict() for i, rec in self.segments.items()},
            "chunks": {str(i): rec.to_dict() for i, rec in self.chunks.items()},
            "speakers": self.speakers,
            "references": self.references,
            "artifacts": self.artifacts,
            "diagnostics": self.diagnostics,
            "errors": self.errors[-200:],
            "notes": self.notes[-200:],
        }

    def save(self) -> Path:
        """
        Write the manifest atomically, keeping the previous copy as a backup.

        Never raises: losing the index is bad, but failing a finished render
        because the index could not be written would be worse. The caller sees
        the failure in the log.

        The new primary is written *before* the backup is refreshed, and the old
        contents are copied rather than moved. This ordering is what makes the
        primary file never absent: `load` treats a missing primary as a reason to
        fall back to the backup and repair the primary from it, so a window with
        no primary would let a concurrent reader resurrect the previous state —
        a status poll during a stage write could otherwise revert a completed
        stage or a terminal status.
        """
        self.updated_at = now()
        payload = self.to_dict()
        try:
            previous: Optional[bytes] = None
            if self.path.exists():
                try:
                    previous = self.path.read_bytes()
                except OSError:
                    previous = None
            jobs.atomic_write_json(self.path, payload)
            if previous is not None:
                try:
                    jobs.atomic_write_bytes(_backup_path(self.path), previous)
                except OSError:
                    pass
            return self.path
        except OSError as exc:
            logger.error("Job %s: could not save manifest: %s", self.job_id, exc)
            return self.path

    # -- stage helpers -----------------------------------------------------
    def stage(self, name: str) -> StageRecord:
        if name not in self.stages:
            self.stages[name] = StageRecord(name=name)
        return self.stages[name]

    def start_stage(self, name: str, fingerprint_value: str = "") -> StageRecord:
        rec = self.stage(name)
        rec.status = STAGE_RUNNING
        rec.started_at = now()
        rec.finished_at = None
        rec.error = None
        rec.attempts += 1
        if fingerprint_value:
            rec.fingerprint = fingerprint_value
        self.save()
        return rec

    def complete_stage(self, name: str, detail: Optional[dict] = None) -> StageRecord:
        rec = self.stage(name)
        rec.status = STAGE_COMPLETE
        rec.finished_at = now()
        rec.error = None
        if detail:
            rec.detail.update(detail)
        self.save()
        return rec

    def partial_stage(self, name: str, detail: Optional[dict] = None) -> StageRecord:
        """Mark a stage as having produced some, but not all, of its output."""
        rec = self.stage(name)
        rec.status = STAGE_PARTIAL
        rec.finished_at = now()
        if detail:
            rec.detail.update(detail)
        self.save()
        return rec

    def fail_stage(self, name: str, error: str, detail: Optional[dict] = None) -> StageRecord:
        rec = self.stage(name)
        rec.status = STAGE_FAILED
        rec.finished_at = now()
        rec.error = str(error)[:2000]
        if detail:
            rec.detail.update(detail)
        self.record_error(name, error)
        self.save()
        return rec

    def skip_stage(self, name: str, reason: str = "") -> StageRecord:
        """
        Mark a stage as deliberately not producing output.

        Distinct from COMPLETE: the stage ran and decided there was nothing to
        do (subtitles with no translated text, for instance). `SKIPPED` is
        terminal-OK — it is done and will not run again — but it is not a claim
        that a result exists, which is what a later reader (the artifact list,
        the completion invariants) needs to tell apart from a real result.
        """
        rec = self.stage(name)
        rec.status = STAGE_SKIPPED
        rec.finished_at = now()
        rec.error = None
        if reason:
            rec.detail["skipped"] = str(reason)[:2000]
        self.save()
        return rec

    def record_error(self, stage: str, error: str, segment_index: Optional[int] = None) -> None:
        self.errors.append({
            "stage": stage,
            "segment_index": segment_index,
            "error": str(error)[:2000],
            "at": now(),
        })

    def note(self, message: str) -> None:
        self.notes.append({"message": str(message)[:2000], "at": now()})

    # -- validity ----------------------------------------------------------
    def differs_from(self, requested_config: dict) -> list[str]:
        """
        Configuration keys whose requested value differs from the recorded one.

        A small, deliberate check rather than a reconciler: it only reports
        whether the run being resumed was set up differently, so the caller can
        refuse instead of silently reusing output made under other settings.
        Only keys the manifest actually recorded are compared, so an omitted
        argument (which `_run_config` still records as None) does not read as a
        change.
        """
        recorded = self.config or {}
        return sorted(
            key for key, value in (requested_config or {}).items()
            if key in recorded and recorded[key] != value
        )

    def stages_for_config_keys(self, keys) -> list[str]:
        """
        The stages whose work CAUSES any of `keys` to change.

        A key belongs to the stage whose configuration owns it (for example
        `lipsync_model_name` belongs to `lipsync`). Invalidating from the
        earliest such stage discards that stage and everything downstream, which
        is exactly the work that must be redone — and nothing upstream of it.
        """
        return config.stages_owning_config_keys(keys)

    def reconcile(self, requested_config: dict, reason: str = "") -> dict:
        """
        Adopt `requested_config` and invalidate only the affected stages.

        This is the dependency-aware alternative to refusing a resume outright:
        changing a lip-sync setting re-renders lip-sync and the final mux while
        Demucs, ASR, diarization, references, translation and TTS all stay
        cached. The recorded config is updated so fingerprints computed on the
        next stage visits reflect the new settings.

        Returns a report: `changed` keys, `retained` keys, and the `invalidated`
        stage names.
        """
        requested_config = dict(requested_config or {})
        changed = self.differs_from(requested_config)
        owned = self.stages_for_config_keys(changed)
        # Invalidate from the earliest affected stage, which carries the rest
        # with it. `min` on JOB_STAGES order, not alphabetically.
        invalidated: list[str] = []
        if owned:
            earliest = min(owned, key=config.JOB_STAGES.index)
            invalidated = self.invalidate_from(
                earliest, reason=reason or f"configuration changed: {', '.join(changed)}")
        # Adopt the new values after invalidation so the saved config is current.
        for key, value in requested_config.items():
            if key in self.config:
                self.config[key] = value
        self.save()
        return {
            "changed": changed,
            "retained": sorted(k for k in requested_config if k not in changed),
            "invalidated": invalidated,
        }

    def stage_is_valid(self, name: str, current_fingerprint: str) -> bool:
        """
        Whether a completed stage can be trusted for the current inputs.

        A stage counts as valid when it finished AND its recorded input
        fingerprint matches. An empty recorded fingerprint is treated as a
        mismatch: we cannot prove the work is current, so it is redone.
        """
        rec = self.stages.get(name)
        if rec is None or not rec.done:
            return False
        if not rec.fingerprint:
            return False
        return rec.fingerprint == current_fingerprint

    def stages_after(self, name: str) -> list[str]:
        """Stage names that depend on `name` (strictly after it, in order)."""
        if name not in config.JOB_STAGES:
            return []
        idx = config.JOB_STAGES.index(name)
        return list(config.JOB_STAGES[idx + 1:])

    def invalidate_from(self, name: str, reason: str = "") -> list[str]:
        """
        Discard `name` and everything downstream. Returns the names invalidated.

        This is the whole point of dependency-aware invalidation: changing the
        target language must not cost us Demucs and ASR, and it must not leave a
        stale translation lying around pretending to be current either.
        """
        affected = [name] + self.stages_after(name)
        for stage_name in affected:
            rec = self.stage(stage_name)
            rec.status = STAGE_PENDING
            rec.finished_at = None
            rec.error = None
            rec.detail = {}
            # Fingerprints are cleared so a stale one can never match later.
            rec.fingerprint = ""
        if reason:
            self.note(f"Invalidated from {name}: {reason}")
        self.save()
        return affected

    def invalidate_speaker(self, speaker_id: str, reason: str = "") -> list[str]:
        """
        Invalidate only the TTS work for one speaker and its downstream output.

        Other speakers' audio is untouched: a reference for Speaker_02 changing
        must not cost Speaker_01 a re-synthesis.
        """
        affected_indices = []
        for idx, seg in self.segments.items():
            if seg.speaker_id == speaker_id:
                seg.status = STAGE_PENDING
                seg.error = None
                seg.cache_key = ""
                seg.audio_path = None
                affected_indices.append(idx)

        # The stage-level results that aggregate per-segment audio are stale.
        for stage_name in ("tts", "sync", "subtitles", "lipsync", "final"):
            rec = self.stage(stage_name)
            if stage_name != "tts":
                rec.status = STAGE_PENDING
                rec.finished_at = None
                rec.fingerprint = ""
            else:
                # TTS is partial by definition now: some segments are pending.
                rec.status = STAGE_PARTIAL
                rec.finished_at = None
                rec.fingerprint = ""
        if reason:
            self.note(f"Invalidated speaker {speaker_id}: {reason}")
        self.save()
        return affected_indices

    def invalidate_segment(self, index: int, reason: str = "") -> list[str]:
        """Invalidate one segment's TTS and the timeline that contains it."""
        seg = self.segments.get(index)
        if seg is not None:
            seg.status = STAGE_PENDING
            seg.error = None
            seg.cache_key = ""
            seg.audio_path = None
        for stage_name in ("tts", "sync", "subtitles", "lipsync", "final"):
            rec = self.stage(stage_name)
            rec.status = STAGE_PARTIAL if stage_name == "tts" else STAGE_PENDING
            rec.finished_at = None
            rec.fingerprint = ""
        if reason:
            self.note(f"Invalidated segment {index}: {reason}")
        self.save()
        return ["tts", "sync", "subtitles", "lipsync", "final"]

    # -- segments ----------------------------------------------------------
    def upsert_segment(self, index: int, **fields) -> SegmentRecord:
        seg = self.segments.get(index)
        if seg is None:
            seg = SegmentRecord(index=index)
            self.segments[index] = seg
        for key, value in fields.items():
            if hasattr(seg, key) and value is not None:
                setattr(seg, key, value)
        return seg

    def upsert_chunk(self, index: int, **fields) -> SegmentRecord:
        """Record per-chunk progress for a chunked run."""
        rec = self.chunks.get(index)
        if rec is None:
            rec = SegmentRecord(index=index)
            self.chunks[index] = rec
        for key, value in fields.items():
            if hasattr(rec, key) and value is not None:
                setattr(rec, key, value)
        return rec

    def chunk_counts(self) -> dict:
        counts = {"total": len(self.chunks), "complete": 0, "pending": 0,
                  "failed": 0}
        for rec in self.chunks.values():
            if rec.status == STAGE_COMPLETE:
                counts["complete"] += 1
            elif rec.status == STAGE_FAILED:
                counts["failed"] += 1
            else:
                counts["pending"] += 1
        return counts

    def segment_counts(self) -> dict:
        counts = {"total": len(self.segments), "complete": 0, "pending": 0,
                  "failed": 0, "partial": 0}
        for seg in self.segments.values():
            if seg.status == STAGE_COMPLETE:
                counts["complete"] += 1
            elif seg.status == STAGE_FAILED:
                counts["failed"] += 1
            elif seg.status == STAGE_PARTIAL:
                counts["partial"] += 1
            else:
                counts["pending"] += 1
        return counts

    # -- progress ----------------------------------------------------------
    def progress(self) -> dict:
        """
        A resume-safe progress view built from durable state, not counters.

        Every value here is derived from the manifest, so it is identical
        whether the job is running, was just restarted, or was never finished.
        """
        stages = {}
        for name in config.JOB_STAGES:
            rec = self.stage(name)
            stages[name] = {"status": rec.status, "error": rec.error,
                            "attempts": rec.attempts}
        counts = self.segment_counts()
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stages": stages,
            "segments": counts,
            "updated_at": self.updated_at,
        }

    def render_progress_text(self) -> str:
        """The human-readable progress block, reconstructed from disk."""
        lines = [f"Job: {self.job_id}", ""]
        for name in config.JOB_STAGES:
            rec = self.stage(name)
            label = name.replace("_", " ").title()
            if name == "tts" and rec.status in (STAGE_PARTIAL, STAGE_RUNNING):
                c = self.segment_counts()
                lines.append(f"{label:<18}{rec.status:<9}({c['complete']}/{c['total']})")
            else:
                lines.append(f"{label:<18}{rec.status}")
        return "\n".join(lines)


def _backup_path(path: Path) -> Path:
    return path.with_name(path.name + BACKUP_SUFFIX)


def _file_size(path: str | Path) -> int:
    try:
        return Path(str(path)).stat().st_size
    except OSError:
        return 0


def software_versions() -> dict:
    """
    Versions of the things that change the meaning of cached artifacts.

    Recorded so a manifest can be judged against the code that will read it.
    Best-effort: a missing version is not worth failing a render over.
    """
    versions = {"schema": config.JOB_SCHEMA_VERSION}
    try:
        # The language registry decides which translation/TTS route is taken,
        # so a change to it can change cached output.
        from src.utils import languages as lang_registry
        versions["languages"] = len(getattr(lang_registry, "LANGUAGES", {}) or {})
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        versions["torch"] = getattr(torch, "__version__", "")
    except Exception:  # noqa: BLE001
        pass
    return versions
