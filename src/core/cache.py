"""
Content-addressed artifact cache (Phase 6).

Two stages are expensive enough that redoing them after a crash is the
difference between "resume" and "start over": translation and TTS. Both are pure
functions of their inputs, so both are cached on a digest of exactly those
inputs.

The rule that matters: **never key on segment number.** A segment's number is a
position, not an identity. If the translation changes, or the speaker's
reference changes, or the model changes, the digest changes and the old artifact
is correctly ignored rather than silently reused. Keying on index alone would
produce a dub where half the lines are spoken by the previous run's voice.

Writes are atomic and validated (`jobs.atomic_write_*`, `jobs.validate_wav`), so
a truncated file from a crash mid-write is never mistaken for a cache hit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.core.manifest import fingerprint

from src.core import jobs

logger = logging.getLogger(__name__)

# Bump to invalidate every cached translation when the key's meaning changes
# (for example when a new field is added that affects the output).
TRANSLATION_KEY_VERSION = 1
TTS_KEY_VERSION = 1


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------
def translation_key(source_text: str, source_lang: str, target_lang: str,
                    backend: str, settings: Optional[dict] = None) -> str:
    """
    Identity of a translation.

    Includes the text, both languages, the backend and any settings that change
    the output (concise/context flags, pivot path). Excludes the segment index
    and timestamps: the same line translated twice is the same translation.
    """
    return fingerprint({
        "v": TRANSLATION_KEY_VERSION,
        "kind": "translation",
        "source_text": (source_text or "").strip(),
        "source_lang": source_lang or "",
        "target_lang": target_lang or "",
        "backend": backend or "",
        "settings": settings or {},
    })


def tts_key(text: str, language: str, model: str,
            speaker_reference: Optional[str] = None,
            reference_text: Optional[str] = None,
            fix_duration: Optional[float] = None,
            settings: Optional[dict] = None,
            reference_identity: Optional[str] = None) -> str:
    """
    Identity of one synthesized line.

    `reference_identity` is preferred over the reference path when the caller
    has it: the same speaker re-extracted to a new path is the same voice, and
    the same path re-recorded is a different one. Falling back to the path keeps
    the key usable where only a path is available.
    """
    ref = reference_identity if reference_identity is not None else (speaker_reference or "")
    return fingerprint({
        "v": TTS_KEY_VERSION,
        "kind": "tts",
        "text": (text or "").strip(),
        "language": language or "",
        "model": model or "",
        "reference": ref,
        "reference_text": (reference_text or "").strip(),
        # Rounded: a duration request differing in the fourth decimal is the
        # same request, and un-rounded floats make the key needlessly brittle.
        "fix_duration": round(float(fix_duration), 3) if fix_duration else None,
        "settings": settings or {},
    })


def reference_identity(path: Optional[str | Path]) -> str:
    """
    A stable identity for a reference clip.

    Hashes the path, size and mtime rather than the contents: re-extracting the
    same speaker to the same file produces a new mtime, which is exactly when we
    want a different identity, and hashing multi-second WAVs on every segment
    would be the slowest part of the run.

    The mtime is kept at full resolution. Truncating it to whole seconds would
    make two re-extractions within the same second look identical, and reuse a
    voice that no longer matches.
    """
    if not path:
        return ""
    p = Path(str(path))
    try:
        st = p.stat()
        return fingerprint({"path": str(p), "size": st.st_size,
                            "mtime": st.st_mtime, "mtime_ns": st.st_mtime_ns})
    except OSError:
        # A missing reference is its own identity; it must not collide with a
        # real one, and it must not match a later successful extraction.
        return fingerprint({"path": str(p), "missing": True})


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class ArtifactCache:
    """
    File-backed cache inside a job's `cache/` directory.

    Layout:
        cache/tts/<key>.wav
        cache/translation.json     (key -> text)

    Small text results live in one JSON file (cheap to read, cheap to diff);
    audio lives in one file per key (large, and we want to read a single one
    without touching the rest).
    """

    def __init__(self, job: jobs.Job):
        self.job = job
        self.root = job.paths.sub("cache")
        self.root.mkdir(parents=True, exist_ok=True)
        self._translations: Optional[dict] = None
        self._translations_dirty = False

    # -- translation -------------------------------------------------------
    @property
    def translation_file(self) -> Path:
        return self.root / "translation.json"

    def _load_translations(self) -> dict:
        if self._translations is None:
            data = jobs.read_json(self.translation_file, default={})
            self._translations = data if isinstance(data, dict) else {}
        return self._translations

    def get_translation(self, key: str) -> Optional[str]:
        value = self._load_translations().get(key)
        return value if isinstance(value, str) and value else None

    def put_translation(self, key: str, text: str) -> None:
        if not text:
            return
        self._load_translations()[key] = text
        self._translations_dirty = True

    def flush_translations(self) -> None:
        """
        Persist accumulated translations in one atomic write.

        Called after a batch rather than per line: a movie has thousands of
        lines and rewriting the JSON per line would dominate the run.
        """
        if not self._translations_dirty:
            return
        jobs.atomic_write_json(self.translation_file, self._load_translations())
        self._translations_dirty = False

    # -- TTS ---------------------------------------------------------------
    def tts_path(self, key: str) -> Path:
        # Key is a hex digest, but sanitise anyway: this function is the only
        # way a cache path is built, so it is the place to guarantee it.
        safe = jobs.safe_name(key, max_length=64)
        if not safe:
            raise jobs.JobError("Empty cache key")
        return jobs.assert_within(self.root / "tts" / f"{safe}.wav", self.root)

    def has_tts(self, key: str, min_duration: float = 0.05) -> bool:
        """
        Whether a usable TTS artifact exists. Validates, does not assume.

        Checks silence as well as shape: a cache entry that is well-formed
        silence is not a hit, it is the same failure waiting to be reused.
        """
        try:
            path = self.tts_path(key)
        except jobs.JobError:
            return False
        return (jobs.validate_wav(path, min_duration=min_duration)
                and not jobs.is_silent(path))

    def get_tts(self, key: str) -> Optional[Path]:
        return self.tts_path(key) if self.has_tts(key) else None

    def put_tts(self, key: str, source_path: str | Path) -> Optional[Path]:
        """
        Copy a generated WAV into the cache, atomically and only if valid.

        Returns the cached path, or None if the source was not usable. A silent
        or truncated result is refused rather than cached: caching a bad file
        would make the failure permanent across every future resume.
        """
        source = Path(str(source_path))
        if not jobs.validate_wav(source):
            logger.warning("Refusing to cache unusable TTS output: %s", source)
            return None
        # A failed backend often returns well-formed silence instead of raising.
        # Caching that would turn one bad segment into a permanently bad one.
        if jobs.is_silent(source):
            logger.warning("Refusing to cache silent TTS output: %s", source)
            return None
        target = self.tts_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = source.read_bytes()
        except OSError as exc:
            logger.warning("Could not read TTS output %s: %s", source, exc)
            return None
        jobs.atomic_write_bytes(target, data)
        return target

    # -- housekeeping ------------------------------------------------------
    def stats(self) -> dict:
        tts_dir = self.root / "tts"
        count = 0
        size = 0
        if tts_dir.is_dir():
            for p in tts_dir.glob("*.wav"):
                count += 1
                try:
                    size += p.stat().st_size
                except OSError:
                    pass
        return {"tts_entries": count, "tts_bytes": size,
                "translations": len(self._load_translations())}
