"""
Phase 6: durable jobs, resume, and content-addressed caching.

These tests exercise the real modules on the real filesystem. Nothing here is
mocked: the whole point of Phase 6 is what happens to bytes on disk across a
process boundary, and a mock cannot demonstrate that.
"""

import numpy as np
import pytest
import soundfile as sf

from pathlib import Path

from src.core import jobs
from src.core.cache import ArtifactCache, reference_identity, tts_key
from src.core.manifest import (
    Manifest, STAGE_COMPLETE, STAGE_FAILED, STAGE_PARTIAL, STAGE_PENDING,
    fingerprint,
)
from src.core.retry import (
    FailureKind, classify, run_with_retry, should_retry_segment, summarise,
)
from src.utils import config

SR = 16000


def tone(duration=1.0, freq=220.0):
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return 0.4 * np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.fixture

def jobs_dir(tmp_path, monkeypatch):
    """Point JOBS_DIR at a temp dir so tests never touch the real jobs folder."""
    target = tmp_path / "jobs"
    monkeypatch.setattr(config, "JOBS_DIR", target)
    target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture

def job(jobs_dir):
    return jobs.Job.create(source_path="/tmp/movie.mp4")


class TestJobWorkspace:

    def test_job_id_is_safe_and_unique(self, jobs_dir):
        j = jobs.Job.create(source_path="/tmp/My Movie (2024).mkv")
        assert jobs.is_valid_job_id(j.job_id)
        # A new render is a new job unless the caller asks for the old id back.
        assert jobs.Job.create(source_path="/tmp/My Movie (2024).mkv").job_id != j.job_id

    def test_reopening_a_job_keeps_its_files(self, jobs_dir):
        j = jobs.Job.create(source_path="/tmp/a.mp4")
        artifact = j.stage_dir("separation") / "vocals.wav"
        sf.write(str(artifact), tone(1.0), SR)

        reopened = jobs.Job.load(j.job_id, jobs_dir=jobs_dir)
        assert reopened is not None
        assert reopened.stage_dir("separation") == j.stage_dir("separation")
        assert artifact.exists()

    def test_stage_dir_accepts_stage_or_directory_name(self, job):
        assert job.stage_dir("references") == job.stage_dir("speaker_refs")

    def test_unknown_stage_is_refused(self, job):
        with pytest.raises(jobs.JobError):
            job.stage_dir("not_a_stage")

    def test_path_traversal_is_refused(self, job):
        with pytest.raises(jobs.PathEscapeError):
            jobs.assert_within(job.root / ".." / ".." / "etc" / "passwd", job.root)

    def test_invalid_job_id_is_refused(self, jobs_dir):
        for bad in ("../escape", "a/b", "", "x" * 200):
            with pytest.raises(jobs.JobError):
                jobs.require_job_id(bad)

    def test_log_file_accumulates(self, job):
        job.log("first")
        job.log("second")
        text = job.paths.logs.read_text(encoding="utf-8")
        assert "first" in text and "second" in text
        assert text.index("first") < text.index("second")


class TestAtomicWrites:

    def test_json_round_trips(self, tmp_path):
        target = tmp_path / "data.json"
        payload = {"b": 2, "a": [1, 2, 3], "nested": {"x": None}}
        jobs.atomic_write_json(target, payload)
        assert jobs.read_json(target) == payload

    def test_json_is_sort_keyed_and_utf8(self, tmp_path):
        target = tmp_path / "data.json"
        jobs.atomic_write_json(target, {"z": "తెలుగు", "a": 1})
        raw = target.read_text(encoding="utf-8")
        assert raw.index('"a"') < raw.index('"z"')
        assert "తెలుగు" in raw

    def test_identical_state_writes_identical_bytes(self, tmp_path):
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        jobs.atomic_write_json(a, {"x": 1, "y": [2, 3]})
        jobs.atomic_write_json(b, {"y": [2, 3], "x": 1})
        assert a.read_bytes() == b.read_bytes()

    def test_no_temp_file_is_left_behind(self, tmp_path):
        target = tmp_path / "data.json"
        for _ in range(5):
            jobs.atomic_write_json(target, {"n": 1})
        leftovers = [p.name for p in tmp_path.iterdir() if p.name != "data.json"]
        assert leftovers == []

    def test_unreadable_json_returns_default(self, tmp_path):
        target = tmp_path / "broken.json"
        target.write_text("{not json")
        assert jobs.read_json(target, default={"ok": True}) == {"ok": True}


class TestValidation:

    def test_valid_wav_is_accepted(self, tmp_path):
        p = tmp_path / "good.wav"
        sf.write(str(p), tone(1.0), SR)
        assert jobs.validate_wav(p, min_duration=0.1)

    def test_silence_is_shape_valid_but_flagged(self, tmp_path):
        # Silence is a well-formed WAV, so `validate_wav` accepts it; `is_silent`
        # is the check that catches it. Keeping the two separate lets a caller
        # validate shape without paying to read samples.
        p = tmp_path / "silent.wav"
        sf.write(str(p), np.zeros(SR, dtype=np.float32), SR)
        assert jobs.validate_wav(p)
        assert jobs.is_silent(p)

    def test_a_real_tone_is_not_silent(self, tmp_path):
        p = tmp_path / "tone.wav"
        sf.write(str(p), tone(1.0), SR)
        assert not jobs.is_silent(p)

    def test_truncated_file_is_rejected(self, tmp_path):
        p = tmp_path / "cut.wav"
        sf.write(str(p), tone(1.0), SR)
        data = p.read_bytes()
        p.write_bytes(data[: len(data) // 3])
        assert not jobs.validate_wav(p)

    def test_missing_and_empty_files_are_rejected(self, tmp_path):
        assert not jobs.validate_wav(tmp_path / "nope.wav")
        empty = tmp_path / "empty.wav"
        empty.touch()
        assert not jobs.validate_wav(empty)

    def test_very_short_wav_is_rejected_by_min_duration(self, tmp_path):
        p = tmp_path / "blip.wav"
        sf.write(str(p), tone(0.01), SR)
        assert not jobs.validate_wav(p, min_duration=0.5)


class TestManifest:

    def test_create_records_source_and_config(self, job):
        m = Manifest.create(job, source_path="/tmp/movie.mp4", source_lang="en",
                            target_lang="te", run_config={"tts_model_name": "xtts"})
        assert m.source["name"] == "movie.mp4"
        assert m.config["tts_model_name"] == "xtts"
        assert m.status == "RUNNING"

    def test_save_and_load_round_trip(self, job):
        m = Manifest.create(job, source_path="/tmp/m.mp4", source_lang="en",
                            target_lang="te", run_config={"a": 1})
        m.complete_stage("extraction", {"path": "/x/audio.wav"})
        m.upsert_segment(3, start=1.0, end=2.5, speaker_id="SPK_1",
                         translated_text="hello", status=STAGE_COMPLETE)
        m.save()

        reloaded = Manifest.load(job)
        assert reloaded.stages["extraction"].status == STAGE_COMPLETE
        assert reloaded.stages["extraction"].detail["path"] == "/x/audio.wav"
        assert reloaded.segments[3].speaker_id == "SPK_1"
        assert reloaded.segments[3].end == 2.5

    def test_stage_validity_depends_on_fingerprint(self, job):
        m = Manifest.create(job, source_path="/tmp/m.mp4", source_lang="en",
                            target_lang="te", run_config={})
        m.start_stage("separation", "abc123")
        m.complete_stage("separation")
        assert m.stage_is_valid("separation", "abc123")
        # A different input fingerprint must not be accepted as current.
        assert not m.stage_is_valid("separation", "different")
        # An empty recorded fingerprint cannot be proven current.
        m.start_stage("tts", "")
        m.complete_stage("tts")
        assert not m.stage_is_valid("tts", "")

    def test_incomplete_stage_is_never_valid(self, job):
        m = Manifest.create(job, source_path="/tmp/m.mp4", source_lang="en",
                            target_lang="te", run_config={})
        m.start_stage("tts", "abc")
        assert not m.stage_is_valid("tts", "abc")

    def test_invalidate_from_drops_downstream_only(self, job):
        m = Manifest.create(job, source_path="/tmp/m.mp4", source_lang="en",
                            target_lang="te", run_config={})
        for name in config.JOB_STAGES:
            m.start_stage(name, "fp-" + name)
            m.complete_stage(name)

        affected = m.invalidate_from("translation", reason="target changed")

        assert "translation" in affected and "final" in affected
        # Upstream work is untouched: this is the whole point of the design.
        assert m.stage_is_valid("extraction", "fp-extraction")
        assert m.stage_is_valid("separation", "fp-separation")
        assert m.stage_is_valid("diarization", "fp-diarization")
        assert not m.stage_is_valid("tts", "fp-tts")

    def test_invalidate_speaker_spares_other_speakers(self, job):
        m = Manifest.create(job, source_path="/tmp/m.mp4", source_lang="en",
                            target_lang="te", run_config={})
        m.upsert_segment(0, speaker_id="SPK_1", status=STAGE_COMPLETE, cache_key="k0")
        m.upsert_segment(1, speaker_id="SPK_2", status=STAGE_COMPLETE, cache_key="k1")

        m.invalidate_speaker("SPK_1")

        assert m.segments[0].status == STAGE_PENDING
        assert m.segments[0].cache_key == ""
        assert m.segments[1].status == STAGE_COMPLETE
        assert m.segments[1].cache_key == "k1"

    def test_segment_counts(self, job):
        m = Manifest.create(job, source_path="/tmp/m.mp4", source_lang="en",
                            target_lang="te", run_config={})
        m.upsert_segment(0, status=STAGE_COMPLETE)
        m.upsert_segment(1, status=STAGE_FAILED)
        m.upsert_segment(2, status=STAGE_PARTIAL)
        m.upsert_segment(3, status=STAGE_PENDING)
        assert m.segment_counts() == {"total": 4, "complete": 1, "failed": 1,
                                      "partial": 1, "pending": 1}

    def test_chunks_do_not_collide_with_segments(self, job):
        m = Manifest.create(job, source_path="/tmp/m.mp4", source_lang="en",
                            target_lang="te", run_config={})
        m.upsert_segment(3, status=STAGE_FAILED, translated_text="segment three")
        m.upsert_chunk(3, status=STAGE_COMPLETE)

        assert m.segments[3].translated_text == "segment three"
        assert m.segments[3].status == STAGE_FAILED
        assert m.chunk_counts()["complete"] == 1

        m.save()
        reloaded = Manifest.load(job)
        assert reloaded.segments[3].status == STAGE_FAILED
        assert reloaded.chunks[3].status == STAGE_COMPLETE

    def test_corrupt_manifest_recovers_from_backup(self, job):
        m = Manifest.create(job, source_path="/tmp/m.mp4", source_lang="en",
                            target_lang="te", run_config={})
        m.complete_stage("extraction")
        m.save()
        m.complete_stage("separation")
        m.save()  # writes the backup of the extraction state

        job.paths.manifest.write_text("{corrupt")
        recovered = Manifest.load(job)
        assert recovered is not None
        assert recovered.stages["extraction"].status == STAGE_COMPLETE

    def test_unreadable_manifest_returns_none(self, job):
        assert Manifest.load(job) is None

    def test_schema_mismatch_is_refused(self, job):
        m = Manifest.create(job, source_path="/tmp/m.mp4", source_lang="en",
                            target_lang="te", run_config={})
        data = m.to_dict()
        data["schema_version"] = config.JOB_SCHEMA_VERSION + 1
        jobs.atomic_write_json(job.paths.manifest, data)
        with pytest.raises(jobs.JobError):
            Manifest.load(job)

    def test_fingerprint_is_order_independent(self):
        assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
        assert fingerprint({"a": 1}) != fingerprint({"a": 2})


class TestArtifactCache:

    def test_tts_round_trip(self, job, tmp_path):
        cache = ArtifactCache(job)
        src = tmp_path / "seg.wav"
        sf.write(str(src), tone(1.0), SR)

        stored = cache.put_tts("key-one", src)
        assert stored is not None and stored.exists()
        assert cache.has_tts("key-one")
        assert cache.get_tts("key-one") == stored
        # The cached copy is independent of the original.
        src.unlink()
        assert cache.get_tts("key-one").exists()

    def test_silent_output_is_refused(self, job, tmp_path):
        cache = ArtifactCache(job)
        src = tmp_path / "silent.wav"
        sf.write(str(src), np.zeros(SR, dtype=np.float32), SR)
        assert cache.put_tts("key-silent", src) is None
        assert not cache.has_tts("key-silent")

    def test_missing_key_is_a_miss(self, job):
        assert ArtifactCache(job).get_tts("never-stored") is None

    def test_same_content_yields_same_key(self):
        a = tts_key(text="hello", language="te", model="xtts")
        b = tts_key(text="hello", language="te", model="xtts")
        assert a == b

    def test_each_input_changes_the_key(self):
        base = dict(text="hello", language="te", model="xtts")
        baseline = tts_key(**base)
        assert tts_key(**{**base, "text": "hello!"}) != baseline
        assert tts_key(**{**base, "language": "hi"}) != baseline
        assert tts_key(**{**base, "model": "piper"}) != baseline
        assert tts_key(**{**base, "reference_text": "clip"}) != baseline
        assert tts_key(**{**base, "fix_duration": 2.0}) != baseline
        assert tts_key(**{**base, "settings": {"cfg": True}}) != baseline
        assert tts_key(**{**base, "reference_identity": "abc"}) != baseline

    def test_reference_identity_tracks_file_content(self, tmp_path):
        p = tmp_path / "ref.wav"
        sf.write(str(p), tone(1.0), SR)
        first = reference_identity(p)
        sf.write(str(p), tone(1.0, freq=440.0), SR)
        assert reference_identity(p) != first

    def test_reference_identity_of_nothing_is_empty(self):
        assert reference_identity(None) == ""
        assert reference_identity("") == ""

    def test_a_missing_reference_has_its_own_identity(self):
        # A missing clip must not collide with a real one, nor with a later
        # successful extraction to the same path.
        missing = reference_identity("/nonexistent/path.wav")
        assert missing != ""
        assert missing != reference_identity(None)

    def test_translations_round_trip_after_flush(self, job):
        cache = ArtifactCache(job)
        cache.put_translation("k1", "నమస్కారం")
        # Not on disk until flushed.
        assert ArtifactCache(job).get_translation("k1") is None
        cache.flush_translations()
        assert ArtifactCache(job).get_translation("k1") == "నమస్కారం"

    def test_cache_survives_a_new_instance(self, job, tmp_path):
        src = tmp_path / "seg.wav"
        sf.write(str(src), tone(1.0), SR)
        key = tts_key(text="persisted", language="te", model="xtts")
        ArtifactCache(job).put_tts(key, src)
        assert ArtifactCache(job).get_tts(key) is not None


class TestFailureClassification:

    def test_transient_failures_are_recognised(self):
        for exc in (TimeoutError("connection timed out"),
                    OSError("device or resource busy"),
                    RuntimeError("CUDA out of memory")):
            assert classify(exc) is FailureKind.TRANSIENT

    def test_fatal_failures_are_recognised(self):
        for exc in (FileNotFoundError("no such file or directory"),
                    ValueError("unsupported language"),
                    RuntimeError("source video is corrupt")):
            assert classify(exc) is FailureKind.FATAL

    def test_fatal_marker_beats_transient_marker(self):
        exc = RuntimeError("model unavailable: request timed out")
        assert classify(exc) is FailureKind.FATAL

    def test_cancellation_is_not_retried(self):
        class CancelledError(Exception):
            pass
        assert classify(CancelledError("stopped")) is FailureKind.FATAL

    def test_fatal_exception_types_are_never_retried(self):
        assert classify(KeyError("missing")) is FailureKind.FATAL
        assert classify(TypeError("wrong type")) is FailureKind.FATAL

    def test_segment_retry_respects_the_limit(self):
        exc = TimeoutError("timed out")
        assert should_retry_segment(exc, attempts_so_far=0, max_attempts=2)
        assert not should_retry_segment(exc, attempts_so_far=2, max_attempts=2)
        assert not should_retry_segment(ValueError("bad"), 0, max_attempts=5)


class TestRetryPolicy:

    def test_a_success_needs_one_attempt(self):
        outcome = run_with_retry(lambda: 42, attempts=3)
        assert outcome.ok and outcome.value == 42 and outcome.attempts == 1

    def test_transient_failure_is_retried_then_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("timed out")
            return "done"

        outcome = run_with_retry(flaky, attempts=5, backoff=0.0)
        assert outcome.ok and outcome.value == "done" and outcome.attempts == 3

    def test_fatal_failure_is_not_retried(self):
        calls = {"n": 0}

        def broken():
            calls["n"] += 1
            raise ValueError("unsupported")

        outcome = run_with_retry(broken, attempts=5, backoff=0.0)
        assert not outcome.ok and calls["n"] == 1
        assert outcome.kind is FailureKind.FATAL

    def test_exhausted_attempts_report_the_last_error(self):
        outcome = run_with_retry(lambda: (_ for _ in ()).throw(TimeoutError("slow")),
                                 attempts=3, backoff=0.0)
        assert not outcome.ok and outcome.attempts == 3
        assert "slow" in outcome.error
        assert len(outcome.history) == 3

    def test_cancellation_propagates_immediately(self):
        class CancelledError(Exception):
            pass

        calls = {"n": 0}

        def cancelled():
            calls["n"] += 1
            raise CancelledError("user stopped it")

        with pytest.raises(CancelledError):
            run_with_retry(cancelled, attempts=5, backoff=0.0)
        assert calls["n"] == 1

    def test_attempt_count_is_at_least_one(self):
        outcome = run_with_retry(lambda: "x", attempts=0)
        assert outcome.ok

    def test_backoff_is_applied_between_attempts(self):
        slept = []
        run_with_retry(lambda: (_ for _ in ()).throw(TimeoutError("t")),
                       attempts=3, backoff=0.5, sleep=slept.append)
        assert slept == [0.5, 1.0]

    def test_summarise_describes_the_history(self):
        outcome = run_with_retry(lambda: (_ for _ in ()).throw(TimeoutError("t")),
                                 attempts=2, backoff=0.0)
        assert "2 attempt(s)" in summarise(outcome.history)


class TestCleanup:

    def test_keep_resumable_drops_only_regenerable_bulk(self, job):
        for stage in ("separation", "tts", "final", "transcription"):
            (job.stage_dir(stage) / "artifact.bin").write_bytes(b"x" * 1024)

        job.clean(level=config.CLEANUP_KEEP_RESUMABLE)

        # Demucs stems are bulky and cheap to redo from the source.
        assert not (job.stage_dir("separation") / "artifact.bin").exists()
        # Per-segment TTS is exactly what a resume must not have to redo.
        assert (job.stage_dir("tts") / "artifact.bin").exists()
        assert (job.stage_dir("transcription") / "artifact.bin").exists()
        assert (job.stage_dir("final") / "artifact.bin").exists()

    def test_final_only_keeps_the_final_output_and_manifest(self, job):
        for stage in ("separation", "tts", "final"):
            (job.stage_dir(stage) / "artifact.bin").write_bytes(b"x" * 16)
        job.clean(level=config.CLEANUP_FINAL_ONLY)
        assert not (job.stage_dir("separation") / "artifact.bin").exists()
        assert not (job.stage_dir("tts") / "artifact.bin").exists()
        # The delivered video is the one thing that must survive.
        assert (job.stage_dir("final") / "artifact.bin").exists()

    def test_clean_refuses_while_the_job_is_active(self, job):
        artifact = job.stage_dir("separation") / "artifact.bin"
        artifact.write_bytes(b"x" * 16)
        assert job.clean(level=config.CLEANUP_FINAL_ONLY, active=True) == []
        assert artifact.exists()

    def test_clean_keep_all_is_a_no_op(self, job):
        artifact = job.stage_dir("separation") / "artifact.bin"
        artifact.write_bytes(b"x" * 16)
        assert job.clean(level=config.CLEANUP_KEEP_ALL) == []
        assert artifact.exists()

    def test_disk_usage_reports_bytes(self, job):
        (job.stage_dir("tts") / "a.wav").write_bytes(b"y" * 512)
        usage = job.disk_usage()
        assert usage["total_bytes"] >= 512

    def test_list_jobs_finds_created_jobs(self, jobs_dir):
        a = jobs.Job.create(source_path="/tmp/a.mp4")
        b = jobs.Job.create(source_path="/tmp/b.mp4")
        found = jobs.list_jobs(jobs_dir=jobs_dir)
        assert a.job_id in found and b.job_id in found


def _make_job_dir(jobs_dir, job_id):
    """Create a job directory with an explicit id, bypassing `new_job_id`."""
    root = Path(jobs_dir) / job_id
    root.mkdir(parents=True, exist_ok=True)
    return job_id


class TestJobOrdering:
    """
    `list_jobs` must order by real creation time, not by source filename.

    A job id is `<source-stem>-YYYYMMDD-HHMMSS-<hash>`, so sorting the raw id
    string puts the *source filename* first. With the API's list truncation that
    made a freshly finished render vanish from `/api/jobs` whenever its source
    name sorted below the cut-off. These tests pin the timestamp ordering.
    """

    def test_newest_first_ignores_source_filename(self, jobs_dir):
        # The newer job has the *lower*-sorting stem, so lexicographic id order
        # would wrongly put `zzz_video` first. Timestamp order must not.
        older = _make_job_dir(jobs_dir, "zzz_video-20261020-120000-aaaaaaaa")
        newer = _make_job_dir(jobs_dir, "aaa_video-20261022-120000-bbbbbbbb")
        assert jobs.list_jobs(jobs_dir=jobs_dir) == [newer, older]

    def test_same_timestamp_breaks_the_tie_on_job_id(self, jobs_dir):
        low = _make_job_dir(jobs_dir, "clip-20261022-120000-aaaaaaaa")
        high = _make_job_dir(jobs_dir, "clip-20261022-120000-bbbbbbbb")
        assert jobs.list_jobs(jobs_dir=jobs_dir) == [high, low]

    def test_source_stem_may_contain_hyphens(self, jobs_dir):
        # Parsing must anchor on the canonical suffix, not a fixed split count.
        older = _make_job_dir(jobs_dir, "my-holiday-clip-20261020-120000-aaaaaaaa")
        newer = _make_job_dir(jobs_dir, "a-b-c-d-e-f-20261022-120000-bbbbbbbb")
        assert jobs.list_jobs(jobs_dir=jobs_dir) == [newer, older]

    def test_legacy_id_without_a_stamp_sorts_oldest_and_never_crashes(self, jobs_dir):
        stamped = _make_job_dir(jobs_dir, "clip-20261022-120000-aaaaaaaa")
        legacy = _make_job_dir(jobs_dir, "legacy_job_no_stamp")
        assert jobs.list_jobs(jobs_dir=jobs_dir) == [stamped, legacy]

    def test_malformed_stamp_falls_back_instead_of_raising(self, jobs_dir):
        # A well-formed id whose date cannot exist must not break listing.
        broken = _make_job_dir(jobs_dir, "clip-99999999-999999-aaaaaaaa")
        good = _make_job_dir(jobs_dir, "clip-20261022-120000-bbbbbbbb")
        assert jobs.list_jobs(jobs_dir=jobs_dir) == [good, broken]

    def test_mixed_legacy_and_new_are_all_returned(self, jobs_dir):
        ids = [
            "zzz_video-20261022-090000-aaaaaaaa",
            "legacy_no_stamp",
            "aaa_video-20261021-090000-bbbbbbbb",
            "clip-20261020-090000-cccccccc",
        ]
        for job_id in ids:
            _make_job_dir(jobs_dir, job_id)
        found = jobs.list_jobs(jobs_dir=jobs_dir)
        assert sorted(found) == sorted(ids)
        # Newest stamped job first; the unstamped legacy id last.
        assert found[0] == "zzz_video-20261022-090000-aaaaaaaa"
        assert found[-1] == "legacy_no_stamp"

    def test_job_sort_key_reads_the_canonical_timestamp(self):
        import calendar
        import time as _time

        epoch, tie = jobs.job_sort_key("my-clip-20261022-120000-aaaaaaaa")
        expected = calendar.timegm(_time.strptime("20261022120000", "%Y%m%d%H%M%S"))
        assert epoch == expected
        assert tie == "my-clip-20261022-120000-aaaaaaaa"
        # Unstamped ids sort below every real timestamp.
        assert jobs.job_sort_key("legacy")[0] == float("-inf")
