"""
Phase 6: translation persistence.

The promise under test: a translation, once produced, is written to the job's
content-addressed store and survives the process that made it. A new process
reuses the lines whose every input is unchanged and translates only the rest.

Nothing here relies on an in-memory cache. Every "new process" is a freshly
constructed `Translator` and `ArtifactCache` reading the job directory from
disk, which is the only thing that actually crosses a process boundary.
"""

import json

import pytest

from src.core import jobs
from src.core.cache import ArtifactCache, translation_key
from src.core.video_translator import VideoTranslator
from src.translation.text_translator import Translator, TRANSLATION_FLUSH_EVERY
from src.utils import config


class RecordingNLLB:
    """
    A stand-in translation backend that records what it was asked to translate.

    That recording is the measurement: a resumed run that re-translates a line
    shows up as an extra entry in `calls`. `fail_texts` model a line the backend
    could not produce; `crash_texts` model the process dying mid-run.
    """

    def __init__(self, fail_texts=(), crash_texts=()):
        self.calls = []
        self.fail_texts = set(fail_texts)
        self.crash_texts = set(crash_texts)

    def is_available(self):
        return True

    def unload(self):
        pass

    def translate_batch(self, texts, src_flores, tgt_flores, token=None, **kw):
        for t in texts:
            if t in self.crash_texts:
                # A BaseException, not an Exception: this models the process
                # going away, which no per-chunk handler should swallow.
                raise KeyboardInterrupt("simulated process death")
        self.calls.append(list(texts))
        if any(t in self.fail_texts for t in texts):
            raise RuntimeError("backend could not translate")
        return [f"{tgt_flores}:{t}" for t in texts]


def _job(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    monkeypatch.setattr(config, "JOBS_DIR", jobs_dir)
    # Isolate the fallback temp cache too, so a test can prove the job store is
    # what is actually being read.
    temp = tmp_path / "temp"
    temp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "TEMP_DIR", temp)
    src = tmp_path / "movie.mp4"
    if not src.exists():
        src.write_bytes(b"\x00" * 4096)
    return jobs.Job.create(source_path=str(src))


def _segments(texts):
    return [{"text": t, "start": float(i), "end": float(i) + 1.0,
             "speaker_id": "SPEAKER_00"} for i, t in enumerate(texts)]


def _fresh_translator(job, backend):
    """A new Translator + new ArtifactCache over the same job directory."""
    return Translator(nllb_backend=backend, job_cache=ArtifactCache(job))


def _run(translator, segments, target="te", source="en"):
    return translator.translate_segments(segments, target, model="nllb",
                                         source_lang=source)


def _store(job):
    return ArtifactCache(job)._load_translations()


# ---------------------------------------------------------------------------
# The headline contract
# ---------------------------------------------------------------------------
class TestTranslationSurvivesRestart:

    def test_translate_persist_destroy_resume(self, tmp_path, monkeypatch):
        """
        translate -> persist -> destroy -> reconstruct -> resume.

        Segment 3 fails in the first process, so it is the only one the second
        process has to translate.
        """
        job = _job(tmp_path, monkeypatch)
        texts = ["one", "two", "three"]
        segs = _segments(texts)

        first = RecordingNLLB(fail_texts={"three"})
        tr = _fresh_translator(job, first)
        out = _run(tr, segs)
        assert out[0]["translated_text"] and out[1]["translated_text"]
        assert out[2]["translated_text"] == ""  # failed, deliberately not cached

        # Everything that succeeded is on disk, under content-addressed keys.
        persisted = _store(job)
        assert len(persisted) == 2

        # New process: new backend, new translator, new cache object. The only
        # thing carried over is the job directory.
        del tr, first
        second = RecordingNLLB()
        resumed = _fresh_translator(job, second)
        out2 = _run(resumed, segs)

        assert out2[0]["translation"]["status"] == "cached"
        assert out2[1]["translation"]["status"] == "cached"
        assert out2[2]["translation"]["status"] == "ok"
        # Only the missing line was sent to the backend.
        assert second.calls == [["three"]]

    def test_completed_lines_are_reused_and_only_the_gap_is_retried(
            self, tmp_path, monkeypatch):
        """
        segment 1 COMPLETE, segment 2 COMPLETE, segment 3 crashes.

        NEW PROCESS: 1 and 2 are cache hits, 3 is retried.
        """
        import src.translation.text_translator as tt
        monkeypatch.setattr(tt, "TRANSLATION_FLUSH_EVERY", 2)

        job = _job(tmp_path, monkeypatch)
        segs = _segments(["alpha", "beta", "gamma"])

        # First process: 1 and 2 succeed and are flushed; 3 crashes the run.
        dying = RecordingNLLB(crash_texts={"gamma"})
        tr = _fresh_translator(job, dying)
        with pytest.raises(KeyboardInterrupt):
            _run(tr, segs)

        persisted = _store(job)
        assert len(persisted) == 2, "the two finished lines must be durable"
        assert translation_key("gamma", "en", "te", "nllb",
                               {"optimize": False, "concise": "normal"}) not in persisted

        del tr, dying

        # New process.
        second = RecordingNLLB()
        resumed = _fresh_translator(job, second)
        out = _run(resumed, segs)

        assert out[0]["translation"]["status"] == "cached"
        assert out[1]["translation"]["status"] == "cached"
        assert out[2]["translation"]["status"] == "ok"
        assert second.calls == [["gamma"]], "only the unfinished line is retried"

    def test_partial_completion_at_scale(self, tmp_path, monkeypatch):
        """
        A crash after most of a long job keeps everything already translated.

        120 lines, checkpoint every 50. The crash lands in the third block, so
        the first two blocks (100 lines) must survive.
        """
        job = _job(tmp_path, monkeypatch)
        texts = [f"line {i}" for i in range(120)]
        segs = _segments(texts)

        dying = RecordingNLLB(crash_texts={"line 100"})
        tr = _fresh_translator(job, dying)
        with pytest.raises(KeyboardInterrupt):
            _run(tr, segs)

        persisted = _store(job)
        assert len(persisted) == 100, (
            "every block written before the crash must survive; "
            f"got {len(persisted)}")

        del tr, dying

        second = RecordingNLLB()
        resumed = _fresh_translator(job, second)
        _run(resumed, segs)

        # The 100 cached lines were not re-sent; only the final 20 were.
        sent = [t for call in second.calls for t in call]
        assert sent == texts[100:]

    def test_no_in_memory_dependence(self, tmp_path, monkeypatch):
        """A fresh translator starts with an empty in-memory cache and still hits."""
        job = _job(tmp_path, monkeypatch)
        segs = _segments(["solo"])

        tr = _fresh_translator(job, RecordingNLLB())
        _run(tr, segs)

        second = RecordingNLLB()
        resumed = _fresh_translator(job, second)
        assert resumed.cache == {}, "the temp cache must not be involved"
        _run(resumed, segs)
        assert second.calls == []


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------
class TestTranslationInvalidation:

    def test_changing_one_source_line_invalidates_only_that_line(
            self, tmp_path, monkeypatch):
        job = _job(tmp_path, monkeypatch)
        segs = _segments(["a", "b", "c"])

        _run(_fresh_translator(job, RecordingNLLB()), segs)
        assert len(_store(job)) == 3

        segs[1]["text"] = "b CHANGED"
        second = RecordingNLLB()
        _run(_fresh_translator(job, second), segs)

        assert second.calls == [["b CHANGED"]], "only the edited line is re-translated"

    def test_changing_target_language_does_not_reuse(self, tmp_path, monkeypatch):
        job = _job(tmp_path, monkeypatch)
        segs = _segments(["hello", "world"])

        _run(_fresh_translator(job, RecordingNLLB()), segs, target="te")

        second = RecordingNLLB()
        out = _run(_fresh_translator(job, second), segs, target="hi")

        assert all(o["translation"]["status"] == "ok" for o in out)
        assert [t for call in second.calls for t in call] == ["hello", "world"]

    def test_changing_the_backend_does_not_reuse(self, tmp_path, monkeypatch):
        job = _job(tmp_path, monkeypatch)
        segs = _segments(["hello"])

        _run(_fresh_translator(job, RecordingNLLB()), segs)

        # Same text and languages, different backend token.
        tr = _fresh_translator(job, RecordingNLLB())
        out = tr.translate_segments(segs, "te", model="google", source_lang="en")
        # Google is unavailable here (no network), so it fails rather than
        # reusing the nllb result. Either way it must not be reported "cached".
        assert out[0]["translation"]["status"] != "cached"

    def test_changing_concise_setting_does_not_reuse(self, tmp_path, monkeypatch):
        job = _job(tmp_path, monkeypatch)
        segs = _segments(["hello"])

        _run(_fresh_translator(job, RecordingNLLB()), segs)
        second = RecordingNLLB()
        tr = _fresh_translator(job, second)
        tr.translate_segments(segs, "te", model="nllb", source_lang="en",
                              concise="very_concise")
        assert second.calls == [["hello"]]


# ---------------------------------------------------------------------------
# Production wiring
# ---------------------------------------------------------------------------
class TestProductionWiring:

    def test_process_video_hands_the_job_cache_to_the_translator(
            self, tmp_path, monkeypatch):
        """The real wiring point: a resume gives the translator the job store."""
        monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00" * 4096)

        vt = VideoTranslator.__new__(VideoTranslator)
        from src.core.session import SessionContext
        vt.session = SessionContext()
        vt.translator = Translator(nllb_backend=RecordingNLLB())
        vt.cancel_requested = False
        vt.job = vt.manifest = vt.cache = None

        # Stop at the dispatch boundary: the wiring we care about happens before
        # any pipeline stage runs.
        monkeypatch.setattr(vt, "_process_pipeline",
                            lambda **kw: iter(()))

        list(vt.process_video(str(video), resume=True))

        assert vt.cache is not None
        assert vt.translator.job_cache is vt.cache, (
            "the translator must be given the job's ArtifactCache")

    def test_without_resume_the_translator_keeps_no_job_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00" * 4096)

        vt = VideoTranslator.__new__(VideoTranslator)
        from src.core.session import SessionContext
        vt.session = SessionContext()
        vt.translator = Translator(nllb_backend=RecordingNLLB())
        vt.translator.job_cache = ArtifactCache  # pretend a previous run left it
        vt.cancel_requested = False
        vt.job = vt.manifest = vt.cache = None

        monkeypatch.setattr(vt, "_process_pipeline", lambda **kw: iter(()))
        list(vt.process_video(str(video), resume=False))

        assert vt.translator.job_cache is None, (
            "a non-resume run must not point the translator at a job store")


class TestConfigDriftFailSafe:

    def test_resuming_with_changed_settings_is_refused(self, tmp_path, monkeypatch):
        """
        `process_video(..., resume=True)` with different settings must not
        silently reuse the previous run's output.

        Reopening a job keeps the stored configuration and fingerprints are
        computed from it, so without this check the changed settings would be
        ignored. `resume_job` is the supported way to continue a job as it was.
        """
        monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00" * 4096)

        def _fresh():
            vt = VideoTranslator.__new__(VideoTranslator)
            from src.core.session import SessionContext
            vt.session = SessionContext()
            vt.translator = Translator(nllb_backend=RecordingNLLB())
            vt.cancel_requested = False
            vt.job = vt.manifest = vt.cache = None
            monkeypatch.setattr(vt, "_process_pipeline", lambda **kw: iter(()))
            return vt

        job_id = jobs.Job.create(source_path=str(video)).job_id

        # First run records the configuration.
        list(_fresh().process_video(str(video), resume=True, job_id=job_id,
                                    target_lang="te"))

        # A resume asking for a different target language must refuse.
        with pytest.raises(jobs.JobError):
            list(_fresh().process_video(str(video), resume=True, job_id=job_id,
                                        target_lang="hi"))

    def test_resuming_with_the_same_settings_is_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00" * 4096)

        def _fresh():
            vt = VideoTranslator.__new__(VideoTranslator)
            from src.core.session import SessionContext
            vt.session = SessionContext()
            vt.translator = Translator(nllb_backend=RecordingNLLB())
            vt.cancel_requested = False
            vt.job = vt.manifest = vt.cache = None
            monkeypatch.setattr(vt, "_process_pipeline", lambda **kw: iter(()))
            return vt

        job_id = jobs.Job.create(source_path=str(video)).job_id
        list(_fresh().process_video(str(video), resume=True, job_id=job_id,
                                    target_lang="te"))
        # Same settings: no refusal.
        list(_fresh().process_video(str(video), resume=True, job_id=job_id,
                                    target_lang="te"))


class TestEmptyResultsAreNeverPersisted:
    """
    A failed line must not be written to the cache as an empty translation.

    This is what makes a re-run able to recover: if an outage persisted ""
    against a line's key, every later run would read that "" back as a hit and
    never retry the line, so the outage would become permanent.
    """

    def test_a_failed_line_is_not_cached_as_empty(self, tmp_path, monkeypatch):
        job = _job(tmp_path, monkeypatch)
        segs = _segments(["alpha", "beta"])

        # The backend cannot translate "alpha"; "beta" is fine.
        tr = _fresh_translator(job, RecordingNLLB(fail_texts={"alpha"}))
        out = _run(tr, segs)

        assert out[0]["translated_text"] == ""
        assert out[1]["translated_text"].endswith(":beta")

        # Nothing empty is on disk, and the failed line has no entry at all.
        persisted = _store(job)
        assert all(v for v in persisted.values()), "no empty value may persist"
        failed_key = translation_key("alpha", "en", "te", "nllb",
                                     {"optimize": False, "concise": "normal"})
        assert failed_key not in persisted

    def test_the_failed_line_is_retried_on_a_new_process(self, tmp_path, monkeypatch):
        job = _job(tmp_path, monkeypatch)
        segs = _segments(["alpha", "beta"])

        _run(_fresh_translator(job, RecordingNLLB(fail_texts={"alpha"})), segs)

        # New process, backend healthy: the previously failed line must be sent.
        second = RecordingNLLB()
        out = _run(_fresh_translator(job, second), segs)

        sent = [t for call in second.calls for t in call]
        assert "alpha" in sent, "the failed line must be retried, not read back"
        assert out[0]["translated_text"].endswith(":alpha")
