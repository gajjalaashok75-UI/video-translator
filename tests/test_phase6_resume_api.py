"""
Phase 6: the `resume_job` entry point.

Resuming is where durable state either pays off or lies. These tests check the
contract: a resumable job restarts the pipeline with the same job id and the
manifest's recorded configuration, a finished job is a no-op, and a job whose
preconditions no longer hold fails with a reason a person can act on rather
than halfway through a four-hour render.
"""

import pytest

from src.core import jobs
from src.core.manifest import Manifest
from src.core.video_translator import VideoTranslator
from src.utils import config

RATE = 16000


@pytest.fixture

def jobs_dir(tmp_path, monkeypatch):
    target = tmp_path / "jobs"
    monkeypatch.setattr(config, "JOBS_DIR", target)
    target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture

def source_video(tmp_path):
    p = tmp_path / "movie.mp4"
    p.write_bytes(b"\x00" * 4096)
    return p


@pytest.fixture

def vt():
    obj = VideoTranslator.__new__(VideoTranslator)
    from src.core.session import SessionContext
    obj.session = SessionContext()
    obj.cancel_requested = False
    obj.device = "cpu"
    obj._captured = None
    return obj


def _manifest(job, source_video, status="RUNNING"):
    m = Manifest.create(
        job, source_path=str(source_video), source_lang="en", target_lang="te",
        run_config={
            "source_lang": "en", "target_lang": "te", "tts_model_name": "edge",
            "transcription_beam_size": 7, "chunk_duration": 300,
            # Not a real parameter: must be filtered out, not passed through.
            "not_a_real_parameter": "x",
            "hf_token": "must-not-be-forwarded-from-the-manifest",
        })
    m.status = status
    m.save()
    return m


def _capture_process_video(vt, captured, yields=()):
    """Replace `process_video` with a recorder, keeping the real signature."""

    def fake_process_video(video_path, **kwargs):
        captured.append({"video_path": str(video_path), **kwargs})
        for item in yields:
            yield item
    vt.process_video = fake_process_video


class TestResumeJob:

    def test_a_running_job_restarts_the_pipeline_with_the_same_id(
            self, vt, jobs_dir, source_video):
        job = jobs.Job.create(source_path=str(source_video))
        _manifest(job, source_video)
        captured = []
        _capture_process_video(vt, captured)

        list(vt.resume_job(job.job_id))

        assert len(captured) == 1
        call = captured[0]
        assert call["video_path"] == str(source_video)
        assert call["job_id"] == job.job_id
        assert call["resume"] is True

    def test_recorded_configuration_is_replayed(self, vt, jobs_dir, source_video):
        job = jobs.Job.create(source_path=str(source_video))
        _manifest(job, source_video)
        captured = []
        _capture_process_video(vt, captured)

        list(vt.resume_job(job.job_id))

        call = captured[0]
        assert call["tts_model_name"] == "edge"
        assert call["target_lang"] == "te"
        assert call["transcription_beam_size"] == 7
        assert call["chunk_duration"] == 300

    def test_unknown_config_keys_are_not_forwarded(self, vt, jobs_dir, source_video):
        job = jobs.Job.create(source_path=str(source_video))
        _manifest(job, source_video)
        captured = []
        _capture_process_video(vt, captured)

        list(vt.resume_job(job.job_id))

        # A manifest written by another version must not crash the resume.
        assert "not_a_real_parameter" not in captured[0]

    def test_a_persisted_token_is_never_replayed(self, vt, jobs_dir, source_video):
        job = jobs.Job.create(source_path=str(source_video))
        _manifest(job, source_video)
        captured = []
        _capture_process_video(vt, captured)

        list(vt.resume_job(job.job_id, hf_token="fresh-token"))

        assert captured[0]["hf_token"] == "fresh-token"

    def test_a_complete_job_is_a_no_op(self, vt, jobs_dir, source_video):
        job = jobs.Job.create(source_path=str(source_video))
        m = _manifest(job, source_video, status="COMPLETE")
        final = job.stage_dir("final") / "movie_dubbed.mp4"
        final.write_bytes(b"video")
        m.artifacts["final_output"] = str(final)
        m.save()

        captured = []
        _capture_process_video(vt, captured)
        events = list(vt.resume_job(job.job_id))

        assert captured == [], "a finished job must not be re-run"
        assert any(kind == "result" for kind, _ in events)

    def test_an_unknown_job_id_fails_clearly(self, vt, jobs_dir):
        # The message names the path, which is what a person needs to fix it.
        with pytest.raises(jobs.JobError, match="does not exist"):
            list(vt.resume_job("20260101-000000-abcdef"))

    def test_a_missing_manifest_fails_clearly(self, vt, jobs_dir, source_video):
        job = jobs.Job.create(source_path=str(source_video))
        with pytest.raises(jobs.JobError, match="no readable manifest"):
            list(vt.resume_job(job.job_id))

    def test_a_deleted_source_fails_clearly(self, vt, jobs_dir, source_video):
        job = jobs.Job.create(source_path=str(source_video))
        _manifest(job, source_video)
        source_video.unlink()

        with pytest.raises(jobs.JobError, match="which is gone"):
            list(vt.resume_job(job.job_id))

    def test_a_malformed_job_id_is_refused(self, vt, jobs_dir):
        with pytest.raises(jobs.JobError):
            list(vt.resume_job("../escape"))

    def test_a_failed_stage_is_reported_before_resuming(self, vt, jobs_dir, source_video):
        job = jobs.Job.create(source_path=str(source_video))
        m = _manifest(job, source_video)
        m.start_stage("tts", "fp")
        m.fail_stage("tts", "backend died")
        m.save()

        captured = []
        _capture_process_video(vt, captured)
        events = list(vt.resume_job(job.job_id))

        # The resume runs, and the progress line reflects what is left to do.
        assert len(captured) == 1
        logs = " ".join(text for kind, text in events if kind == "log")
        assert "Resuming job" in logs

    def test_completed_stages_survive_the_resume_round_trip(
            self, vt, jobs_dir, source_video):
        job = jobs.Job.create(source_path=str(source_video))
        m = _manifest(job, source_video)
        m.start_stage("extraction", "fp-extraction")
        m.complete_stage("extraction")
        m.start_stage("tts", "fp-tts")
        m.fail_stage("tts", "out of memory")
        m.save()

        captured = []
        _capture_process_video(vt, captured)
        list(vt.resume_job(job.job_id))

        reloaded = Manifest.load(jobs.Job.load(job.job_id, jobs_dir=jobs_dir))
        # The completed stage is still provably current...
        assert reloaded.stage_is_valid("extraction", "fp-extraction")
        # ...and the failed one is not, so it will be redone.
        assert not reloaded.stage_is_valid("tts", "fp-tts")
        assert reloaded.stages["tts"].status == "FAILED"


class TestCancellationIsResumable:
    """
    A cancelled job is stopped, not broken.

    The distinction matters: `resume_job` refuses a COMPLETE job and continues
    anything else, so marking a cancel as FAILED would be a lie the resume path
    would then act on.
    """

    def test_cancel_marks_the_job_stopped_not_failed(self, vt, jobs_dir, source_video):
        job = jobs.Job.create(source_path=str(source_video))
        vt.job = job
        vt.manifest = _manifest(job, source_video)
        vt.manifest.start_stage("tts", "fp")
        vt.cancel_requested = True

        from src.core.video_translator import CancelledError
        with pytest.raises(CancelledError):
            vt._check_cancel()

        assert vt.manifest.status == "CANCELLED"
        # The in-flight stage is partial, so its work is kept and its gap retried.
        assert vt.manifest.stages["tts"].status == "PARTIAL"

    def test_a_cancelled_job_can_be_resumed(self, vt, jobs_dir, source_video):
        job = jobs.Job.create(source_path=str(source_video))
        m = _manifest(job, source_video)
        m.status = "CANCELLED"
        m.start_stage("extraction", "fp-extraction")
        m.complete_stage("extraction")
        m.start_stage("tts", "fp-tts")
        m.partial_stage("tts")
        m.save()

        captured = []
        _capture_process_video(vt, captured)
        list(vt.resume_job(job.job_id))

        assert len(captured) == 1, "a cancelled job must resume"
        assert captured[0]["resume"] is True

    def test_a_cancelled_job_does_not_lose_completed_stages(self, vt, jobs_dir, source_video):
        job = jobs.Job.create(source_path=str(source_video))
        m = _manifest(job, source_video)
        m.status = "CANCELLED"
        m.start_stage("extraction", "fp-extraction")
        m.complete_stage("extraction")
        m.save()

        reloaded = Manifest.load(jobs.Job.load(job.job_id, jobs_dir=jobs_dir))
        assert reloaded.stage_is_valid("extraction", "fp-extraction")
