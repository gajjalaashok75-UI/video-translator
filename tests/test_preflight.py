"""
A8: preflight checks and the durable result summary.

Both endpoints answer "what will this render do / what did it do", so they are
tested against the real application, the real registries and the real manifest —
not mocks. The only environment controlled is the capability states, which is
exactly what these endpoints reason about, so they are injected rather than
faked at a higher level.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import preflight
from src.api.main import app
from src.api.schemas import ComponentState, JobCreateRequest
from src.core import jobs as joblib
from src.core.manifest import Manifest
from src.utils import config


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The real app, with job and upload directories isolated."""
    from src.api import settings
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(settings, "UPLOADS_DIR", tmp_path / "uploads")
    (tmp_path / "jobs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)
    with TestClient(app) as c:
        yield c


def _all_available() -> dict:
    """Every probed component present, so only the request decides readiness."""
    states = {name: ComponentState.AVAILABLE for name in preflight_states()}
    states["cuda_gpu"] = ComponentState.AVAILABLE
    return states


def preflight_states() -> list[str]:
    from src.api import capabilities as caps
    return list(caps._PROBES)


def _req(**overrides) -> JobCreateRequest:
    base = dict(upload_id="up-abc", source_lang="en", target_lang="hi",
                tts_model_name="edge", audio_model_name="Torchaudio HDemucs (Recommended)",
                transcription_model_name="Large v3 Turbo (Fast)",
                translation_model_name="Google Translate (Online, Fast)")
    base.update(overrides)
    return JobCreateRequest(**base)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
class TestPreflightSeverity:

    def test_a_ready_render_has_no_blocking_checks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(preflight, "_media_checks",
                            lambda request: [])
        result = preflight.build_preflight(_req(), _all_available())
        assert result.ready is True
        assert result.blocking == []

    def test_an_unsupported_tts_pair_blocks(self, tmp_path, monkeypatch):
        """The most common Indic failure: a cloning backend that cannot speak
        the target. This is blocking, not a warning."""
        monkeypatch.setattr(preflight, "_media_checks", lambda request: [])
        # IndicF5 speaks only the Indic set, so an English target is impossible.
        result = preflight.build_preflight(
            _req(target_lang="en", tts_model_name="indicf5"),
            _all_available())
        assert result.ready is False
        blocking = {c.id: c for c in result.blocking}
        assert "tts_language" in blocking
        assert blocking["tts_language"].field == "tts_model_name"

    def test_an_unknown_language_blocks(self, monkeypatch):
        monkeypatch.setattr(preflight, "_media_checks", lambda request: [])
        result = preflight.build_preflight(
            _req(target_lang="klingon"), _all_available())
        assert result.ready is False
        assert any(c.id == "languages" for c in result.blocking)

    def test_an_unavailable_required_backend_blocks(self, monkeypatch):
        monkeypatch.setattr(preflight, "_media_checks", lambda request: [])
        states = _all_available()
        states["google_translate"] = ComponentState.NOT_INSTALLED
        result = preflight.build_preflight(_req(), states)
        assert result.ready is False
        assert any(c.id == "option_translation_model_name"
                   for c in result.blocking)

    def test_an_unavailable_optional_feature_is_only_a_warning(self, monkeypatch):
        """
        A feature that is off must not block: the user should learn the option
        exists, not be stopped by a feature they did not select.
        """
        monkeypatch.setattr(preflight, "_media_checks", lambda request: [])
        states = _all_available()
        states["wav2lip"] = ComponentState.NOT_INSTALLED
        result = preflight.build_preflight(_req(enable_lipsync=False), states)
        assert result.ready is True
        assert any(c.id == "option_lipsync_model_name"
                   for c in result.warnings)

    def test_selecting_the_unavailable_feature_makes_it_blocking(self, monkeypatch):
        monkeypatch.setattr(preflight, "_media_checks", lambda request: [])
        states = _all_available()
        states["wav2lip"] = ComponentState.NOT_INSTALLED
        result = preflight.build_preflight(_req(enable_lipsync=True), states)
        assert result.ready is False
        assert any(c.id == "option_lipsync_model_name"
                   for c in result.blocking)

    def test_an_unavailable_enhancement_feature_blocks_when_enabled(self, monkeypatch):
        monkeypatch.setattr(preflight, "_media_checks", lambda request: [])
        states = _all_available()
        states["voicefixer"] = ComponentState.NOT_INSTALLED
        result = preflight.build_preflight(
            _req(enable_audio_enhancement=True), states)
        assert result.ready is False
        assert any(c.id == "feature_enable_audio_enhancement"
                   for c in result.blocking)

    def test_a_disabled_enhancement_is_not_reported(self, monkeypatch):
        monkeypatch.setattr(preflight, "_media_checks", lambda request: [])
        states = _all_available()
        states["voicefixer"] = ComponentState.NOT_INSTALLED
        result = preflight.build_preflight(
            _req(enable_audio_enhancement=False), states)
        assert all(c.id != "feature_enable_audio_enhancement"
                   for c in result.checks)

    def test_an_impossible_speaker_range_blocks(self, monkeypatch):
        monkeypatch.setattr(preflight, "_media_checks", lambda request: [])
        result = preflight.build_preflight(
            _req(min_speakers=5, max_speakers=2), _all_available())
        assert result.ready is False
        assert any(c.id == "output" for c in result.blocking)

    def test_an_unknown_subtitle_mode_blocks(self, monkeypatch):
        monkeypatch.setattr(preflight, "_media_checks", lambda request: [])
        result = preflight.build_preflight(
            _req(subtitle_mode="laser"), _all_available())
        assert result.ready is False
        assert any(c.id == "output" for c in result.blocking)

    def test_an_unknown_option_value_blocks(self, monkeypatch):
        monkeypatch.setattr(preflight, "_media_checks", lambda request: [])
        result = preflight.build_preflight(
            _req(tts_model_name="Not A Real Backend"), _all_available())
        assert result.ready is False
        assert any(c.id == "option_tts_model_name" for c in result.blocking)


class TestPreflightEndpoint:

    def test_the_endpoint_answers_without_a_job_directory(self, client):
        """Preflight must be usable before anything is staged for real."""
        r = client.post("/api/jobs/preflight",
                        json={"upload_id": "up-missing", "target_lang": "hi"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ready"] is False
        assert any(c["id"] in ("video", "audio") for c in body["blocking"])

    def test_the_endpoint_is_advisory_and_creates_nothing(self, client, tmp_path):
        before = sorted(p.name for p in (tmp_path / "jobs").iterdir())
        client.post("/api/jobs/preflight",
                    json={"upload_id": "up-missing", "target_lang": "hi"})
        after = sorted(p.name for p in (tmp_path / "jobs").iterdir())
        assert before == after

    def test_the_response_model_is_the_published_shape(self, client):
        r = client.post("/api/jobs/preflight",
                        json={"upload_id": "up-x", "target_lang": "hi"})
        body = r.json()
        assert set(body) == {"ready", "checks", "blocking", "warnings"}
        check = body["checks"][0]
        assert set(check) == {"id", "label", "severity", "detail", "field"}


# ---------------------------------------------------------------------------
# Durable result summary
# ---------------------------------------------------------------------------
@pytest.fixture
def job(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    (tmp_path / "jobs").mkdir(parents=True, exist_ok=True)
    source = tmp_path / "movie.mp4"
    source.write_bytes(b"\x00" * 2048)
    j = joblib.Job.create(source_path=str(source))
    j._test_source = source  # the staged copy lives in the job's source dir
    return j


def _manifest(job, **config_overrides) -> Manifest:
    run_config = {
        "source_lang": "en", "target_lang": "hi",
        "translation_model_name": "Google Translate (Online, Fast)",
        "context_model_name": "Tencent HY-MT1.5 (Local, Better Context)",
        "optimize_translation": False, "enable_diarization": False,
        "enable_lipsync": False, "tts_model_name": "edge", "tts_voice": None,
        "subtitle_mode": "external", "lipsync_model_name": None,
    }
    run_config.update(config_overrides)
    return Manifest.create(job, source_path=str(job._test_source),
                           source_lang="en", target_lang="hi",
                           run_config=run_config)


class TestJobSummary:

    def test_it_describes_the_configured_render(self, job):
        from src.api import summary
        m = _manifest(job, enable_diarization=True, enable_lipsync=True,
                      lipsync_model_name="Wav2Lip-GAN (Low Quality - Fast)",
                      tts_voice="hi-IN-SwaraNeural")
        m.stages["diarization"].detail = {"speakers": 3}
        s = summary.build_summary(m, "completed")

        assert s.source_language == "en"
        assert s.target_language == "hi"
        assert s.target_language_name == "Hindi"
        assert s.voice_mode == "preserve_speaker"
        assert s.speakers == 3
        assert s.lipsync is True
        assert s.lipsync_model == "Wav2Lip-GAN (Low Quality - Fast)"
        assert s.tts_voice == "hi-IN-SwaraNeural"
        assert s.state == "completed"

    def test_diarization_off_means_a_standard_voice(self, job):
        from src.api import summary
        s = summary.build_summary(_manifest(job), "completed")
        assert s.voice_mode == "standard"
        assert s.speakers == 0
        assert s.diarization is False

    def test_the_context_backend_is_only_reported_when_used(self, job):
        from src.api import summary
        off = summary.build_summary(_manifest(job), "completed")
        assert off.context_backend is None
        assert off.optimize_translation is False

        on = summary.build_summary(
            _manifest(job, optimize_translation=True), "completed")
        assert on.context_backend is not None

    def test_the_duration_comes_from_the_sync_stage(self, job):
        from src.api import summary
        m = _manifest(job)
        m.stages["sync"].detail = {"duration": 42.5}
        assert summary.build_summary(m, "completed").duration_seconds == 42.5

    def test_a_missing_duration_is_none_not_zero(self, job):
        from src.api import summary
        s = summary.build_summary(_manifest(job), "completed")
        assert s.duration_seconds is None

    def test_the_subtitle_count_comes_from_the_recorded_artifacts(self, job):
        from src.api import summary
        m = _manifest(job)
        m.artifacts["subtitle_count"] = 17
        assert summary.build_summary(m, "completed").subtitle_count == 17

    def test_it_survives_a_reload_from_disk(self, job):
        from src.api import summary
        m = _manifest(job, enable_diarization=True)
        m.stages["diarization"].detail = {"speakers": 2}
        m.save()

        reloaded = Manifest.load(job)
        s = summary.build_summary(reloaded, "completed")
        assert s.voice_mode == "preserve_speaker"
        assert s.speakers == 2

    def test_the_unknown_language_name_never_raises(self, job):
        """A summary is a read model: a bad code must degrade, not 500."""
        from src.api import summary
        m = _manifest(job)
        m.source["language"] = "zz"
        s = summary.build_summary(m, "completed")
        assert s.source_language == "zz"
        assert s.source_language_name is None
