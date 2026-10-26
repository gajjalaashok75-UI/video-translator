"""
F1: the read-only API contract.

These tests exercise the real FastAPI application through `TestClient`, which
runs the actual routes, response models and serialisation — not mocks. The only
things they fake are the *environment*: a missing package, a missing token, a
missing GPU. That is what capability probing is about, so it must be tested
against the real probe functions with the environment controlled.

The suite must pass on a CPU-only machine with no HF_TOKEN, no model weights and
no optional packages, and it must never download anything. If a test here
triggers a model download, that is a bug in the endpoint, not in the test.
"""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

from src.api import capabilities as caps
from src.api.main import app
from src.api.schemas import ComponentState


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------
def test_health_is_cheap_and_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["api_version"]
    assert body["uptime_seconds"] >= 0


def test_health_does_not_import_the_pipeline():
    """
    `/api/health` must not drag in torch/transformers/the pipeline.

    This is the whole point of the bootstrap: a health check that imports the
    dubbing stack would take tens of seconds and could fail on a machine where
    an optional model is missing.
    """
    import subprocess
    code = (
        "from fastapi.testclient import TestClient\n"
        "from src.api.main import app\n"
        "c = TestClient(app)\n"
        "assert c.get('/api/health').status_code == 200\n"
        "import sys\n"
        "assert 'src.core.video_translator' not in sys.modules, 'pipeline imported'\n"
        "assert 'transformers' not in sys.modules, 'transformers imported'\n"
        "print('CLEAN')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "CLEAN" in proc.stdout


# ---------------------------------------------------------------------------
# /api/system
# ---------------------------------------------------------------------------
def test_system_reports_environment_without_gpu(client):
    r = client.get("/api/system")
    assert r.status_code == 200
    body = r.json()
    assert body["python_version"]
    assert body["platform"]
    assert body["device"] in ("cpu", "cuda")
    # A CPU-only box must still answer, with nulls rather than an error.
    assert isinstance(body["cuda_available"], bool)
    assert isinstance(body["ffmpeg_available"], bool)
    if not body["cuda_available"]:
        assert body["gpu_name"] is None
        assert body["gpu_count"] == 0
    assert body["api_version"]


def test_system_never_leaks_token(client):
    """`/api/system` reports configuration, never credentials."""
    body = client.get("/api/system").json()
    assert "token" not in {k.lower() for k in body}


# ---------------------------------------------------------------------------
# /api/models
# ---------------------------------------------------------------------------
REQUIRED_COMPONENTS = {
    "ffmpeg", "edge_tts", "piper", "indicf5", "xtts", "f5", "vibevoice",
    "voicefixer", "rubberband", "faster_whisper", "demucs", "pyannote",
    "speechbrain", "wav2lip", "gfpgan", "live_portrait",
}


def test_models_covers_the_required_components(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    names = {entry["name"] for entry in r.json()}
    assert REQUIRED_COMPONENTS <= names


def test_models_uses_explicit_states_not_booleans(client):
    r = client.get("/api/models")
    valid = {state.value for state in ComponentState}
    for entry in r.json():
        assert entry["state"] in valid, entry
        assert isinstance(entry["available"], bool)
        assert entry["reason"]
        # `available` must agree with the state, never contradict it.
        assert entry["available"] is (entry["state"] == ComponentState.AVAILABLE.value)


def test_models_reports_missing_packages_as_not_installed(monkeypatch):
    """A package that cannot be imported is `not_installed`, not `unknown`."""
    monkeypatch.setattr(caps, "module_available", lambda name: False)
    status = caps._probe_piper()
    assert status.state == ComponentState.NOT_INSTALLED
    assert status.available is False


def test_models_does_not_download_anything(client, monkeypatch):
    """
    Probing must be network-free.

    Any attempt to reach Hugging Face would show up as a `hf_hub_download` or
    `snapshot_download` call; those are trapped here so the test fails loudly if
    a future probe starts downloading.
    """
    calls = []

    def trap(*args, **kwargs):
        calls.append(args)
        raise AssertionError("capability probing attempted a download")

    for mod_name in ("huggingface_hub", "transformers.utils.hub"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        monkeypatch.setattr(mod, "hf_hub_download", trap, raising=False)
        monkeypatch.setattr(mod, "snapshot_download", trap, raising=False)

    assert client.get("/api/models").status_code == 200
    assert not calls


# ---------------------------------------------------------------------------
# IndicF5: implementation vs weights/access
# ---------------------------------------------------------------------------
def test_indicf5_distinguishes_implementation_from_weights(monkeypatch):
    """
    The three IndicF5 situations must be three distinct states.

    This is the case the user cares most about: "is it broken, or do I just need
    a token?" — the answer has to be actionable.
    """
    # (a) runtime packages missing -> not_installed
    monkeypatch.setattr(caps, "_indicf5_implementation_missing", lambda: ["vocos"])
    status = caps._probe_indicf5()
    assert status.state == ComponentState.NOT_INSTALLED
    assert status.detail["implementation_available"] == "false"

    # (b) runtime present, no token, no cache -> token_required
    monkeypatch.setattr(caps, "_indicf5_implementation_missing", lambda: [])
    monkeypatch.setattr(caps, "hf_cached_file", lambda *a, **k: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    status = caps._probe_indicf5()
    assert status.state == ComponentState.TOKEN_REQUIRED
    assert status.detail["implementation_available"] == "true"
    assert status.detail["weights_cached"] == "false"
    assert "IndicF5" in status.reason

    # (c) runtime present, token present, no cache -> model_not_downloaded
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_not_a_secret")
    status = caps._probe_indicf5()
    assert status.state == ComponentState.MODEL_NOT_DOWNLOADED
    assert status.detail["token_configured"] == "true"

    # (d) cached weights -> available
    monkeypatch.setattr(caps, "hf_cached_file", lambda *a, **k: "/cache/model.safetensors")
    status = caps._probe_indicf5()
    assert status.state == ComponentState.AVAILABLE
    assert status.available is True


def test_indicf5_detail_never_contains_the_token(monkeypatch):
    monkeypatch.setattr(caps, "_indicf5_implementation_missing", lambda: [])
    monkeypatch.setattr(caps, "hf_cached_file", lambda *a, **k: None)
    monkeypatch.setenv("HF_TOKEN", "hf_super_secret_value")
    status = caps._probe_indicf5()
    assert "hf_super_secret_value" not in str(status.model_dump())


def test_hf_cached_file_returns_none_without_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    assert caps.hf_cached_file("ai4bharat/IndicF5", "model.safetensors") is None


def test_hf_cached_file_finds_a_snapshot(tmp_path):
    snapshot = (tmp_path / "models--ai4bharat--IndicF5" / "snapshots" / "abc123")
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")
    import os as _os
    old = _os.environ.get("HF_HUB_CACHE")
    _os.environ["HF_HUB_CACHE"] = str(tmp_path)
    try:
        found = caps.hf_cached_file("ai4bharat/IndicF5", "model.safetensors")
        assert found is not None and found.name == "model.safetensors"
    finally:
        if old is None:
            _os.environ.pop("HF_HUB_CACHE", None)
        else:
            _os.environ["HF_HUB_CACHE"] = old


# ---------------------------------------------------------------------------
# /api/languages
# ---------------------------------------------------------------------------
def test_languages_is_derived_from_the_registry(client):
    from src.utils import languages as registry

    r = client.get("/api/languages")
    assert r.status_code == 200
    body = r.json()

    codes = {lang["code"] for lang in body["languages"]}
    assert set(registry.LANGUAGES) <= codes
    # No invented languages: everything reported must exist in the registry or
    # be a source-only entry the registry knows about.
    allowed = set(registry.LANGUAGES) | set(registry.EXTRA_SOURCE_NAMES)
    assert codes <= allowed


def test_languages_auto_detect_is_a_source_not_a_target(client):
    body = client.get("/api/languages").json()
    assert body["auto_detect_code"] == "auto"
    assert body["sources"][0]["code"] == "auto"
    assert "auto" not in {t["code"] for t in body["targets"]}


def test_languages_targets_put_priority_indic_first(client):
    from src.utils import languages as registry

    body = client.get("/api/languages").json()
    order = [t["code"] for t in body["targets"]]
    assert order[:len(registry.PRIORITY_INDIC)] == list(registry.PRIORITY_INDIC)


def test_languages_keeps_the_two_cloning_concepts_apart(client):
    """
    Hindi can be cloned by architecture (IndicF5 owns it) even where the weights
    are absent. The two fields must not be conflated.
    """
    body = client.get("/api/languages").json()
    hi = next(l for l in body["languages"] if l["code"] == "hi")
    assert hi["dedicated_engine"] == "indicf5"
    assert hi["cloning_engine"] == "indicf5"
    assert hi["cloning_supported_by_architecture"] is True
    # Whatever the machine state, the two fields must be independent booleans.
    assert isinstance(hi["cloning_model_available_on_this_machine"], bool)


def test_languages_marks_source_only_languages_as_not_targets(client):
    body = client.get("/api/languages").json()
    urdu = [l for l in body["languages"] if l["code"] == "ur"]
    if urdu:  # present only when EXTRA_SOURCE_NAMES lists it
        assert urdu[0]["source"] is True
        assert urdu[0]["target"] is False


def test_languages_reports_script_and_edge_support(client):
    body = client.get("/api/languages").json()
    hi = next(l for l in body["languages"] if l["code"] == "hi")
    assert hi["script"] == "Deva"
    assert hi["edge_supported"] is True
    assert hi["synth_lang"] == "hi"
    assert hi["chars_per_second"] > 0


# ---------------------------------------------------------------------------
# /api/capabilities
# ---------------------------------------------------------------------------
def test_capabilities_reports_registry_backends(client):
    from src.api import discovery
    from src.synthesis.capabilities import CAPABILITIES

    body = client.get("/api/capabilities").json()
    default_capabilities = discovery.router_module().default_capabilities
    assert {b["name"] for b in body["translation_backends"]} == {
        c.name for c in default_capabilities()}
    assert {b["name"] for b in body["tts_backends"]} == set(CAPABILITIES)


def test_router_module_avoids_the_translation_package():
    """
    The router must load without `src.translation.__init__`, which imports
    transformers. This is what lets the route matrix work on a minimal install.

    Run in a subprocess so an earlier test in the session cannot have already
    imported transformers and masked the result.
    """
    import subprocess

    code = (
        "from src.api import discovery\n"
        "m = discovery.router_module()\n"
        "assert hasattr(m, 'default_capabilities')\n"
        "assert hasattr(m, 'TranslationRouter')\n"
        "assert [c.name for c in m.default_capabilities()], 'no backends'\n"
        "import sys\n"
        "assert 'transformers' not in sys.modules, 'transformers imported'\n"
        "assert 'src.translation.text_translator' not in sys.modules\n"
        "print('CLEAN')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "CLEAN" in proc.stdout


def test_router_backend_list_matches_the_package():
    """
    The standalone-loaded router must expose the same backends as the normal
    package import, so bypassing `__init__` does not fork the backend list.
    """
    from src.api import discovery

    standalone = [c.name for c in discovery.router_module().default_capabilities()]
    from src.translation.router import default_capabilities as package_fn
    assert standalone == [c.name for c in package_fn()]


def test_capabilities_cloning_engines_match_the_registry(client):
    from src.synthesis.capabilities import CAPABILITIES

    body = client.get("/api/capabilities").json()
    expected = {n for n, c in CAPABILITIES.items() if c.clones}
    assert set(body["cloning_engines"]) == expected


def test_capabilities_presets_come_from_config(client):
    from src.utils import config

    body = client.get("/api/capabilities").json()
    assert [p["name"] for p in body["quality_presets"]] == list(config.QUALITY_PRESETS)
    for preset in body["quality_presets"]:
        assert preset["settings"] == config.QUALITY_PRESETS[preset["name"]]


def test_capabilities_targets_match_the_language_registry(client):
    from src.utils import languages as registry

    body = client.get("/api/capabilities").json()
    assert set(body["targets"]) == set(registry.LANGUAGES)
    assert body["priority_indic"] == list(registry.PRIORITY_INDIC)


def test_capabilities_routes_are_opt_in_and_consistent(client):
    """
    The route matrix is opt-in, so the default response stays fast and free of
    the translation package's heavy imports.
    """
    body = client.get("/api/capabilities").json()
    assert body["routes"] == []
    assert any("include_routes" in note for note in body["notes"])

    body = client.get("/api/capabilities", params={"include_routes": "true"}).json()
    assert body["routes"]
    for route in body["routes"]:
        assert route["source"] and route["target"]
        if route["is_passthrough"]:
            assert route["source"] == route["target"]
            assert route["backend"] is None
        else:
            assert route["backend"]


def test_capabilities_route_matrix_is_cached(client):
    """A second request must not rebuild the matrix."""
    from src.api import discovery

    client.get("/api/capabilities", params={"include_routes": "true"})
    before = dict(discovery._ROUTE_CACHE)
    assert before
    client.get("/api/capabilities", params={"include_routes": "true"})
    assert discovery._ROUTE_CACHE == before


def test_capabilities_default_response_avoids_transformers(client):
    """
    A plain `/api/capabilities` must not import transformers.

    This is the difference between a dashboard endpoint that answers in
    milliseconds and one that spends seconds importing the ML stack.
    """
    import subprocess

    code = (
        "from fastapi.testclient import TestClient\n"
        "from src.api.main import app\n"
        "c = TestClient(app)\n"
        "r = c.get('/api/capabilities')\n"
        "assert r.status_code == 200, r.status_code\n"
        "assert r.json()['routes'] == []\n"
        "import sys\n"
        "assert 'transformers' not in sys.modules, 'transformers imported'\n"
        "assert 'src.translation.text_translator' not in sys.modules\n"
        "print('CLEAN')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "CLEAN" in proc.stdout


def test_capabilities_notes_when_no_backend_is_available(monkeypatch):
    """With nothing installed the route matrix is empty and says why."""
    from src.api import discovery

    states = {name: ComponentState.NOT_INSTALLED for name in caps._PROBES}
    states["cuda_gpu"] = ComponentState.UNAVAILABLE
    body = discovery.capabilities_response(states, include_routes=True)
    assert body.routes == []
    assert any("translation backend" in note for note in body.notes)


# ---------------------------------------------------------------------------
# /api/options
# ---------------------------------------------------------------------------
def test_options_reproduce_the_app_lists(client):
    """
    The lists must match the literals in `app.py`, because during F1 the Gradio
    UI is still the UI of record and the two must not silently diverge.
    """
    body = client.get("/api/options").json()
    assert [o["value"] for o in body["translation_models"]] == [
        "Auto (NLLB-200 for Indic, else selected)",
        "Google Translate (Online, Fast)",
        "NLLB-200 (Local, Indic + 200 Languages)",
        "Tencent HY-MT1.5 (Local, Better Context)",
        "Llama 3.1 8B (Local, Instruct)",
        "ALMA-R 7B (Local, Advanced)",
    ]
    assert [o["value"] for o in body["transcription_models"]] == [
        "Large v3 Turbo (Fast)", "Large v3", "Medium", "Base", "Small"]
    assert [o["value"] for o in body["diarization_backends"]] == [
        "pyannote/SpeechBrain (Default)", "pyannote/Community-1 (Advanced)",
        "NVIDIA NeMo (Advanced)"]
    assert [o["value"] for o in body["lipsync_models"]] == [
        "Wav2Lip-GAN (Low Quality - Fast)",
        "Wav2Lip + GFPGAN (Low Quality - Medium)",
        "LivePortrait (High Quality - Slow)"]
    assert [o["value"] for o in body["ocr_engines"]] == ["PaddleOCR", "EasyOCR"]
    assert [o["value"] for o in body["audio_separators"]] == [
        "Torchaudio HDemucs (Recommended)"]


def test_options_tts_models_match_the_capability_registry(client):
    """The TTS list must not drift from the synthesis capability registry."""
    from src.synthesis.capabilities import CAPABILITIES

    body = client.get("/api/options").json()
    values = {o["value"] for o in body["tts_models"]}
    assert values == set(CAPABILITIES) | {"vibevoice-7b"}


def test_options_offer_every_subtitle_mode(client):
    """The UI builds its control from this list, so it must be complete."""
    body = client.get("/api/options").json()
    assert [o["value"] for o in body["subtitle_modes"]] == [
        "none", "external", "embedded", "burned"]


def test_options_records_its_provenance(client):
    body = client.get("/api/options").json()
    assert body["source"]["translation_models"].startswith("app.py")


def test_options_availability_follows_probes(client, monkeypatch):
    monkeypatch.setattr(caps, "module_available", lambda name: False)
    states = caps.component_states()
    from src.api import options as options_module

    edge = next(o for o in options_module.tts_models(states) if o.value == "edge")
    assert edge.available is False
    assert edge.state == ComponentState.NOT_INSTALLED


# ---------------------------------------------------------------------------
# /api/voices
# ---------------------------------------------------------------------------
def test_voices_returns_edge_voices(client):
    r = client.get("/api/voices", params={"tts_model": "edge", "language": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "hi"
    assert body["cloning"] is False
    assert body["voices"]
    assert all(v.startswith("hi-") for v in body["voices"])


def test_voices_accepts_a_display_name(client):
    r = client.get("/api/voices", params={"tts_model": "edge", "language": "Hindi"})
    assert r.status_code == 200
    assert r.json()["language"] == "hi"


def test_voices_marks_cloning_backends(client):
    r = client.get("/api/voices", params={"tts_model": "indicf5", "language": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["cloning"] is True
    assert body["note"]


def test_voices_rejects_an_unknown_backend(client):
    r = client.get("/api/voices", params={"tts_model": "nope", "language": "hi"})
    assert r.status_code == 400
    assert "nope" in r.json()["error"]["message"]


def test_voices_rejects_an_unknown_language(client):
    r = client.get("/api/voices", params={"tts_model": "edge", "language": "zz"})
    assert r.status_code == 400
    assert "zz" in r.json()["error"]["message"]


def test_voices_rejects_auto_as_a_target(client):
    r = client.get("/api/voices", params={"tts_model": "edge", "language": "auto"})
    assert r.status_code == 400


def test_voices_rejects_a_source_only_language(client):
    """Urdu can be transcribed but not spoken, so it has no voices."""
    r = client.get("/api/voices", params={"tts_model": "edge", "language": "ur"})
    assert r.status_code == 400
    assert "source-only" in r.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
def test_no_endpoint_discloses_the_token(client, monkeypatch):
    """
    With a token configured, no read-only endpoint may echo it.

    The token is deliberately a distinctive string so a substring match cannot
    pass by accident.
    """
    secret = "hf_SECRET_CANARY_9f3a2b7c1d"
    monkeypatch.setenv("HF_TOKEN", secret)
    for path in ("/api/health", "/api/system", "/api/models", "/api/languages",
                 "/api/capabilities", "/api/options",
                 "/api/voices?tts_model=edge&language=hi"):
        response = client.get(path)
        assert secret not in response.text, f"{path} leaked the token"
        assert "SECRET_CANARY" not in response.text, f"{path} leaked the token"


def test_models_does_not_expose_absolute_paths(client):
    """
    Expected asset locations are reported relative to the repo, so a response
    does not disclose the deployment's filesystem layout.
    """
    body = client.get("/api/models").json()
    for entry in body:
        for value in entry.get("detail", {}).values():
            assert not value.startswith("/"), entry


def test_cors_is_restricted_to_the_dev_origin(client):
    r = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"

    r = client.get("/api/health", headers={"Origin": "https://evil.example.com"})
    assert r.headers.get("access-control-allow-origin") != "https://evil.example.com"


def test_openapi_schema_is_published(client):
    """The contract must be discoverable for the TypeScript client."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    for path in ("/api/health", "/api/system", "/api/models", "/api/languages",
                 "/api/capabilities", "/api/options", "/api/voices"):
        assert path in paths
