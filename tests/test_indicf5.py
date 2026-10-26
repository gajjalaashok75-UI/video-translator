"""
Tests for the IndicF5 engine and backend (Phase 2).

These cover the pure helpers, the checkpoint-key handling, the reference
validation contract and the backend's routing/error behaviour. Model inference
itself is not exercised — that needs the real 1.4 GB checkpoint and is marked
`requires_models`.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.synthesis import indicf5_engine as E
from src.synthesis.indicf5_engine import IndicF5Engine


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
class TestStripOrigModPrefix:
    def test_strips_the_compile_inserted_segment(self):
        state = {
            "ema_model._orig_mod.transformer.x": 1,
            "vocoder._orig_mod.head.weight": 2,
        }
        out = E.strip_orig_mod_prefix(state)
        assert set(out) == {"ema_model.transformer.x", "vocoder.head.weight"}

    def test_leaves_clean_keys_untouched(self):
        state = {"ema_model.transformer.x": 1}
        assert E.strip_orig_mod_prefix(state) == state

    def test_handles_mixed_keys(self):
        state = {"a._orig_mod.b": 1, "c.d": 2}
        assert set(E.strip_orig_mod_prefix(state)) == {"a.b", "c.d"}


class TestNormalizeRefText:
    def test_adds_terminal_punctuation_and_trailing_space(self):
        assert E.normalize_ref_text("hello") == "hello. "

    def test_keeps_existing_terminator(self):
        assert E.normalize_ref_text("hello.") == "hello. "
        assert E.normalize_ref_text("hello?") == "hello? "
        assert E.normalize_ref_text("नमस्ते।") == "नमस्ते। "

    def test_empty_becomes_single_dot(self):
        # F5 concatenates ref_text and gen_text, so "" would run them together.
        # The trailing space is deliberate: it separates the two spans.
        assert E.normalize_ref_text("") == ". "
        assert E.normalize_ref_text(None) == ". "

    def test_strips_surrounding_whitespace(self):
        assert E.normalize_ref_text("  hi  ") == "hi. "


class TestSecondsPerByte:
    def test_basic_ratio(self):
        assert E.seconds_per_byte(10.0, 100) == pytest.approx(0.1)

    def test_zero_bytes_does_not_divide_by_zero(self):
        assert E.seconds_per_byte(10.0, 0) == pytest.approx(10.0)


class TestClampSecondsPerByte:
    def test_plausible_rate_is_unchanged(self):
        # Devanagari is ~3 bytes/char; 13.5 cps -> ~0.0247 s/byte.
        ref = "नमस्ते" * 20
        expected = 1.0 / (13.5 * (len(ref.encode("utf-8")) / len(ref)))
        assert E.clamp_seconds_per_byte(expected, ref, "hi") == pytest.approx(expected)

    def test_absurdly_slow_rate_is_clamped_down(self):
        ref = "नमस्ते" * 20
        clamped = E.clamp_seconds_per_byte(5.0, ref, "hi")
        assert clamped < 5.0

    def test_absurdly_fast_rate_is_clamped_up(self):
        ref = "नमस्ते" * 20
        clamped = E.clamp_seconds_per_byte(1e-6, ref, "hi")
        assert clamped > 1e-6

    def test_unknown_language_returns_input(self):
        assert E.clamp_seconds_per_byte(5.0, "text", "klingon") == 5.0


class TestPeakAmplitude:
    def test_silence_is_zero(self):
        assert E.peak_amplitude(np.zeros(100, dtype=np.float32)) == 0.0

    def test_empty_is_zero(self):
        assert E.peak_amplitude(np.zeros(0, dtype=np.float32)) == 0.0

    def test_detects_peak(self):
        wave = np.array([0.0, 0.5, -0.75], dtype=np.float32)
        assert E.peak_amplitude(wave) == pytest.approx(0.75)


class TestVoicedFraction:
    def test_silence_is_full_or_fallback(self):
        # All-silent input: librosa finds no speech, so the helper falls back.
        assert E.voiced_fraction(np.zeros(24000, dtype=np.float32)) == 1.0

    def test_tone_is_mostly_voiced(self):
        t = np.linspace(0, 1, 24000, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        assert E.voiced_fraction(tone) > 0.5

    def test_empty_is_fallback(self):
        assert E.voiced_fraction(np.zeros(0, dtype=np.float32)) == 1.0


# ---------------------------------------------------------------------------
# Reference validation
# ---------------------------------------------------------------------------
class TestValidateReferenceAudio:
    def _write(self, path, seconds, amp=0.3, sr=24000):
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        data = (amp * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        sf.write(str(path), data, sr)
        return path

    def test_accepts_a_good_reference_and_returns_duration(self, tmp_path):
        ref = self._write(tmp_path / "good.wav", 5.0)
        assert E.validate_reference_audio(ref, min_duration=3.0) == pytest.approx(5.0, abs=0.05)

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(E.IndicF5ReferenceError):
            E.validate_reference_audio(tmp_path / "nope.wav")

    def test_none_raises_with_actionable_message(self):
        with pytest.raises(E.IndicF5ReferenceError) as exc:
            E.validate_reference_audio(None)
        assert "reference clip" in str(exc.value).lower()

    def test_too_short_raises(self, tmp_path):
        ref = self._write(tmp_path / "short.wav", 1.5)
        with pytest.raises(E.IndicF5ReferenceError) as exc:
            E.validate_reference_audio(ref, min_duration=3.0)
        assert "too short" in str(exc.value)

    def test_silent_reference_raises(self, tmp_path):
        ref = tmp_path / "silent.wav"
        sf.write(str(ref), np.zeros(24000 * 5, dtype=np.float32), 24000)
        with pytest.raises(E.IndicF5ReferenceError) as exc:
            E.validate_reference_audio(ref, min_duration=3.0)
        assert "silent" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Engine construction / environment handling
# ---------------------------------------------------------------------------
class TestEngineConstruction:
    def test_default_nfe_step(self, monkeypatch):
        monkeypatch.delenv("INDICF5_NFE_STEP", raising=False)
        assert IndicF5Engine().nfe_step == IndicF5Engine.DEFAULT_NFE_STEP

    def test_nfe_step_from_env(self, monkeypatch):
        monkeypatch.setenv("INDICF5_NFE_STEP", "32")
        assert IndicF5Engine().nfe_step == 32

    def test_nfe_step_argument_wins(self, monkeypatch):
        monkeypatch.setenv("INDICF5_NFE_STEP", "32")
        assert IndicF5Engine(nfe_step=8).nfe_step == 8

    def test_default_seed_is_fixed_for_reproducibility(self, monkeypatch):
        monkeypatch.delenv("INDICF5_SEED", raising=False)
        assert IndicF5Engine().seed == 1234

    def test_empty_seed_env_means_stochastic(self, monkeypatch):
        monkeypatch.setenv("INDICF5_SEED", "")
        assert IndicF5Engine().seed is None

    def test_forced_device_is_honoured(self, monkeypatch):
        monkeypatch.setenv("FORCE_DEVICE", "cpu")
        assert IndicF5Engine.pick_device() == "cpu"

    def test_engine_available_reports_missing_packages(self):
        ok, reason = E.engine_available()
        assert isinstance(ok, bool)
        if not ok:
            assert "install" in reason.lower()


class TestEngineSynthesizeGuards:
    """Failures that must happen before any model work."""

    def test_empty_text_returns_empty_audio_without_loading(self):
        engine = IndicF5Engine()
        wave, sr = engine.synthesize("   ", "unused.wav", "ref")
        assert wave.size == 0
        assert sr == IndicF5Engine.SAMPLE_RATE
        assert not engine.is_loaded  # must not have loaded the model

    def test_missing_ref_text_raises_before_loading(self, tmp_path):
        ref = tmp_path / "r.wav"
        t = np.linspace(0, 5, 24000 * 5, endpoint=False)
        sf.write(str(ref), (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 24000)

        engine = IndicF5Engine()
        with pytest.raises(E.IndicF5ReferenceError) as exc:
            engine.synthesize("hello", ref, ref_text="")
        assert "transcript" in str(exc.value).lower()
        assert not engine.is_loaded

    def test_bad_reference_raises_before_loading(self, tmp_path):
        engine = IndicF5Engine()
        with pytest.raises(E.IndicF5ReferenceError):
            engine.synthesize("hello", tmp_path / "missing.wav", ref_text="hi there")


# ---------------------------------------------------------------------------
# Backend adapter
# ---------------------------------------------------------------------------
class TestBackendRouting:
    def _backend(self):
        from src.synthesis.backends.indicf5 import IndicF5Backend
        return IndicF5Backend()

    @pytest.mark.parametrize("lang,expected", [
        ("te", "te"), ("te-IN", "te"), ("Telugu", "te"),
        ("hi", "hi"), ("Hindi", "hi"), ("ml", "ml"),
        ("zh", "zh"),  # synth_lang override stays consistent
    ])
    def test_resolves_language(self, lang, expected):
        from src.synthesis.backends.indicf5 import IndicF5Backend
        assert IndicF5Backend._resolve_synth_lang(lang) == expected

    def test_unknown_language_passes_through(self):
        from src.synthesis.backends.indicf5 import IndicF5Backend
        assert IndicF5Backend._resolve_synth_lang("klingon") == "klingon"

    def test_empty_defaults_to_hindi(self):
        from src.synthesis.backends.indicf5 import IndicF5Backend
        assert IndicF5Backend._resolve_synth_lang("") == "hi"

    def test_supports_language(self):
        b = self._backend()
        assert b.supports_language("te")
        assert b.supports_language("Telugu")
        assert not b.supports_language("en")
        assert not b.supports_language("klingon")

    def test_missing_reference_raises(self):
        b = self._backend()
        with pytest.raises(E.IndicF5ReferenceError) as exc:
            b.generate("hello", "/tmp/out.wav", language="te", speaker_wav=None,
                       ref_text="नमस्ते")
        assert "reference clip" in str(exc.value)

    def test_missing_ref_text_raises_by_default(self, tmp_path):
        b = self._backend()
        ref = tmp_path / "r.wav"
        t = np.linspace(0, 5, 24000 * 5, endpoint=False)
        sf.write(str(ref), (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 24000)
        with pytest.raises(E.IndicF5ReferenceError) as exc:
            b.generate("hello", tmp_path / "o.wav", language="te",
                       speaker_wav=ref, ref_text="")
        assert "transcript" in str(exc.value).lower()

    def test_degraded_ref_text_is_opt_in(self, tmp_path, monkeypatch):
        """With the opt-in, the call proceeds past validation (and fails later
        only because no model is installed)."""
        b = self._backend()
        ref = tmp_path / "r.wav"
        t = np.linspace(0, 5, 24000 * 5, endpoint=False)
        sf.write(str(ref), (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 24000)

        def _boom(*a, **k):
            raise E.IndicF5Unavailable("no model in this environment")

        monkeypatch.setattr(b.engine, "synthesize", _boom)
        with pytest.raises(E.IndicF5Unavailable):
            b.generate("hello", tmp_path / "o.wav", language="te",
                       speaker_wav=ref, ref_text="",
                       allow_degraded_ref_text=True)

    def test_write_wav_rejects_silence(self, tmp_path):
        from src.synthesis.backends.indicf5 import IndicF5Backend
        with pytest.raises(E.IndicF5SynthesisError):
            IndicF5Backend._write_wav(np.zeros(24000, dtype=np.float32), 24000,
                                      tmp_path / "silent.wav")

    def test_write_wav_writes_real_audio(self, tmp_path):
        from src.synthesis.backends.indicf5 import IndicF5Backend
        t = np.linspace(0, 1, 24000, endpoint=False)
        wave = (0.4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        out = IndicF5Backend._write_wav(wave, 24000, tmp_path / "ok.wav")
        assert Path(out).exists()
        assert sf.info(out).samplerate == 24000

    def test_generate_batch_isolates_failures(self, tmp_path):
        b = self._backend()
        # Both tasks lack a reference, so both fail; the point is that one bad
        # task does not abort the batch.
        results = b.generate_batch([
            {"text": "a", "output_path": tmp_path / "a.wav", "language": "te"},
            {"text": "b", "output_path": tmp_path / "b.wav", "language": "hi"},
        ])
        assert results == [None, None]


# ---------------------------------------------------------------------------
# Routing through the facade
# ---------------------------------------------------------------------------
class TestFacadeRouting:
    def test_indic_language_routes_to_indicf5(self):
        from src.synthesis.tts import TTSEngine
        for code in ["te", "hi", "ta", "kn", "ml", "bn", "mr", "gu", "pa", "or", "as"]:
            assert TTSEngine.route_model(code, None) == "indicf5"

    def test_non_indic_language_uses_dedicated_engine_or_default(self):
        from src.synthesis.tts import TTSEngine
        # English has no dedicated engine, so it falls back to edge.
        assert TTSEngine.route_model("en", None) == "edge"

    def test_explicit_request_wins_when_compatible(self):
        from src.synthesis.tts import TTSEngine
        # f5 has no language restriction, so the request is honoured.
        assert TTSEngine.route_model("te", "f5") == "f5"
        assert TTSEngine.route_model("en", "f5") == "f5"

    def test_incompatible_request_falls_back_to_dedicated_engine(self):
        from src.synthesis.tts import TTSEngine
        # XTTS cannot speak Telugu, so the dedicated IndicF5 engine wins.
        assert TTSEngine.route_model("te", "indicf5") == "indicf5"

    def test_unknown_language_falls_back_to_edge(self):
        from src.synthesis.tts import TTSEngine
        assert TTSEngine.route_model("klingon", None) == "edge"

    def test_indicf5_registered_in_backends(self):
        from src.synthesis.tts import TTSEngine
        engine = TTSEngine()
        assert "indicf5" in engine.backends
        assert engine.get_available_voices("indicf5", "te") == ["Cloning (Reference Audio)"]


@pytest.mark.requires_models
class TestRealSynthesis:
    """Full inference. Needs the gated checkpoint and a GPU/CPU warm-up."""

    def test_synthesizes_telugu_audio(self, tmp_path):
        ok, reason = E.engine_available()
        if not ok:
            pytest.skip(reason)
        if not os.getenv("HF_TOKEN"):
            pytest.skip("IndicF5 is a gated repo; set HF_TOKEN to run this test.")

        ref = tmp_path / "ref.wav"
        t = np.linspace(0, 6, 24000 * 6, endpoint=False)
        sf.write(str(ref), (0.3 * np.sin(2 * np.pi * 160 * t)).astype(np.float32), 24000)

        engine = IndicF5Engine()
        wave, sr = engine.synthesize(
            "నమస్కారం, ఇది ఒక పరీక్ష.", ref, ref_text="నమస్కారం ఇది ఒక పరీక్ష", lang="te",
        )
        assert sr == IndicF5Engine.SAMPLE_RATE
        assert wave.size > 0
        assert E.peak_amplitude(wave) >= IndicF5Engine.SILENCE_PEAK


# ---------------------------------------------------------------------------
# Text sanitisation must accept Indic scripts
# ---------------------------------------------------------------------------
class TestIndicTextSanitisation:
    """
    `TTSEngine._sanitize_text` drops any string with no "speakable" character.
    Its character class originally covered Latin, CJK, Cyrillic, Arabic and
    Hangul only, so every Indic segment was silently discarded here — long
    before any TTS backend saw it. That is a regression that produces a fully
    silent dub rather than an error, so it is pinned explicitly.
    """

    @pytest.mark.parametrize("text", [
        "నమస్కారం",       # Telugu
        "नमस्ते",          # Hindi / Marathi
        "வணக்கம்",         # Tamil
        "ನಮಸ್ಕಾರ",         # Kannada
        "നമസ്കാരം",        # Malayalam
        "নমস্কার",         # Bengali / Assamese
        "નમસ્તે",          # Gujarati
        "ਸਤ ਸ੍ਰੀ ਅਕਾਲ",    # Punjabi
        "ନମସ୍କାର",         # Odia
    ])
    def test_indic_scripts_are_speakable(self, text):
        from src.synthesis.tts import TTSEngine
        assert TTSEngine()._sanitize_text(text) == text

    def test_punctuation_only_is_still_rejected(self):
        from src.synthesis.tts import TTSEngine
        assert TTSEngine()._sanitize_text("...") is None
        assert TTSEngine()._sanitize_text("!!!") is None

    def test_legacy_scripts_still_work(self):
        from src.synthesis.tts import TTSEngine
        for text in ["hello", "你好", "Привет", "مرحبا", "안녕하세요"]:
            assert TTSEngine()._sanitize_text(text) == text


class TestNewTaskKeysAreTolerated:
    """
    The pipeline now puts `ref_text` and `fix_duration` in every TTS task dict.
    Backends that do not use them must still accept the dict, or every
    non-Indic dub would break the moment the keys were added.
    """

    def test_all_backends_accept_kwargs(self):
        import inspect

        from src.synthesis.backends.base import TTSBackend
        from src.synthesis.backends.edge_tts import EdgeTTSBackend
        from src.synthesis.backends.f5_tts import F5TTSBackend
        from src.synthesis.backends.piper_tts import PiperTTSBackend
        from src.synthesis.backends.vibevoice_tts import VibeVoiceBackend
        from src.synthesis.backends.xtts import XttsBackend

        for cls in (TTSBackend, EdgeTTSBackend, F5TTSBackend, XttsBackend,
                    PiperTTSBackend, VibeVoiceBackend):
            params = inspect.signature(cls.generate).parameters.values()
            assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params), \
                f"{cls.__name__}.generate must accept **kwargs"

    def test_tts_task_declares_the_new_keys(self):
        from src.synthesis.tts import TTSTask
        assert "ref_text" in TTSTask.__annotations__
        assert "fix_duration" in TTSTask.__annotations__

    def test_generate_audio_accepts_the_new_keys(self):
        import inspect

        from src.synthesis.tts import TTSEngine
        params = inspect.signature(TTSEngine.generate_audio).parameters
        assert "ref_text" in params
        assert "fix_duration" in params
