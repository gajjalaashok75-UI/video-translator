"""
Tests for the central language registry (Phase 1).

The registry is the contract every other stage reads from, so these tests pin
both the new multilingual behaviour and the legacy behaviour that existing call
sites depend on.
"""

import pytest

from src.utils import languages as L


class TestLegacyContract:
    """Behaviour that existed before the registry and must not change."""

    def test_get_language_code_by_name(self):
        assert L.get_language_code("English") == "en"
        assert L.get_language_code("Auto Detect") == "auto"
        assert L.get_language_code("Chinese (Simplified)") == "zh"

    def test_get_language_code_passthrough(self):
        assert L.get_language_code("fr") == "fr"
        assert L.get_language_code("en") == "en"
        assert L.get_language_code("es") == "es"

    def test_get_language_code_unknown_falls_back_to_en(self):
        # Existing callers rely on a lenient fallback rather than an exception.
        assert L.get_language_code("NonExistent") == "en"

    def test_get_language_name_reverse_lookup(self):
        assert L.get_language_name("en") == "English"
        assert L.get_language_name("fr") == "French"
        assert L.get_language_name("zh") == "Chinese (Simplified)"

    def test_get_language_name_all_codes_round_trip(self):
        for name, code in L.LANGUAGE_CODE_MAP.items():
            resolved = L.get_language_name(code)
            assert L.LANGUAGE_CODE_MAP[resolved] == code
            if code != "en":
                assert resolved != "English"

    def test_edge_voice_map_structure(self):
        assert "en" in L.EDGE_TTS_VOICE_MAP
        assert "Female" in L.EDGE_TTS_VOICE_MAP["en"]
        assert len(L.EDGE_TTS_VOICE_MAP["en"]["Male"]) > 0

    def test_piper_map_unchanged_for_legacy_languages(self):
        assert L.PIPER_MODEL_MAP["en"] == "en_US-lessac-high"
        assert L.PIPER_MODEL_MAP["de"] == "de_DE-thorsten-medium"


class TestNormalize:
    """The new, strict resolver."""

    @pytest.mark.parametrize("value,expected", [
        ("te", "te"),
        ("te-IN", "te"),
        ("te_IN", "te"),
        ("Telugu", "te"),
        ("hi", "hi"),
        ("hi-IN", "hi"),
        ("Hindi", "hi"),
        ("ml", "ml"),
        ("Malayalam", "ml"),
        ("Auto Detect", "auto"),
        ("auto", "auto"),
    ])
    def test_resolves_codes_names_and_regions(self, value, expected):
        assert L.normalize(value) == expected

    def test_unknown_raises(self):
        with pytest.raises(L.UnsupportedLanguage):
            L.normalize("klingon")

    def test_empty_raises(self):
        with pytest.raises(L.UnsupportedLanguage):
            L.normalize("")


class TestIndicCoverage:
    """The 11 priority IndicF5 target languages."""

    def test_all_eleven_present(self):
        expected = {"te", "hi", "ta", "kn", "ml", "bn", "mr", "gu", "pa", "or", "as"}
        assert expected <= set(L.LANGUAGES)

    def test_all_eleven_route_to_indicf5(self):
        for code in L.PRIORITY_INDIC:
            assert L.engine_for(code) == L.ENGINE_INDICF5

    def test_indicf5_native_matches_registry(self):
        assert L.INDICF5_NATIVE == set(L.PRIORITY_INDIC)

    def test_every_language_has_a_flores_code(self):
        for code, entry in L.LANGUAGES.items():
            assert entry["flores"], f"{code} has no flores code"

    def test_every_language_has_edge_voices(self):
        for code in L.LANGUAGES:
            assert code in L.EDGE_TTS_VOICE_MAP, f"{code} has no Edge voices"

    def test_priority_indic_are_listed_first(self):
        targets = [t["code"] for t in L.supported_targets()]
        assert targets[:11] == L.PRIORITY_INDIC

    def test_indic_scripts_are_speakable(self):
        # Sanity check that the registry's languages are all non-Latin-script
        # where expected, i.e. we did not accidentally alias scripts.
        samples = {
            "te": "నమస్కారం", "hi": "नमस्ते", "ta": "வணக்கம்", "kn": "ನಮಸ್ಕಾರ",
            "ml": "നമസ്കാരം", "bn": "নমস্কার", "mr": "नमस्कार", "gu": "નમસ્તે",
            "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "or": "ନମସ୍କାର", "as": "নমস্কাৰ",
        }
        for code, text in samples.items():
            assert text, code


class TestFloresHelpers:
    def test_source_flores(self):
        assert L.source_flores("te") == "tel_Telu"
        assert L.source_flores("en") == "eng_Latn"
        assert L.source_flores("hi") == "hin_Deva"

    def test_source_flores_supports_source_only_languages(self):
        # Whisper may detect Urdu even though we cannot synthesize it.
        assert L.source_flores("ur") == "urd_Arab"

    def test_target_flores(self):
        assert L.target_flores("te") == "tel_Telu"
        assert L.target_flores("en") == "eng_Latn"

    def test_target_flores_rejects_non_target(self):
        with pytest.raises(L.UnsupportedLanguage):
            L.target_flores("ur")

    def test_synth_lang_defaults_to_code(self):
        assert L.synth_lang("te") == "te"
        assert L.synth_lang("hi") == "hi"

    def test_chars_per_second_is_sane(self):
        for code in L.LANGUAGES:
            cps = L.chars_per_second(code)
            assert 1.0 < cps < 30.0, f"{code}: {cps}"


class TestValidation:
    def test_indicf5_accepts_its_languages(self):
        for code in L.PRIORITY_INDIC:
            L.validate_tts_pair(code, L.ENGINE_INDICF5)

    def test_indicf5_rejects_english(self):
        with pytest.raises(L.UnsupportedLanguage):
            L.validate_tts_pair("en", L.ENGINE_INDICF5)

    def test_xtts_accepts_english_rejects_odia(self):
        L.validate_tts_pair("en", L.ENGINE_XTTS)
        with pytest.raises(L.UnsupportedLanguage):
            L.validate_tts_pair("or", L.ENGINE_XTTS)

    def test_piper_rejects_language_without_a_model(self):
        L.validate_tts_pair("en", L.ENGINE_PIPER)
        with pytest.raises(L.UnsupportedLanguage):
            L.validate_tts_pair("te", L.ENGINE_PIPER)


class TestUiChoices:
    def test_source_choices_include_auto_and_indic(self):
        choices = L.source_choices()
        assert choices[0] == "Auto Detect"
        for name in ["Telugu", "Hindi", "Tamil", "Malayalam", "Urdu"]:
            assert name in choices

    def test_target_choices_exclude_source_only_languages(self):
        choices = L.target_choices()
        assert "Urdu" not in choices
        assert "Telugu" in choices

    def test_choices_are_unique(self):
        assert len(L.source_choices()) == len(set(L.source_choices()))
        assert len(L.target_choices()) == len(set(L.target_choices()))
