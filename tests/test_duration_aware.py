"""
Phase 5 tests: duration-aware translation (layer 1).

The load-bearing property is that shortening can only ever improve timing, never
corrupt dialogue: every candidate that changes script, grows, empties out, drops
numerals or drifts too far from the original is rejected and the original
translation is kept.
"""

import pytest

from src.translation.duration_aware import (
    SpeechShortener,
    estimate_speech_duration,
    predict_ratio,
    rule_based_shorten,
    should_shorten,
    validate_shortening,
)
from src.utils import config


class TestValidation:
    def test_shorter_same_script_is_accepted(self):
        ok, reason = validate_shortening(
            "Well, I think that we should really go now, you know",
            "I think we should go now")
        assert ok, reason

    def test_empty_is_rejected(self):
        assert validate_shortening("hello there", "")[0] is False

    def test_not_shorter_is_rejected(self):
        ok, reason = validate_shortening("go now", "go now please")
        assert not ok and reason == "not_shorter"

    def test_script_change_is_rejected(self):
        """An LLM that translated instead of shortening must not slip through."""
        ok, reason = validate_shortening("This is a sentence", "ఇది ఒక వాక్యం")
        assert not ok
        assert reason.startswith("script_changed")

    def test_dropped_numeral_is_rejected(self):
        ok, reason = validate_shortening(
            "We meet at 7 and pay 500 rupees", "We meet and pay rupees")
        assert not ok and reason == "dropped_numerals"

    def test_kept_numeral_is_accepted(self):
        ok, _ = validate_shortening("We meet at 7 o'clock sharp",
                                    "Meet at 7")
        assert ok

    def test_new_words_are_rejected(self):
        """A candidate made of words that were never in the original is a
        hallucination or a re-translation, not a shortening."""
        ok, reason = validate_shortening(
            "The quick brown fox jumps over the lazy dog",
            "zxqv")
        assert not ok
        assert reason.startswith("new_words")

    def test_identical_is_rejected_as_not_shorter(self):
        assert validate_shortening("same text", "same text")[0] is False


class TestRuleBasedShorten:
    def test_removes_english_fillers(self):
        out = rule_based_shorten("Well, you know, I mean, it is fine", "en")
        assert "you know" not in out
        assert "it is fine" in out

    def test_keeps_text_when_nothing_to_remove(self):
        text = "The meeting starts at noon"
        assert rule_based_shorten(text, "en") == text

    def test_result_always_passes_validation(self):
        for text in ["Well, you know, this is important",
                     "अच्छा, यह जरूरी है", "అంటే, ఇది ముఖ్యం"]:
            out = rule_based_shorten(text, "en")
            if out != text:
                assert validate_shortening(text, out)[0]

    def test_empty_input_is_safe(self):
        assert rule_based_shorten("", "en") == ""


class TestDurationEstimation:
    def test_estimate_scales_with_length(self):
        short = estimate_speech_duration("hi", "en")
        long = estimate_speech_duration("hi there friend how are you", "en")
        assert long > short

    def test_source_evidence_beats_the_table(self):
        """Same text, longer original window => longer expected duration."""
        text = "ఇది ఒక వాక్యం"
        small = estimate_speech_duration(text, "te", source_text="Short.",
                                         slot=1.0, source_lang="en")
        big = estimate_speech_duration(text, "te", source_text="Short.",
                                       slot=4.0, source_lang="en")
        assert big > small

    def test_ratio_of_one_when_text_matches_window(self):
        # 2s window, English cps ~15.5 => ~31 chars fits.
        text = "x" * 31
        assert predict_ratio(text, "en", 2.0) == pytest.approx(1.0, abs=0.05)

    def test_zero_window_is_neutral(self):
        assert predict_ratio("anything", "en", 0.0) == 1.0

    def test_should_shorten_only_on_clear_overrun(self):
        # A 2s window cannot hold 200 characters.
        assert should_shorten("x" * 200, "en", 2.0) is True
        assert should_shorten("hi", "en", 5.0) is False

    def test_trigger_is_configurable(self):
        text = "x" * 40
        assert should_shorten(text, "en", 2.0, trigger=1.01) is True
        assert should_shorten(text, "en", 2.0, trigger=99.0) is False


class TestSpeechShortener:
    def test_without_llm_uses_rules(self):
        short = SpeechShortener()
        out, method = short.shorten("Well, you know, it is fine", "en", 1.0)
        assert method == "rules"
        assert len(out) < len("Well, you know, it is fine")

    def test_returns_original_when_nothing_safe(self):
        short = SpeechShortener()
        text = "The quarterly figures are due tomorrow"
        out, method = short.shorten(text, "en", 5.0)
        assert method == "none"
        assert out == text

    def test_llm_output_that_fails_validation_is_discarded(self):
        class BadLLM:
            def load_model(self, token=None):
                pass

            def generate(self, prompt):
                return "totally different content entirely unrelated here"

        short = SpeechShortener(llm=BadLLM())
        text = "The quarterly figures are due tomorrow morning"
        out, method = short.shorten(text, "en", 1.0)
        # Either the rule path wins, or nothing is applied: never the bad LLM.
        assert out != "totally different content entirely unrelated here"

    def test_llm_exception_falls_back_to_rules(self):
        class ExplodingLLM:
            def load_model(self, token=None):
                raise RuntimeError("no gpu")

        short = SpeechShortener(llm=ExplodingLLM())
        out, method = short.shorten("Well, you know, it is fine", "en", 1.0)
        assert method == "rules"

    def test_valid_llm_output_is_preferred_when_it_fits(self):
        class GoodLLM:
            def load_model(self, token=None):
                pass

            def generate(self, prompt):
                return "The figures are due tomorrow"

        short = SpeechShortener(llm=GoodLLM())
        text = "Well, you know, the quarterly figures are due tomorrow morning"
        out, method = short.shorten(text, "en", 1.0)
        assert method == "llm"
        assert out == "The figures are due tomorrow"


class TestShorteningTriggerConfig:
    def test_trigger_sits_above_the_stretch_band(self):
        """Text is only rewritten for overruns a stretch could not have fixed."""
        assert config.TIMING_SHORTEN_TRIGGER > config.TIMING_LONG_RATIO
