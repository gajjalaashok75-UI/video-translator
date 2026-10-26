"""
Phase 4 tests: translation routing across the language matrix.

Covers direct vs pivot decisions, backend selection/availability, and the
registry contracts the router depends on. No model is downloaded; the NLLB
backend is replaced by a fake.
"""

import pytest

from src.translation.router import (
    BackendCapability,
    TranslationPlan,
    TranslationRouter,
)
from src.utils import languages as L


class FakeBackend:
    """Stand-in for NLLBBackend that records calls instead of loading weights."""

    name = "nllb"

    def __init__(self, available=True, fail_on=None):
        self._available = available
        self.calls = []
        self.fail_on = fail_on or set()
        self.unloaded = False

    def is_available(self):
        return self._available

    def is_loaded(self):
        return False

    def unload(self):
        self.unloaded = True

    def translate_batch(self, texts, source_flores, target_flores, token=None, **kw):
        self.calls.append((list(texts), source_flores, target_flores))
        out = []
        for t in texts:
            if t in self.fail_on:
                raise RuntimeError(f"boom on {t!r}")
            out.append(f"[{target_flores}] {t}")
        return out


class TestRouterDirectRoutes:
    """English<->Indic, Indic<->Indic, non-Indic pairs all route directly."""

    @pytest.mark.parametrize("src,tgt", [
        ("en", "te"), ("te", "en"), ("hi", "te"), ("te", "hi"),
        ("ta", "ml"), ("bn", "gu"), ("kn", "pa"),
        ("fr", "te"), ("te", "fr"), ("en", "fr"), ("de", "ja"),
    ])
    def test_direct_when_backend_can(self, src, tgt):
        router = TranslationRouter()
        plan = router.route(src, tgt, preferred_backend="nllb")
        assert plan.steps == [src, tgt]
        assert plan.is_pivot is False
        assert plan.backend == "nllb"

    def test_priority_indic_all_routable_both_ways(self):
        router = TranslationRouter()
        for code in L.PRIORITY_INDIC:
            for other in ("en", "hi", "te", "fr"):
                if other == code:
                    continue
                plan = router.route(code, other, preferred_backend="nllb")
                assert plan.steps[0] == code and plan.steps[-1] == other

    def test_source_equals_target_is_passthrough(self):
        router = TranslationRouter()
        plan = router.route("te", "te")
        assert plan.is_passthrough is True
        assert plan.backend == "none"
        assert plan.steps == ["te"]


class TestRouterBackendSelection:
    def test_preferred_backend_is_honoured(self):
        router = TranslationRouter()
        plan = router.route("en", "te", preferred_backend="google")
        assert plan.backend == "google"

    def test_falls_back_to_backend_that_can_serve(self):
        caps = [
            BackendCapability("nllb", any_to_any=True),
            BackendCapability("limited", directions={("en", "fr")}),
        ]
        router = TranslationRouter(capabilities=caps)
        # 'limited' cannot do en->te, so the router picks nllb.
        plan = router.route("en", "te", preferred_backend="limited")
        assert plan.backend == "nllb"

    def test_unavailable_backend_is_skipped(self):
        router = TranslationRouter()
        plan = router.route("en", "te", preferred_backend="nllb", available={"google"})
        assert plan.backend == "google"

    def test_no_capable_backend_produces_explicit_pivot(self):
        # A backend that only bridges via English: fr->de must pivot.
        caps = [
            BackendCapability("en-hub", directions={
                ("fr", "en"), ("en", "de"),
            }),
        ]
        router = TranslationRouter(capabilities=caps, pivot="en")
        plan = router.route("fr", "de", preferred_backend="en-hub")
        assert plan.is_pivot is True
        assert plan.steps == ["fr", "en", "de"]
        assert plan.hop_backends == ["en-hub", "en-hub"]

    def test_pivot_is_recorded_not_silent(self):
        caps = [BackendCapability("en-hub", directions={("fr", "en"), ("en", "de")})]
        router = TranslationRouter(capabilities=caps)
        plan = router.route("fr", "de", preferred_backend="en-hub")
        assert "pivot" in plan.reason


class TestRouterLanguageErrors:
    def test_unknown_code_raises(self):
        router = TranslationRouter()
        with pytest.raises(L.UnsupportedLanguage):
            router.route("klingon", "te")

    def test_auto_source_is_rejected(self):
        router = TranslationRouter()
        with pytest.raises(L.UnsupportedLanguage):
            router.route("auto", "te")

    def test_source_only_language_cannot_be_a_target(self):
        # Urdu is a valid *source* but the registry rejects it as a speech target.
        router = TranslationRouter()
        with pytest.raises(L.UnsupportedLanguage):
            router.route("en", "ur")


class TestPlanSerialization:
    def test_to_dict_round_trips_route(self):
        router = TranslationRouter()
        plan = router.route("en", "te", preferred_backend="nllb")
        d = plan.to_dict()
        assert d["path"] == "en -> te"
        assert d["backend"] == "nllb"
        assert d["pivot"] is False
        assert d["passthrough"] is False

    def test_pivot_path_string(self):
        caps = [BackendCapability("h", directions={("fr", "en"), ("en", "de")})]
        plan = TranslationRouter(capabilities=caps).route("fr", "de", preferred_backend="h")
        assert plan.path == "fr -> en -> de"
        assert plan.to_dict()["pivot"] is True


class TestRegistryIndicHelpers:
    def test_is_indic_recognises_priority_languages(self):
        for code in L.PRIORITY_INDIC:
            assert L.is_indic(code), code

    def test_is_indic_false_for_non_indic(self):
        for code in ("en", "fr", "de", "ja", "zh"):
            assert not L.is_indic(code), code

    def test_is_indic_does_not_raise_on_unknown(self):
        assert L.is_indic("not-a-language") is False
