"""
Phase 4 tests: the Translator facade.

Covers metadata preservation (timestamps, speaker, scripts), NLLB-backed Indic
translation, per-segment failure isolation, caching, and pivot execution. The
NLLB backend is a fake, so nothing is downloaded.
"""

import pytest

from src.translation.text_translator import Translator, SegmentResult
from src.translation.router import TranslationRouter, BackendCapability
from src.utils import languages as L


class FakeNLLB:
    """Fake NLLB backend: echoes a script-tagged translation per language."""

    def __init__(self, available=True, fail_on=None, raise_chunk_for=None):
        self._available = available
        self.fail_on = fail_on or set()
        self.raise_chunk_for = raise_chunk_for or set()
        self.calls = []
        self.unloaded = False

    def is_available(self):
        return self._available

    def is_loaded(self):
        return False

    def unload(self):
        self.unloaded = True

    def translate_batch(self, texts, source_flores, target_flores, token=None, **kw):
        self.calls.append((list(texts), source_flores, target_flores))
        # Simulate a whole-chunk failure when any text is flagged for it.
        if any(t in self.raise_chunk_for for t in texts):
            raise RuntimeError("chunk-level failure")
        out = []
        for t in texts:
            if t in self.fail_on:
                raise RuntimeError(f"cannot translate {t!r}")
            out.append(f"<{target_flores}> {t}")
        return out


@pytest.fixture
def clean_cache(tmp_path, monkeypatch):
    from src.utils import config
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path)
    return tmp_path


def _translator(backend=None, **kw):
    return Translator(nllb_backend=backend or FakeNLLB(), **kw)


class TestMetadataPreservation:
    def test_timestamps_and_speaker_survive_translation(self, clean_cache):
        tr = _translator()
        segments = [
            {"text": "Hello", "start": 0.0, "end": 1.5, "speaker_id": "SPEAKER_00"},
            {"text": "World", "start": 1.5, "end": 3.0, "speaker_id": "SPEAKER_01"},
        ]
        out = tr.translate_segments(segments, "te", model="nllb", source_lang="en")
        assert [s["start"] for s in out] == [0.0, 1.5]
        assert [s["end"] for s in out] == [1.5, 3.0]
        assert [s["speaker_id"] for s in out] == ["SPEAKER_00", "SPEAKER_01"]

    def test_original_source_text_is_preserved(self, clean_cache):
        tr = _translator()
        out = tr.translate_segments([{"text": "Hello", "start": 0, "end": 1}],
                                    "te", model="nllb", source_lang="en")
        assert out[0]["source_text"] == "Hello"
        assert out[0]["translated_text"].startswith("<tel_Telu>")

    def test_input_segments_not_mutated(self, clean_cache):
        tr = _translator()
        segments = [{"text": "Hello", "start": 0, "end": 1}]
        tr.translate_segments(segments, "te", model="nllb", source_lang="en")
        assert "translated_text" not in segments[0]

    def test_metadata_records_route(self, clean_cache):
        tr = _translator()
        out = tr.translate_segments([{"text": "Hello", "start": 0, "end": 1}],
                                    "te", model="nllb", source_lang="en")
        meta = out[0]["translation"]
        assert meta["path"] == "en -> te"
        assert meta["backend"] == "nllb"
        assert meta["status"] == "ok"
        assert meta["pivot"] is False


class TestIndicRouting:
    def test_auto_uses_nllb_for_indic_target(self, clean_cache):
        backend = FakeNLLB()
        tr = _translator(backend)
        out = tr.translate_segments([{"text": "Hi", "start": 0, "end": 1}],
                                    "te", model=None, source_lang="en")
        # NLLB flores codes must reach the backend, not ISO codes.
        assert backend.calls[-1][2] == "tel_Telu"
        assert out[0]["translation"]["backend"] == "nllb"

    def test_auto_prefers_nllb_when_source_is_indic(self, clean_cache):
        backend = FakeNLLB()
        tr = _translator(backend)
        out = tr.translate_segments([{"text": "नमस्ते", "start": 0, "end": 1}],
                                    "fr", model=None, source_lang="hi")
        assert out[0]["translation"]["backend"] == "nllb"

    def test_auto_uses_google_for_non_indic_pairs(self, clean_cache, monkeypatch):
        # A non-Indic pair must keep the fast online backend by default.
        from src.translation import text_translator as tt
        fake_google = type("G", (), {"translate": lambda self, t: f"G:{t}"})()
        tr = _translator()
        monkeypatch.setattr(tr, "get_google_translator", lambda lang: (fake_google, "fr"))
        out = tr.translate_segments([{"text": "Hello", "start": 0, "end": 1}],
                                    "fr", model=None, source_lang="en")
        assert out[0]["translation"]["backend"] == "google"
        assert out[0]["translated_text"] == "G:Hello"

    def test_indic_to_indic_direct_single_pass(self, clean_cache):
        backend = FakeNLLB()
        tr = _translator(backend)
        out = tr.translate_segments([{"text": "नमस्ते", "start": 0, "end": 1}],
                                    "te", model="nllb", source_lang="hi")
        assert len(backend.calls) == 1
        assert backend.calls[0][1] == "hin_Deva"
        assert out[0]["translation"]["pivot"] is False

    def test_non_indic_to_indic_uses_native_script_flores(self, clean_cache):
        backend = FakeNLLB()
        tr = _translator(backend)
        tr.translate_segments([{"text": "Bonjour", "start": 0, "end": 1}],
                              "ta", model="nllb", source_lang="fr")
        assert backend.calls[-1] == (["Bonjour"], "fra_Latn", "tam_Taml")


class TestPassthrough:
    def test_same_language_copies_text(self, clean_cache):
        backend = FakeNLLB()
        tr = _translator(backend)
        out = tr.translate_segments([{"text": "Hello", "start": 0, "end": 1}],
                                    "en", model="nllb", source_lang="en")
        assert out[0]["translated_text"] == "Hello"
        assert out[0]["translation"]["status"] == "passthrough"
        assert backend.calls == []


class TestFailureIsolation:
    def test_single_bad_segment_does_not_sink_the_batch(self, clean_cache):
        # A segment that raises is retried alone, then recorded as failed.
        backend = FakeNLLB(fail_on={"BAD"})
        tr = _translator(backend)
        segments = [
            {"text": "Good", "start": 0, "end": 1},
            {"text": "BAD", "start": 1, "end": 2},
            {"text": "Fine", "start": 2, "end": 3},
        ]
        out = tr.translate_segments(segments, "te", model="nllb", source_lang="en")
        assert out[0]["translation"]["status"] == "ok"
        assert out[1]["translation"]["status"] == "failed"
        assert out[1]["translated_text"] == ""
        assert out[2]["translation"]["status"] == "ok"

    def test_failures_are_recorded_for_retry(self, clean_cache):
        backend = FakeNLLB(fail_on={"BAD"})
        tr = _translator(backend)
        tr.translate_segments(
            [{"text": "BAD", "start": 5.0, "end": 6.0, "speaker_id": "S1"}],
            "te", model="nllb", source_lang="en")
        assert len(tr.last_failures) == 1
        f = tr.last_failures[0]
        assert f["source_text"] == "BAD"
        assert f["speaker_id"] == "S1"
        assert f["start"] == 5.0

    def test_failed_segment_is_not_cached(self, clean_cache):
        backend = FakeNLLB(fail_on={"BAD"})
        tr = _translator(backend)
        tr.translate_segments([{"text": "BAD", "start": 0, "end": 1}],
                              "te", model="nllb", source_lang="en")
        assert "en|te|BAD|nllb" not in tr.cache

    def test_chunk_failure_falls_back_to_per_segment(self, clean_cache):
        # The chunk raises, so the facade retries each text individually and the
        # healthy ones still come back translated.
        backend = FakeNLLB(raise_chunk_for={"BAD"})
        tr = _translator(backend)
        segments = [
            {"text": "Good", "start": 0, "end": 1},
            {"text": "BAD", "start": 1, "end": 2},
        ]
        out = tr.translate_segments(segments, "te", model="nllb", source_lang="en")
        assert out[0]["translation"]["status"] == "ok"
        assert out[1]["translation"]["status"] == "failed"

    def test_nllb_unavailable_falls_back_to_other_backend(self, clean_cache, monkeypatch):
        # When NLLB weights are absent, an Indic request must not silently fail;
        # it routes to a backend the router considers available.
        backend = FakeNLLB(available=False)
        tr = _translator(backend)
        plan = tr.router.route("en", "te", preferred_backend="nllb",
                               available=tr._available_backends())
        assert plan.backend != "nllb"


class TestCaching:
    def test_second_call_hits_cache(self, clean_cache):
        backend = FakeNLLB()
        tr = _translator(backend)
        segs = [{"text": "Hello", "start": 0, "end": 1}]
        tr.translate_segments(segs, "te", model="nllb", source_lang="en")
        n = len(backend.calls)
        out = tr.translate_segments(segs, "te", model="nllb", source_lang="en")
        assert len(backend.calls) == n  # no new backend call
        assert out[0]["translation"]["status"] == "cached"

    def test_cache_key_includes_language_pair(self, clean_cache):
        backend = FakeNLLB()
        tr = _translator(backend)
        tr.translate_segments([{"text": "Hello", "start": 0, "end": 1}],
                              "te", model="nllb", source_lang="en")
        from src.core.cache import translation_key
        key = translation_key("Hello", "en", "te", "nllb",
                              {"optimize": False, "concise": "normal"})
        assert key in tr.cache


class TestPivotExecution:
    def test_pivot_runs_both_hops(self, clean_cache):
        # A backend that only knows en<->fr and en<->de: fr->de must pivot.
        class HubNLLB(FakeNLLB):
            pass

        caps = [BackendCapability("nllb", directions={("fr", "en"), ("en", "de")})]
        tr = Translator(router=TranslationRouter(capabilities=caps), nllb_backend=HubNLLB())
        out = tr.translate_segments([{"text": "Bonjour", "start": 0, "end": 1}],
                                    "de", model="nllb", source_lang="fr")
        assert out[0]["translation"]["pivot"] is True
        assert out[0]["translation"]["path"] == "fr -> en -> de"
        assert out[0]["translated_text"] == "<deu_Latn> <eng_Latn> Bonjour"

    def test_pivot_hop1_failure_marks_segment_failed(self, clean_cache):
        class FailingHop1(FakeNLLB):
            def translate_batch(self, texts, source_flores, target_flores, token=None, **kw):
                if target_flores == "eng_Latn":
                    raise RuntimeError("hop1 down")
                return super().translate_batch(texts, source_flores, target_flores, token, **kw)

        caps = [BackendCapability("nllb", directions={("fr", "en"), ("en", "de")})]
        tr = Translator(router=TranslationRouter(capabilities=caps), nllb_backend=FailingHop1())
        out = tr.translate_segments([{"text": "Bonjour", "start": 0, "end": 1}],
                                    "de", model="nllb", source_lang="fr")
        assert out[0]["translation"]["status"] == "failed"
        assert "pivot hop1" in out[0]["translation"]["error"]


class TestContextAwareMetadata:
    def test_concise_level_is_recorded(self, clean_cache):
        tr = _translator()
        out = tr.translate_segments([{"text": "Hello", "start": 0, "end": 1}],
                                    "te", model="nllb", source_lang="en",
                                    concise="concise")
        assert out[0]["translation"]["concise"] == "concise"

    def test_context_not_applied_by_sentence_mt(self, clean_cache):
        # NLLB has no dialogue-context input, so optimize=True must be recorded
        # as not applied rather than silently claimed.
        tr = _translator()
        out = tr.translate_segments([{"text": "Hello", "start": 0, "end": 1}],
                                    "te", model="nllb", source_lang="en",
                                    optimize=True)
        assert out[0]["translation"]["context_applied"] is False
