"""
Phase 4 tests: the NLLB backend.

Exercises `translate_batch` with a stub tokenizer/model so the real code path
(FLORES handling, batching, decoding, error cases) is tested without downloading
the 2.4 GB checkpoint.
"""

import pytest
import torch

from src.translation.nllb_backend import NLLBBackend


class StubTokenizer:
    """Minimal tokenizer: records src_lang, encodes text, decodes 'translations'."""

    unk_token_id = 3

    def __init__(self, known=("eng_Latn", "tel_Telu", "hin_Deva", "tam_Taml", "mal_Mlym")):
        self.known = set(known)
        self.src_lang = None
        self.seen_src_langs = []

    def convert_tokens_to_ids(self, tok):
        return 42 if tok in self.known else self.unk_token_id

    def __call__(self, texts, **kwargs):
        self.seen_src_langs.append(self.src_lang)
        # Encode as length of text; batch of size len(texts). The real tokenizer
        # returns a BatchEncoding that supports .to(device); mirror that.
        class _Enc(dict):
            def to(self, device):
                return self

        return _Enc(input_ids=torch.zeros((len(texts), 4), dtype=torch.long))

    def batch_decode(self, generated, skip_special_tokens=True):
        return [f"<{t}>" for t in generated]


class StubModel:
    def __init__(self):
        self.forced_bos = []

    def generate(self, **kwargs):
        self.forced_bos.append(kwargs.get("forced_bos_token_id"))
        # One "translated token id" per row of the input batch.
        n = kwargs["input_ids"].shape[0]
        return [[1] for _ in range(n)]

    def eval(self):
        return self


def _backend():
    b = NLLBBackend()
    b.tokenizer = StubTokenizer()
    b.model = StubModel()
    b.device = "cpu"
    return b


class TestTranslateBatch:
    def test_returns_one_result_per_input(self):
        b = _backend()
        out = b.translate_batch(["a", "b", "c"], "eng_Latn", "tel_Telu")
        assert len(out) == 3
        assert all(o.strip() for o in out)

    def test_sets_source_language_on_tokenizer(self):
        b = _backend()
        b.translate_batch(["hello"], "eng_Latn", "tel_Telu")
        assert b.tokenizer.seen_src_langs == ["eng_Latn"]

    def test_forces_target_bos_token(self):
        b = _backend()
        b.translate_batch(["hello"], "eng_Latn", "tel_Telu")
        assert b.model.forced_bos == [42]

    def test_empty_input_returns_empty(self):
        b = _backend()
        assert b.translate_batch([], "eng_Latn", "tel_Telu") == []

    def test_unknown_target_raises(self):
        b = _backend()
        with pytest.raises(ValueError, match="does not know target FLORES code"):
            b.translate_batch(["hello"], "eng_Latn", "xyz_Latn")

    def test_indic_to_indic_direct_single_call(self):
        b = _backend()
        calls_before = len(b.model.forced_bos)
        b.translate_batch(["नमस्ते"], "hin_Deva", "tam_Taml")
        # Exactly one generate() call: no English pivot for Indic-Indic.
        assert len(b.model.forced_bos) == calls_before + 1


class TestFloresSupport:
    def test_supports_known_code(self):
        b = _backend()
        assert b.supports_language("tel_Telu") is True

    def test_rejects_unknown_code(self):
        b = _backend()
        assert b.supports_language("xyz_Latn") is False


class TestIsAvailable:
    def test_local_directory_counts_as_available(self, tmp_path):
        b = NLLBBackend(model_id=str(tmp_path))
        assert b.is_available() is True
