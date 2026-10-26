"""
NLLB-200 translation backend.

`facebook/nllb-200-distilled-600M` is a single ungated M2M-100 checkpoint that
covers 200 FLORES-200 languages, including all 11 priority Indic languages, and
translates between any two of them *directly* — no English pivot. It runs on
plain `transformers` with no fairseq/IndicTransToolkit preprocessing layer, and
it emits each target language in its own native script, which is exactly what
IndicF5 needs.

This is the local backend the translation router prefers for Indic pairs. It is
a sentence-level model: feeding it a whole transcript makes it collapse the
paragraph (see the Multiva reference), so callers translate one dialogue
segment at a time and keep the timeline mapping intact.

Only `transformers` + `torch` are imported at module scope; the model itself is
loaded lazily so importing this module is cheap and CPU-only operation works.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "facebook/nllb-200-distilled-600M"


class NLLBBackend:
    """Thin wrapper around an NLLB-200 checkpoint, keyed by FLORES-200 codes."""

    name = "nllb"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        self._requested_device = device
        self.tokenizer = None
        self.model = None
        self.device = None

    # -- lifecycle ---------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def is_available(self) -> bool:
        """
        True when the weights are already on disk (HF cache or a local path).

        Routing uses this to avoid promising an NLLB route and then downloading
        a 2.4 GB checkpoint mid-render. `NLLB_LOCAL_FILES_ONLY=1` forces the
        check to require a real local snapshot even when offline.
        """
        import os

        from src.utils import config

        local_only = getattr(config, "NLLB_LOCAL_FILES_ONLY", False)
        if os.path.isdir(self.model_id):
            return True
        try:
            from huggingface_hub import try_to_load_from_cache

            hit = try_to_load_from_cache(self.model_id, "config.json")
            return isinstance(hit, str)
        except Exception:
            if local_only:
                return False
            # No hub tooling available; assume it can be fetched.
            return not local_only

    def load(self, token: Optional[str] = None) -> None:
        """Load tokenizer + weights on first use. Idempotent."""
        if self.is_loaded:
            return

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = self._requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading NLLB backend %s on %s...", self.model_id, self.device)
        started = time.time()

        from src.utils import config

        local_only = getattr(config, "NLLB_LOCAL_FILES_ONLY", False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=token, local_files_only=local_only)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, token=token, local_files_only=local_only)
        # CPU stays float32; half precision on CPU is slow and can NaN.
        if self.device == "cuda":
            self.model = self.model.half().to("cuda")
        else:
            self.model = self.model.to("cpu")
        self.model.eval()

        logger.info("NLLB backend ready in %.1fs.", time.time() - started)

    def unload(self) -> None:
        """Release the model. Called before a TTS model is loaded."""
        if not self.is_loaded:
            return
        import torch

        logger.info("Unloading NLLB backend %s...", self.model_id)
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- inference ---------------------------------------------------------
    def supports_language(self, flores_code: str) -> bool:
        """True if the loaded (or default) tokenizer knows this FLORES code."""
        return self._flores_is_known(flores_code)

    def _flores_is_known(self, flores_code: str) -> bool:
        from transformers import AutoTokenizer

        tok = self.tokenizer
        if tok is None:
            # Probe with a cached tokenizer so routing can ask before loading.
            try:
                tok = AutoTokenizer.from_pretrained(self.model_id)
            except Exception:
                return True  # cannot check; assume the pair is supported
        unk = tok.unk_token_id
        try:
            return tok.convert_tokens_to_ids(flores_code) != unk
        except Exception:
            return False

    def translate_batch(
        self,
        texts: Sequence[str],
        source_flores: str,
        target_flores: str,
        token: Optional[str] = None,
        max_new_tokens: int = 512,
        num_beams: int = 5,
    ) -> list[str]:
        """
        Translate a batch of strings from `source_flores` to `target_flores`.

        Raises on failure; the caller owns per-segment isolation. Returns one
        string per input, in the same order.
        """
        if not texts:
            return []
        self.load(token=token)

        import torch

        self.tokenizer.src_lang = source_flores
        bos_id = self.tokenizer.convert_tokens_to_ids(target_flores)
        if bos_id == self.tokenizer.unk_token_id:
            raise ValueError(
                f"NLLB does not know target FLORES code {target_flores!r}"
            )

        encodings = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **encodings,
                forced_bos_token_id=bos_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=1,
            )

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [d.strip() for d in decoded]

    def translate(self, text: str, source_flores: str, target_flores: str, **kwargs) -> str:
        out = self.translate_batch([text], source_flores, target_flores, **kwargs)
        return out[0] if out else ""
