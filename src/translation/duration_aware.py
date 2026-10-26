"""
Phase 5, layer 1: duration-aware translation.

NLLB is not an instruction-following model, so asking it to "be concise" does
nothing. Shortening therefore happens as a separate, explicit stage *after*
normal translation and only when a segment is predicted to badly overrun its
window.

Two mechanisms, in order of preference:

  1. `SpeechShortener` with an LLM backend: rewrite the translated line shorter
     while keeping the meaning.
  2. `rule_based_shorten`: a conservative, model-free fallback that removes
     fillers and discourse markers.

Both are gated by `validate_shortening`, which rejects anything that changed
script, grew, emptied out, dropped numerals, or drifted so far from the original
that it is no longer recognisably the same sentence. A rejected shortening is
discarded and the original translation is kept, so the worst case is "no
improvement", never "wrong dialogue".

The original translation always stays in the segment metadata under
`translation.original_translated_text`.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

from src.utils import config
from src.utils import languages as L

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Duration estimation
# ---------------------------------------------------------------------------
def estimate_speech_duration(text: str, lang: str,
                             source_text: Optional[str] = None,
                             slot: Optional[float] = None,
                             source_lang: Optional[str] = None) -> float:
    """
    Predict how long `text` will take to say in `lang`.

    When the original line and its measured slot are supplied this uses the
    self-calibrating `DurationModel`; otherwise it falls back to the registry's
    characters-per-second prior. Either way this only decides *whether* to
    shorten; the final timing is always measured from real generated audio.
    """
    if not text:
        return 0.0
    if source_text and slot and slot > 0.05:
        from src.processing.duration_model import DurationModel
        return DurationModel(source_lang=source_lang, target_lang=lang).estimate(
            text, source_text=source_text, slot=slot)
    cps = L.chars_per_second(lang) or 14.0
    return L.chars(text) / cps


def predict_ratio(text: str, lang: str, target_duration: float,
                  source_text: Optional[str] = None,
                  source_lang: Optional[str] = None) -> float:
    """Predicted generated/target ratio for `text` in a window of `target_duration`."""
    if target_duration <= 0:
        return 1.0
    return estimate_speech_duration(
        text, lang, source_text=source_text, slot=target_duration,
        source_lang=source_lang) / target_duration


def should_shorten(text: str, lang: str, target_duration: float,
                   trigger: Optional[float] = None,
                   source_text: Optional[str] = None,
                   source_lang: Optional[str] = None) -> bool:
    """
    Decide whether a segment is worth rewriting.

    Only clear overruns trigger. A line that a small time-stretch will fix is
    left alone, because rewriting dialogue is the most invasive correction
    available and the most likely to damage meaning.
    """
    if not text or target_duration <= 0:
        return False
    threshold = config.TIMING_SHORTEN_TRIGGER if trigger is None else trigger
    return predict_ratio(text, lang, target_duration,
                         source_text=source_text,
                         source_lang=source_lang) > threshold


# ---------------------------------------------------------------------------
# Script / content validation
# ---------------------------------------------------------------------------
def _dominant_script(text: str) -> str:
    """
    Name of the script most of `text` is written in.

    Derived from Unicode character names ('TELUGU LETTER A' -> 'TELUGU') rather
    than a table, so it covers every script the registry can target without
    being maintained separately.
    """
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        counts[name.split()[0]] = counts.get(name.split()[0], 0) + 1
    if not counts:
        return "UNKNOWN"
    return max(counts, key=counts.get)


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    """Lowercased word tokens. Unicode-aware, so it works for every script."""
    return _WORD_RE.findall(text.lower())


def validate_shortening(original: str, candidate: str,
                        min_containment: float = 0.7) -> tuple[bool, str]:
    """
    Accept or reject a shortened candidate.

    Returns `(ok, reason)`. Rejection is the safe outcome: the caller keeps the
    original translation.

    Checks, in order of how badly each failure would corrupt the dub:
      - empty output
      - not actually shorter
      - changed script (the classic "LLM translated instead of shortening" bug)
      - dropped numerals (times, prices, counts carry plot meaning)
      - words that were never in the original (hallucination / re-translation)

    The last check measures how much of the CANDIDATE is drawn from the
    original, not the reverse. Shortening deletes words, so a symmetric overlap
    would punish exactly the behaviour we want; what matters is that the words
    which remain were all there to begin with.
    """
    original = (original or "").strip()
    candidate = (candidate or "").strip()

    if not candidate:
        return False, "empty"
    if len(candidate) >= len(original):
        return False, "not_shorter"

    src_script = _dominant_script(original)
    cand_script = _dominant_script(candidate)
    if src_script != "UNKNOWN" and cand_script != "UNKNOWN" and src_script != cand_script:
        return False, f"script_changed({src_script}->{cand_script})"

    src_digits = set(re.findall(r"\d+", original))
    cand_digits = set(re.findall(r"\d+", candidate))
    if src_digits - cand_digits:
        return False, "dropped_numerals"

    cand_tokens = _tokens(candidate)
    if cand_tokens:
        src_tokens = set(_tokens(original))
        containment = sum(1 for t in cand_tokens if t in src_tokens) / len(cand_tokens)
        if containment < min_containment:
            return False, f"new_words({containment:.2f})"

    return True, "ok"


# ---------------------------------------------------------------------------
# Rule-based shortening
# ---------------------------------------------------------------------------
# Discourse markers and fillers that carry no plot meaning. Removing them is the
# least destructive way to save time. Kept small and conservative on purpose:
# every entry here is a word a dub can drop without losing information.
_FILLERS: dict[str, tuple[str, ...]] = {
    "en": ("well,", "you know,", "i mean,", "actually,", "basically,",
           "honestly,", "sort of", "kind of", "just ", "really "),
    "hi": ("अच्छा,", "मतलब,", "वैसे,", "तो ", "बस "),
    "ur": ("اچھا،", "یعنی،", "ویسے،"),
    "te": ("అంటే,", "అయితే,", "సరే,", "కాస్త "),
    "ta": ("அதாவது,", "சரி,", "கொஞ்சம் "),
    "bn": ("মানে,", "আসলে,", "তো "),
    "mr": ("म्हणजे,", "बरं,"),
    "gu": ("મતલબ,", "સારું,"),
    "kn": ("ಅಂದರೆ,", "ಸರಿ,"),
    "ml": ("അതായത്,", "ശരി,"),
    "pa": ("ਮਤਲਬ,", "ਠੀਕ "),
}


def rule_based_shorten(text: str, lang: str) -> str:
    """
    Remove fillers and redundant punctuation. Model-free, so it always works.

    Returns the input unchanged when nothing can be safely removed, which the
    caller treats as "shortening not possible".
    """
    if not text:
        return text

    out = text
    for filler in _FILLERS.get(L.normalize(lang), ()):
        # Case-insensitive for Latin fillers, literal otherwise.
        if filler.isascii():
            out = re.sub(re.escape(filler), "", out, flags=re.IGNORECASE)
        else:
            out = out.replace(filler, "")

    # Collapse the whitespace the removals left behind, and tidy doubled commas.
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    out = out.strip(" ,;")

    if not out:
        return text
    ok, _ = validate_shortening(text, out)
    return out if ok else text


# ---------------------------------------------------------------------------
# LLM-backed shortening
# ---------------------------------------------------------------------------
_SHORTEN_INSTRUCTION = (
    "Rewrite the following {language} line so it says the same thing in fewer "
    "words. Keep every name, number and fact. Do not translate it. Do not add "
    "anything. Reply with the rewritten line only.\n\n{text}"
)


class SpeechShortener:
    """
    Shorten translated dialogue so it fits its window.

    The LLM is optional. Without one (or when it produces something the
    validator rejects), the rule-based path still runs, so duration-aware
    handling degrades to something useful instead of failing.
    """

    def __init__(self, llm=None, llm_model_id: Optional[str] = None):
        self.llm = llm
        self.llm_model_id = llm_model_id

    def _get_llm(self):
        if self.llm is not None:
            return self.llm
        if not self.llm_model_id:
            return None
        try:
            from src.translation.text_translator import LLMTranslator
            self.llm = LLMTranslator(self.llm_model_id)
            return self.llm
        except Exception as exc:  # noqa: BLE001
            logger.warning("Shortener could not load an LLM (%s); using rules only.", exc)
            return None

    def _llm_shorten(self, text: str, lang: str, token=None) -> Optional[str]:
        llm = self._get_llm()
        if llm is None:
            return None

        prompt = _SHORTEN_INSTRUCTION.format(
            language=L.get_language_name(lang) or lang, text=text)
        try:
            llm.load_model(token=token)
            out = llm.generate(prompt) if hasattr(llm, "generate") else None
            return (out or "").strip() or None
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM shortening failed for %r: %s", text[:40], exc)
            return None

    def shorten(self, text: str, lang: str, target_duration: float,
                token=None, source_text: Optional[str] = None,
                source_lang: Optional[str] = None) -> tuple[str, str]:
        """
        Try to shorten `text` to fit `target_duration`.

        Returns `(new_text, method)`. `new_text` is the original when no safe
        shortening was found, so callers can always use the result.
        """
        if not text:
            return text, "none"

        candidates: list[tuple[str, str]] = []

        llm_out = self._llm_shorten(text, lang, token=token)
        if llm_out:
            candidates.append((llm_out, "llm"))

        rule_out = rule_based_shorten(text, lang)
        if rule_out != text:
            candidates.append((rule_out, "rules"))

        best: tuple[str, str] = (text, "none")
        best_len = len(text)
        for cand, method in candidates:
            ok, reason = validate_shortening(text, cand)
            if not ok:
                logger.debug("Rejected %s shortening (%s) for %r", method, reason, text[:40])
                continue
            # Prefer the candidate that actually fits, else the shortest one.
            if len(cand) < best_len:
                best, best_len = (cand, method), len(cand)
            if predict_ratio(cand, lang, target_duration,
                             source_text=source_text,
                             source_lang=source_lang) <= config.TIMING_LONG_RATIO:
                return cand, method

        return best
