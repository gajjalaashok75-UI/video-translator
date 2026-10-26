"""
Translation routing.

Mirrors `TTSEngine.route_model`: a language-driven decision, kept in one place
so the pipeline never grows scattered per-pair `if` statements. Given a source
and target language it returns a `TranslationPlan` describing *how* the text
will be translated, and that plan is attached to the translation metadata so the
path is inspectable after the fact.

Design rules:

* The registry (`src/utils/languages.py`) is the only source of language
  knowledge. Routing asks it `is_indic()` and `source_flores()` /
  `target_flores()`; it never hard-codes a pair.
* A backend declares the directions it can do (`supports()`). The router picks
  the first backend whose capability matches, so adding a backend is adding a
  capability, not editing a dispatch table.
* A direct route is preferred. When no backend can go direct, the router builds
  an explicit pivot (`fr -> en -> te`) and records it. Pivoting is never silent.
* `source == target` is a passthrough, not a translation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

from src.utils import languages

logger = logging.getLogger(__name__)

# The pivot used when no backend can translate a pair directly. English is the
# best-connected hub for the models we ship (NLLB, the LLM translators).
DEFAULT_PIVOT = "en"


@dataclass
class TranslationPlan:
    """
    How a (source, target) pair will be translated.

    `steps` is the ordered list of language codes the text passes through, so a
    direct route is `[src, tgt]` and a pivot is `[src, pivot, tgt]`. `backend`
    names the engine that performs each hop (one entry per hop).
    """

    source: str
    target: str
    backend: str
    steps: list[str]
    hop_backends: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def is_passthrough(self) -> bool:
        return self.source == self.target

    @property
    def is_pivot(self) -> bool:
        return len(self.steps) > 2

    @property
    def path(self) -> str:
        """Compact, log-friendly description, e.g. 'fr -> en -> te'."""
        return " -> ".join(self.steps)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "backend": self.backend,
            "path": self.path,
            "steps": list(self.steps),
            "hop_backends": list(self.hop_backends),
            "pivot": self.is_pivot,
            "passthrough": self.is_passthrough,
            "reason": self.reason,
        }


class BackendCapability:
    """
    Declares what a backend can translate.

    `directions` is a set of `(source, target)` language-code pairs the backend
    handles directly. `any_to_any` backends (NLLB) accept every registered pair.
    """

    def __init__(self, name: str, directions: Optional[set[tuple[str, str]]] = None,
                 any_to_any: bool = False):
        self.name = name
        self.directions = directions or set()
        self.any_to_any = any_to_any

    def supports(self, source: str, target: str) -> bool:
        if source == target:
            return True
        if self.any_to_any:
            return True
        return (source, target) in self.directions


def default_capabilities() -> list[BackendCapability]:
    """
    Capabilities of the backends we ship.

    NLLB is `any_to_any` over the registered languages — it is a 200-language
    M2M model. The LLM translators are prompt-driven and can attempt any pair,
    but they are the *fallback*, so they are registered last and also treated as
    any-to-any; the router only reaches them when NLLB is unavailable.
    """
    return [
        BackendCapability("nllb", any_to_any=True),
        BackendCapability("hymt", any_to_any=True),
        BackendCapability("llama", any_to_any=True),
        BackendCapability("alma", any_to_any=True),
        BackendCapability("google", any_to_any=True),
    ]


class TranslationRouter:
    """
    Chooses a translation plan for a language pair.

    `preferred_backend` is the backend the user picked (or the default). When it
    can serve the pair directly, it is used. When it cannot, the router looks for
    another backend that can, and only then falls back to an explicit pivot.
    """

    def __init__(self, capabilities: Optional[Sequence[BackendCapability]] = None,
                 pivot: str = DEFAULT_PIVOT):
        self.capabilities = list(capabilities or default_capabilities())
        self.pivot = pivot

    def _capability(self, name: str) -> Optional[BackendCapability]:
        for cap in self.capabilities:
            if cap.name == name:
                return cap
        return None

    def route(self, source: str, target: str, preferred_backend: Optional[str] = None,
              available: Optional[set[str]] = None) -> TranslationPlan:
        """
        Decide how to translate `source` -> `target`.

        `available` restricts which backends may be chosen (e.g. NLLB before its
        weights are on disk). When None, every registered capability is eligible.

        Raises `languages.UnsupportedLanguage` for codes the registry does not
        know, so a typo fails loudly instead of silently translating from/to
        English.
        """
        src = languages.normalize(source)
        tgt = languages.normalize(target)

        if src == "auto":
            raise languages.UnsupportedLanguage(
                "source language is 'auto'; resolve it (e.g. from ASR) before routing"
            )

        # Registry rejects a code that cannot be a translation source/target.
        languages.source_flores(src)
        languages.target_flores(tgt, for_speech=False)

        if src == tgt:
            return TranslationPlan(
                source=src, target=tgt, backend="none", steps=[src],
                reason="source and target are identical; text passes through",
            )

        def usable(cap: "BackendCapability") -> bool:
            return available is None or cap.name in available

        # 1. Direct route on the preferred backend.
        preferred = preferred_backend or self.capabilities[0].name
        cap = self._capability(preferred)
        if cap and usable(cap) and cap.supports(src, tgt):
            return TranslationPlan(
                source=src, target=tgt, backend=preferred,
                steps=[src, tgt], hop_backends=[preferred],
                reason="direct route on the requested backend",
            )

        # 2. Direct route on any other available backend.
        for cap in self.capabilities:
            if usable(cap) and cap.supports(src, tgt):
                logger.info(
                    "Router: %s cannot do %s->%s directly; using %s.",
                    preferred, src, tgt, cap.name,
                )
                return TranslationPlan(
                    source=src, target=tgt, backend=cap.name,
                    steps=[src, tgt], hop_backends=[cap.name],
                    reason=f"direct route on '{cap.name}' (requested '{preferred}' cannot)",
                )

        # 3. Explicit pivot through the hub language.
        return self._pivot_plan(src, tgt, preferred, available)

    def _pivot_plan(self, src: str, tgt: str, preferred: str,
                    available: Optional[set[str]] = None) -> TranslationPlan:
        pivot = languages.normalize(self.pivot)
        if pivot in (src, tgt):
            raise languages.UnsupportedLanguage(
                f"no backend can translate {src}->{tgt} directly and the pivot "
                f"'{pivot}' is one of the endpoints"
            )

        hop1 = self.route(src, pivot, preferred_backend=preferred, available=available)
        hop2 = self.route(pivot, tgt, preferred_backend=preferred, available=available)
        backends = [hop1.backend, hop2.backend]
        logger.info("Router: pivoting %s -> %s -> %s (backends %s).", src, pivot, tgt, backends)
        return TranslationPlan(
            source=src, target=tgt, backend=backends[0],
            steps=[src, pivot, tgt], hop_backends=backends,
            reason=f"no direct backend; explicit pivot through '{pivot}'",
        )
