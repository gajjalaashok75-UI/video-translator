"""
Phase 5: what each TTS backend can do about duration.

Declared in one place so the timing pipeline can ask "will this backend fit the
audio itself?" instead of hard-coding backend names in the merge stage. Only
IndicF5 currently honours an exact duration; the rest are corrected by
measurement plus a bounded stretch.
"""

from __future__ import annotations

from typing import NamedTuple, Optional


class TTSCapability(NamedTuple):
    # Backend accepts an explicit total duration and tracks it closely.
    fix_duration: bool = False
    # Backend accepts a speaking-rate multiplier.
    speed: bool = False
    # Voice is a clone of a supplied reference, so it must not be swapped.
    clones: bool = False


# IndicF5's `fix_duration` is the only exact-duration control in the project,
# and it is accurate to roughly 10 ms (Phase 2, verified).
CAPABILITIES: dict[str, TTSCapability] = {
    "indicf5": TTSCapability(fix_duration=True, speed=True, clones=True),
    "f5": TTSCapability(fix_duration=False, speed=True, clones=True),
    "xtts": TTSCapability(fix_duration=False, speed=True, clones=True),
    "vibevoice": TTSCapability(fix_duration=False, speed=False, clones=True),
    "edge": TTSCapability(fix_duration=False, speed=False, clones=False),
    "piper": TTSCapability(fix_duration=False, speed=False, clones=False),
}

# The task key each backend reads its duration request from.
DURATION_KEY = "fix_duration"


def capabilities_for(model_name: Optional[str]) -> TTSCapability:
    """Capability record for a backend, defaulting to the least capable."""
    if not model_name:
        return TTSCapability()
    return CAPABILITIES.get(str(model_name).lower(), TTSCapability())


def honours_duration(model_name: Optional[str]) -> bool:
    """True when the backend fits the audio to a requested duration itself."""
    return capabilities_for(model_name).fix_duration
