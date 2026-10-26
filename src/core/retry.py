"""
Failure classification and retry policy (Phase 6).

A four-hour render must not die because one of 1,200 TTS segments hit a
temporary file lock, and it must not grind for another four hours retrying a
corrupt source video. So failures are classified first and retried according to
what they are:

  * **TRANSIENT** - a lock, a busy port, a subprocess that can be re-run, a model
    load that failed for want of memory. Worth a few attempts with backoff.
  * **SEGMENT_LOCAL** - one line failed to synthesize or translate. The other
    999 are fine; record this one and carry on.
  * **FATAL** - a corrupt source, an unsupported configuration, a model that
    cannot be obtained. No number of retries helps; stop and say why.

The classification is by message and exception type rather than by call site, so
it can be applied uniformly without every stage having to know about it.
"""

from __future__ import annotations

import errno
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, TypeVar

from src.utils import config

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FailureKind(str, Enum):
    TRANSIENT = "TRANSIENT"
    SEGMENT_LOCAL = "SEGMENT_LOCAL"
    FATAL = "FATAL"


# Substrings that mark a failure as worth retrying. Lowercase; matched against
# the lowercased message.
_TRANSIENT_MARKERS = (
    "temporarily unavailable", "temporary failure", "connection reset",
    "connection aborted", "connection refused", "timed out", "timeout",
    "resource temporarily unavailable", "device or resource busy",
    "text file busy", "too many open files", "try again", "rate limit",
    "429", "502", "503", "504", "cuda out of memory", "out of memory",
    "no space left", "broken pipe", "ssl", "network is unreachable",
    "killed", "worker died", "deadlock",
)

# Substrings that mark a failure as permanent.
_FATAL_MARKERS = (
    "no such file or directory", "not found", "invalid data", "corrupt",
    "unsupported", "unrecognized", "cannot identify image", "moov atom not found",
    "invalid data found when processing input", "no video stream",
    "no audio stream", "permission denied", "not a directory", "is a directory",
    "read-only file system", "model unavailable", "cannot be downloaded",
    "no module named", "import error", "schema version",
)

# Exception types that are never worth retrying: they mean the request itself
# was wrong, not that the environment hiccuped.
_FATAL_TYPES = (FileNotFoundError, IsADirectoryError, NotImplementedError,
                PermissionError, KeyError, TypeError, AttributeError,
                ValueError)

# OSError errno values that are genuinely transient.
_TRANSIENT_ERRNOS = frozenset({
    errno.EAGAIN, errno.EBUSY, errno.ETIMEDOUT, errno.ECONNRESET,
    errno.ECONNREFUSED, errno.EPIPE, errno.ENFILE, errno.EMFILE,
})


def classify(exc: BaseException) -> FailureKind:
    """
    Decide how a failure should be handled.

    Explicit cancellation is neither transient nor fatal: it is a deliberate
    stop, and the caller re-raises it untouched.
    """
    # Imported lazily: CancelledError lives with the pipeline, and this module
    # must stay importable without it.
    if type(exc).__name__ in ("CancelledError", "KeyboardInterrupt", "SystemExit"):
        return FailureKind.FATAL

    message = str(exc).lower()

    if isinstance(exc, OSError) and exc.errno in _TRANSIENT_ERRNOS:
        return FailureKind.TRANSIENT

    # Fatal markers win over transient ones: "no such file or directory" also
    # contains no transient phrase, but "model unavailable: timed out" should
    # be read as fatal because the model is simply not there.
    if any(marker in message for marker in _FATAL_MARKERS):
        return FailureKind.FATAL

    if isinstance(exc, _FATAL_TYPES):
        return FailureKind.FATAL

    if any(marker in message for marker in _TRANSIENT_MARKERS):
        return FailureKind.TRANSIENT

    # Anything unrecognised is treated as transient at the segment level: an
    # unknown one-off error in one line is much more likely to be a hiccup than
    # a permanent condition, and the cost of one extra attempt is small.
    return FailureKind.TRANSIENT


@dataclass
class AttemptRecord:
    """One attempt at something, kept for the manifest."""

    attempt: int
    error: str
    kind: str
    at: float = field(default_factory=time.time)


@dataclass
class RetryOutcome:
    """What happened over the attempts."""

    ok: bool
    value: Optional[T] = None
    error: Optional[str] = None
    kind: Optional[FailureKind] = None
    attempts: int = 0
    history: list[AttemptRecord] = field(default_factory=list)


def run_with_retry(fn: Callable[[], T],
                   attempts: Optional[int] = None,
                   backoff: Optional[float] = None,
                   sleep: Callable[[float], None] = time.sleep,
                   label: str = "",
                   on_retry: Optional[Callable[[int, BaseException], None]] = None) -> RetryOutcome:
    """
    Call `fn`, retrying transient failures with exponential backoff.

    Fatal failures and exhausted attempts return an outcome rather than raising,
    because the caller usually wants to record the failure and continue with the
    rest of the job. Deliberate cancellation is re-raised.
    """
    max_attempts = max(1, int(attempts if attempts is not None else
                              config.RETRY_TRANSIENT_ATTEMPTS))
    delay = float(backoff if backoff is not None else config.RETRY_TRANSIENT_BACKOFF)

    outcome = RetryOutcome(ok=False)
    for attempt in range(1, max_attempts + 1):
        outcome.attempts = attempt
        try:
            outcome.value = fn()
            outcome.ok = True
            return outcome
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001 - classification decides
            if type(exc).__name__ == "CancelledError":
                raise
            kind = classify(exc)
            outcome.error = str(exc)
            outcome.kind = kind
            outcome.history.append(AttemptRecord(attempt, str(exc), kind.value))

            if kind is FailureKind.FATAL:
                logger.error("%s failed permanently: %s", label or "operation", exc)
                return outcome

            if attempt < max_attempts:
                logger.warning("%s attempt %d/%d failed (%s); retrying in %.1fs",
                               label or "operation", attempt, max_attempts, exc, delay)
                if on_retry:
                    try:
                        on_retry(attempt, exc)
                    except Exception:  # noqa: BLE001
                        pass
                sleep(delay)
                delay *= 2
            else:
                logger.error("%s failed after %d attempt(s): %s",
                             label or "operation", attempt, exc)
    return outcome


def should_retry_segment(exc: BaseException, attempts_so_far: int,
                         max_attempts: Optional[int] = None) -> bool:
    """
    Whether one segment is worth another try.

    A segment-local or transient failure is retried while attempts remain; a
    fatal one never is.
    """
    limit = int(max_attempts if max_attempts is not None else
                config.RETRY_SEGMENT_ATTEMPTS)
    if attempts_so_far >= limit:
        return False
    return classify(exc) is not FailureKind.FATAL


def summarise(history: list[AttemptRecord]) -> str:
    """A compact one-line description of what was tried, for logs and manifests."""
    if not history:
        return "no attempts"
    last = history[-1]
    return f"{len(history)} attempt(s); last {last.kind}: {last.error[:200]}"
