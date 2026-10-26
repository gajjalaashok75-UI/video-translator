"""
Sanitising text before it leaves the server.

A job log is written for an operator, not for the public. The real log of a real
render contains lines like:

    stage extraction complete {'path': '/workspace/project/jobs/<id>/audio/audio.wav'}

which is exactly what a debugging operator wants and exactly what an API
response should not hand to a browser. Two different leaks matter:

**Secrets.** A token, an authorization header or a signed model URL must never
appear. `redact_secret` in `src/utils/config.py` already covers the case of
logging a token deliberately ("***" or "None"); this module covers the case of a
token arriving inside a larger string it was never meant to be part of, which is
what an exception message or a third-party log line looks like.

**Internal paths.** Absolute paths disclose the server's directory layout and
the username in it. They are not secret in the way a token is, but they are not
the client's business either, and the phase brief asks for them to be removed
"where practical". They are replaced with a stable placeholder rather than
deleted, so a log line stays readable: `/workspace/project/jobs/<id>/audio/audio.wav`
becomes `…/jobs/<id>/audio/audio.wav`.

The server's own log keeps the original line. Only what the API returns is
filtered, so debugging is not damaged by the filter.
"""

from __future__ import annotations

import os
import re

# The literal placeholder. Chosen over "" so a redacted line still shows that
# something was there.
REDACTED = "***"
_PATH_PLACEHOLDER = "…"

# A secret that appears inside a longer string, usually as `key=value` or
# `key: value`.
# Each entry pairs a pattern with how to rewrite a match. An explicit callable
# per pattern rather than one clever lambda: the patterns capture different
# groups and want different output, and a single rule that tries to serve all of
# them silently drops the key name from the line.
_SECRET_PATTERNS: tuple[tuple[re.Pattern, "re.Callable[[re.Match], str]"], ...] = (
    # Authorization: Bearer <token>  /  Basic <base64>
    (
        re.compile(r"(?i)\b(authorization\s*[:=]\s*)(?:bearer|basic|token)?\s*\S+"),
        lambda m: f"{m.group(1)}{REDACTED}",
    ),
    # Hugging Face tokens: the `hf_` prefix is the whole credential.
    (
        re.compile(r"\bhf_[A-Za-z0-9]{8,}\b"),
        lambda m: REDACTED,
    ),
    # Explicit key/value pairs that name a secret. The key is kept so the line
    # still says what was removed.
    (
        re.compile(
            r"(?i)\b(hf_token|api[_-]?key|access[_-]?token|auth[_-]?token|"
            r"secret|password|passwd|pwd|client[_-]?secret|private[_-]?key)"
            r"(\s*[:=]\s*)[\"']?([^\s\"',;]+)"
        ),
        lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}",
    ),
    # A signed URL query parameter (S3/GCS/HF): the signature is the credential.
    (
        re.compile(
            r"(?i)\b(x-amz-signature|x-goog-signature|signature|sig|token)"
            r"(\s*=\s*)[^\s&\"']+"
        ),
        lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}",
    ),
)

# An absolute POSIX path, or a Windows drive path. The job-relative tail is
# preserved; only the part that discloses the server layout is dropped.
_ABS_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]|/)(?:[^\s'\"<>|:;,)\]]+[\\/])*[^\s'\"<>|:;,)\]]*"
)


def _env_secrets() -> list[str]:
    """
    Values of secret-looking environment variables, for exact-match redaction.

    Read at call time rather than cached: a test sets `HF_TOKEN` after import,
    and a long-lived process may have it rotated. Only non-trivial values are
    returned, so an empty or one-character setting cannot turn into a regex that
    matches everything.
    """
    names = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN",
             "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "API_KEY", "GITHUB_TOKEN")
    values = []
    for name in names:
        value = os.environ.get(name)
        if value and len(value.strip()) >= 8:
            values.append(value.strip())
    return values


def redact_text(text: str, *, keep_paths: bool = False) -> str:
    """
    Remove secrets and (by default) absolute paths from one line of text.

    `keep_paths=True` filters secrets only, which is what a caller wants when a
    path is the point of the message.
    """
    if not text:
        return text
    out = text
    for value in _env_secrets():
        # Literal replacement, not a regex: a token can contain characters that
        # are meaningful in a pattern.
        out = out.replace(value, REDACTED)
    for pattern, rewrite in _SECRET_PATTERNS:
        out = pattern.sub(rewrite, out)
    if not keep_paths:
        out = _ABS_PATH.sub(_shorten_path, out)
    return out


def _shorten_path(match: re.Match) -> str:
    """
    Replace an absolute path with an ellipsis plus its last two components.

    The tail is what identifies the file (`audio/audio.wav`), and it is already
    relative to the job, so it leaks nothing the client cannot see through the
    artifact API.
    """
    raw = match.group(0)
    parts = [p for p in re.split(r"[\\/]+", raw) if p]
    if len(parts) <= 2:
        # A shallow path (`/tmp`) is as likely to be noise as a location.
        return _PATH_PLACEHOLDER
    return f"{_PATH_PLACEHOLDER}/{parts[-2]}/{parts[-1]}"


def redact_lines(lines: list[str], *, keep_paths: bool = False) -> list[str]:
    return [redact_text(line, keep_paths=keep_paths) for line in lines]
