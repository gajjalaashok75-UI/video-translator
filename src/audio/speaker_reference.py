"""
Automatic per-speaker reference selection for voice cloning.

Why this module exists
----------------------
Diarization tells us WHO spoke WHEN, and the transcriber tells us what was said.
Neither, on its own, tells us which stretch of audio is a *good voice-cloning
reference* for a given speaker. The pipeline used to pick the longest segment,
which in real footage is often the one with music underneath, a second speaker
bleeding in, or a phone-line-quality aside.

The cost of getting this wrong is not cosmetic. A cloning model conditions on
the reference for the entire movie, so one bad reference mis-voices a character
in every single scene.

What this module does
---------------------
For each diarized speaker it scores every candidate window of that speaker's
speech and returns the best one, together with the transcript of exactly that
window. The transcript is what IndicF5 calls `ref_text`, and it must correspond
to the audio it is paired with — a mismatch wrecks the model's duration model
and produces rushed, robotic output. We therefore derive the text from the same
word objects that define the window, so they cannot drift apart.

Two-stage selection
-------------------
Scoring on transcript signals alone (word confidence, speech density, internal
pauses, other-speaker overlap, position in the file) is cheap. Judging loudness,
clipping and silence requires reading the waveform, which we do only for the
top few candidates per speaker. A two-hour movie with twelve speakers therefore
costs a handful of audio reads, not thousands.

Deliberate limitations
----------------------
Diarization does not separate overlapping speech. When two people talk at once,
the mix contains both voices and no amount of window selection will remove the
other one — that needs actual source separation, which this module does not do.
What it does instead is *refuse* to use overlapped audio: windows covered by
another speaker's turns are rejected outright rather than quietly polluting the
reference.

Adapted from the reference implementations in `references/Multiva.Ai`
(`engine/reference_audio.py`) and `references/video-dubbing-system`
(`backend/workers/tasks.py`), reworked to run on the segment/word structures
this repository already produces.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from src.utils import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
# Cloning models want roughly 6-12s of clean speech. Below ~3s the speaker
# embedding is unstable; above ~15s F5 starts truncating internally anyway.
TARGET_MIN_SECONDS = getattr(config, "REFERENCE_TARGET_MIN", 6.0)
TARGET_MAX_SECONDS = getattr(config, "REFERENCE_TARGET_MAX", 12.0)

# Accepted only when nothing better exists. Logged loudly when used.
HARD_MIN_SECONDS = getattr(config, "REFERENCE_HARD_MIN", 3.0)

# A pause longer than this is treated as a break between windows, so we never
# splice across a scene cut or a long silence.
MAX_INTERNAL_GAP = getattr(config, "REFERENCE_MAX_INTERNAL_GAP", 0.6)

# Fraction of a candidate window that may be covered by ANOTHER speaker's turns
# before the window is rejected. Overlap means the mix contains two voices.
MAX_OVERLAP_RATIO = getattr(config, "REFERENCE_OVERLAP_REJECT", 0.30)

# Footage before this timestamp is disproportionately music, titles and intros.
INTRO_PENALTY_BEFORE = getattr(config, "REFERENCE_INTRO_PENALTY_BEFORE", 2.0)

# Points for filling the target band. The confidence term spans ~0.3 over the
# range of windows we see, so a weight below that lets a short, very confident
# clip outrank a clean 6-12s one and the reference lands outside the band.
DURATION_WEIGHT = getattr(config, "REFERENCE_DURATION_WEIGHT", 1.0)

# Audio-quality gates.
MAX_CLIPPING_RATIO = getattr(config, "REFERENCE_CLIPPING_MAX", 0.01)
MAX_SILENCE_RATIO = getattr(config, "REFERENCE_SILENCE_MAX", 0.5)
MIN_AUDIO_RMS = getattr(config, "REFERENCE_RMS_THRESHOLD", 0.02)

# How many transcript-scored candidates get a (comparatively expensive)
# waveform probe per speaker.
AUDIO_PROBE_TOP_K = getattr(config, "REFERENCE_AUDIO_PROBE_TOP_K", 5)

# Fraction of the candidate's own timeline that must be voiced speech for it to
# be worth considering at all.
MIN_SPEECH_DENSITY = 0.35


@dataclass
class AudioQuality:
    """Measured properties of a candidate's waveform."""

    rms: float = 0.0
    peak: float = 0.0
    clipping_ratio: float = 0.0
    silence_ratio: float = 1.0
    voiced_duration: float = 0.0
    acceptable: bool = False
    reason: str = "not probed"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReferenceCandidate:
    """One possible reference clip for one speaker."""

    start: float
    end: float
    text: str
    score: float
    speaker: str
    # Signals that fed the score, kept for logging and for tests.
    word_confidence: float = 0.0
    speech_density: float = 0.0
    max_gap: float = 0.0
    overlap_ratio: float = 0.0
    word_count: int = 0
    source: str = "words"  # "words" | "segments"
    quality: Optional[AudioQuality] = None
    rejected_reason: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration"] = self.duration
        data["quality"] = self.quality.to_dict() if self.quality else None
        return data


@dataclass
class SpeakerReference:
    """
    The selected reference for one speaker.

    `reference_audio` may be None, which means "this speaker has no usable
    reference". Callers must treat that as a first-class outcome: the speaker
    falls back to a generic voice, and we never substitute another speaker's
    reference.
    """

    speaker: str
    reference_audio: Optional[str] = None
    reference_text: str = ""
    duration: float = 0.0
    quality_score: float = 0.0
    tier: str = "none"  # "target" | "short" | "degraded" | "none"
    reason: str = ""
    candidate: Optional[ReferenceCandidate] = None

    @property
    def has_audio(self) -> bool:
        return bool(self.reference_audio)

    @property
    def has_text(self) -> bool:
        return bool(self.reference_text.strip())

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "reference_audio": self.reference_audio,
            "reference_text": self.reference_text,
            "duration": self.duration,
            "quality_score": self.quality_score,
            "tier": self.tier,
            "reason": self.reason,
            "candidate": self.candidate.to_dict() if self.candidate else None,
        }


# ---------------------------------------------------------------------------
# Overlap analysis
# ---------------------------------------------------------------------------
def speaker_intervals(diar_segments: Sequence[dict], speaker: str) -> list[tuple[float, float]]:
    """Merged (start, end) turns for one speaker."""
    raw = sorted(
        (float(s["start"]), float(s["end"]))
        for s in diar_segments
        if s.get("speaker") == speaker and float(s["end"]) > float(s["start"])
    )
    merged: list[tuple[float, float]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def other_speaker_overlap(
    start: float, end: float, diar_segments: Sequence[dict], speaker: str
) -> float:
    """
    Fraction of [start, end) covered by any *other* speaker's turns.

    Returns 0.0-1.0. This is the signal that keeps us from conditioning a clone
    on a moment where somebody else was talking over our speaker.
    """
    if end <= start:
        return 0.0

    covered = 0.0
    spans: list[tuple[float, float]] = []
    for seg in diar_segments:
        if seg.get("speaker") == speaker:
            continue
        s, e = float(seg["start"]), float(seg["end"])
        if s < end and e > start:
            spans.append((max(s, start), min(e, end)))

    if not spans:
        return 0.0

    # Merge before summing so overlapping turns are not double-counted.
    spans.sort()
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    covered = sum(e - s for s, e in merged)
    return min(1.0, covered / (end - start))


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------
def _words_for_speaker(segments: Sequence[dict], speaker: str) -> list[dict]:
    """
    Every word this speaker uttered, in order.

    Only words whose segment is attributed to `speaker` are collected, which is
    what guarantees a reference is never built from two voices. Words without
    timings are skipped rather than guessed at.
    """
    words: list[dict] = []
    for seg in segments:
        if seg.get("speaker_id") != speaker:
            continue
        for w in seg.get("words") or []:
            start, end = w.get("start"), w.get("end")
            if start is None or end is None or float(end) <= float(start):
                continue
            text = (w.get("word") or "").strip()
            if not text:
                continue
            words.append({
                "word": text,
                "start": float(start),
                "end": float(end),
                "probability": float(w.get("probability", 1.0) or 1.0),
            })
    words.sort(key=lambda w: (w["start"], w["end"]))
    return words


def _segment_spans_for_speaker(segments: Sequence[dict], speaker: str) -> list[dict]:
    """Fallback candidates when the transcriber gave us no word timings."""
    spans = []
    for seg in segments:
        if seg.get("speaker_id") != speaker:
            continue
        start, end = float(seg["start"]), float(seg["end"])
        if end <= start:
            continue
        spans.append({
            "start": start,
            "end": end,
            "text": (seg.get("text") or "").strip(),
        })
    return spans


def _windows_over_words(
    words: Sequence[dict], max_gap: float
) -> Iterable[list[dict]]:
    """
    Every window of contiguous speech whose internal pauses stay under max_gap.

    Yields progressively longer windows from each starting word, stopping at the
    first pause that breaks the run. Mirrors Multiva's `_best_window_for_gap`,
    which is O(n^2) but over a single speaker's words, so n stays small.
    """
    for i in range(len(words)):
        window: list[dict] = []
        for j in range(i, len(words)):
            if window and words[j]["start"] - window[-1]["end"] > max_gap:
                break
            window.append(words[j])
            yield list(window)


def _features(
    start: float,
    end: float,
    words: Sequence[dict],
    diar_segments: Sequence[dict],
    speaker: str,
) -> dict:
    """Transcript-side signals for one candidate window."""
    duration = end - start
    if duration <= 0:
        return {}

    if words:
        voiced = sum(w["end"] - w["start"] for w in words)
        confidence = sum(w["probability"] for w in words) / len(words)
        max_gap = max(
            (b["start"] - a["end"] for a, b in zip(words, words[1:])),
            default=0.0,
        )
    else:
        # Segment-level fallback: we have no word detail, so assume the span is
        # filled and fully confident rather than penalising it for missing data.
        voiced = duration
        confidence = 1.0
        max_gap = 0.0

    return {
        "duration": duration,
        "density": min(1.0, voiced / duration),
        "confidence": confidence,
        "max_gap": max(0.0, max_gap),
        "overlap": other_speaker_overlap(start, end, diar_segments, speaker),
        "word_count": len(words),
    }


def score_features(features: dict, start: float) -> float:
    """
    Higher is better. Rewards confident, dense, uninterrupted speech.

    Weighting follows Multiva's `_score_window`, extended with an explicit
    overlap penalty because our diarization gives us the other speakers' turns
    and theirs did not.
    """
    if not features or features.get("duration", 0) <= 0:
        return float("-inf")

    score = (
        features["confidence"] * 2.0
        + features["density"] * 1.5
        - features["max_gap"] * 1.0
    )

    # Overlap is disqualifying, not merely discouraging: a window half-covered
    # by another speaker is a duet, not a reference.
    score -= features["overlap"] * 4.0

    if start < INTRO_PENALTY_BEFORE:
        score -= 0.5

    # Preference for longer prompts, saturating at the target maximum. This is
    # deliberately comparable in magnitude to the confidence term: a window that
    # reaches the target band is worth more than a marginally more confident
    # window that falls short of it.
    score += DURATION_WEIGHT * min(features["duration"], TARGET_MAX_SECONDS) / TARGET_MAX_SECONDS

    return score


def candidate_windows(
    segments: Sequence[dict],
    diar_segments: Sequence[dict],
    speaker: str,
    limit: int = 8,
) -> list[ReferenceCandidate]:
    """
    Rank this speaker's possible reference windows, best first.

    Ties break on earliest start so selection is deterministic: the same input
    always yields the same reference, which matters because the whole movie is
    voiced from it and a re-run must not silently change the voice.
    """
    words = _words_for_speaker(segments, speaker)
    candidates: list[ReferenceCandidate] = []

    if words:
        for max_gap in (MAX_INTERNAL_GAP, MAX_INTERNAL_GAP * 2, MAX_INTERNAL_GAP * 4):
            for window in _windows_over_words(words, max_gap):
                start, end = window[0]["start"], window[-1]["end"]
                duration = end - start
                # Too short to clone from, too long for the model to use whole.
                if duration < HARD_MIN_SECONDS or duration > TARGET_MAX_SECONDS:
                    continue
                feats = _features(start, end, window, diar_segments, speaker)
                if not feats or feats["density"] < MIN_SPEECH_DENSITY:
                    continue
                candidates.append(ReferenceCandidate(
                    start=start,
                    end=end,
                    text=" ".join(w["word"] for w in window).strip(),
                    score=score_features(feats, start),
                    speaker=speaker,
                    word_confidence=feats["confidence"],
                    speech_density=feats["density"],
                    max_gap=feats["max_gap"],
                    overlap_ratio=feats["overlap"],
                    word_count=feats["word_count"],
                    source="words",
                ))
    else:
        for span in _segment_spans_for_speaker(segments, speaker):
            start, end = span["start"], span["end"]
            duration = end - start
            if duration <= 0:
                continue
            feats = _features(start, end, [], diar_segments, speaker)
            candidates.append(ReferenceCandidate(
                start=start,
                end=end,
                text=span["text"],
                score=score_features(feats, start),
                speaker=speaker,
                word_confidence=feats["confidence"],
                speech_density=feats["density"],
                overlap_ratio=feats["overlap"],
                source="segments",
            ))

    candidates.sort(key=lambda c: (-c.score, c.start))

    # Drop near-duplicates: shifting a window by one word is not a new option.
    chosen: list[ReferenceCandidate] = []
    for cand in candidates:
        if any(_heavy_overlap(cand, kept) for kept in chosen):
            continue
        chosen.append(cand)
        if len(chosen) >= limit:
            break
    return chosen


def _heavy_overlap(a: ReferenceCandidate, b: ReferenceCandidate) -> bool:
    span = min(a.end, b.end) - max(a.start, b.start)
    return span > 0.5 * min(a.duration, b.duration)


# ---------------------------------------------------------------------------
# Audio quality
# ---------------------------------------------------------------------------
def measure_audio_quality(wave: np.ndarray, sr: int) -> AudioQuality:
    """
    Judge a candidate waveform on loudness, clipping and silence.

    Silence is measured with a short-frame energy floor relative to the clip's
    own peak, so it adapts to quiet recordings instead of assuming a fixed dBFS
    level that would reject an entire softly-mastered film.
    """
    quality = AudioQuality()
    if wave is None or len(wave) == 0 or sr <= 0:
        quality.reason = "empty"
        return quality

    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    duration = len(wave) / float(sr)

    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(wave)))) if wave.size else 0.0

    quality.peak = peak
    quality.rms = rms
    # Samples pinned at full scale are clipped; a few are fine, a stream is not.
    quality.clipping_ratio = float(np.mean(np.abs(wave) >= 0.999)) if wave.size else 0.0

    frame = max(1, int(0.02 * sr))
    frames = wave[: (len(wave) // frame) * frame].reshape(-1, frame)
    if frames.size:
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
        floor = max(peak, 1e-6) * (10 ** (-40 / 20))
        quiet = frame_rms < floor
        quality.silence_ratio = float(np.mean(quiet))
        quality.voiced_duration = float(np.sum(~quiet) * frame / float(sr))
    else:
        quality.silence_ratio = 1.0
        quality.voiced_duration = 0.0

    if rms < MIN_AUDIO_RMS:
        quality.reason = f"too quiet (rms={rms:.4f})"
    elif quality.clipping_ratio > MAX_CLIPPING_RATIO:
        quality.reason = f"clipping ({quality.clipping_ratio:.1%} of samples)"
    elif quality.silence_ratio > MAX_SILENCE_RATIO:
        quality.reason = f"mostly silence ({quality.silence_ratio:.0%})"
    elif duration < HARD_MIN_SECONDS:
        quality.reason = f"too short ({duration:.1f}s)"
    else:
        quality.acceptable = True
        quality.reason = "ok"

    return quality


def _read_span(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    lo = max(0, int(round(start * sr)))
    hi = min(len(audio), int(round(end * sr)))
    if hi <= lo:
        return np.zeros(0, dtype=np.float32)
    return audio[lo:hi]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def select_reference(
    segments: Sequence[dict],
    diar_segments: Sequence[dict],
    speaker: str,
    audio: Optional[np.ndarray] = None,
    sr: int = 0,
    probe_limit: int = AUDIO_PROBE_TOP_K,
) -> tuple[Optional[ReferenceCandidate], str]:
    """
    Choose the best reference window for `speaker`.

    Returns (candidate, tier). `candidate` is None only when this speaker has no
    usable speech at all.

    Tiers, best to worst:
      target   — a full-length, clean, non-overlapped window
      short    — clean but under the target length (voice quality suffers)
      degraded — nothing passed the audio checks; best available, logged loudly
    """
    candidates = candidate_windows(segments, diar_segments, speaker)
    if not candidates:
        return None, "none"

    # Overlapped windows are refused rather than ranked. Diarization does not
    # separate voices, so an overlapped clip contains someone else too.
    clean = [c for c in candidates if c.overlap_ratio <= MAX_OVERLAP_RATIO]
    if not clean:
        best = candidates[0]
        logger.warning(
            "Speaker %s: every candidate window overlaps another speaker "
            "(best overlap %.0f%%). Overlapping speech cannot be separated here; "
            "no clean reference is available.",
            speaker, best.overlap_ratio * 100,
        )
        return best, "degraded"

    if audio is None or sr <= 0:
        # No waveform available: fall back to the transcript-only ranking.
        best = clean[0]
        tier = "target" if best.duration >= TARGET_MIN_SECONDS else "short"
        return best, tier

    # Probe the top few candidates on the waveform. The transcript score got us
    # a shortlist; loudness, clipping and silence decide between them.
    scored: list[tuple[float, ReferenceCandidate]] = []
    for cand in clean[: max(1, probe_limit)]:
        wave = _read_span(audio, sr, cand.start, cand.end)
        quality = measure_audio_quality(wave, sr)
        cand.quality = quality
        if not quality.acceptable:
            cand.rejected_reason = quality.reason
            continue
        # Keep the transcript ordering dominant and let audio quality nudge it.
        audio_bonus = (1.0 - quality.silence_ratio) + min(quality.rms / 0.1, 1.0) * 0.5
        scored.append((cand.score + audio_bonus, cand))

    if not scored:
        best = clean[0]
        reason = "; ".join(
            f"{c.start:.1f}s: {c.rejected_reason}" for c in clean[:probe_limit] if c.rejected_reason
        )
        logger.warning(
            "Speaker %s: no candidate passed the audio quality checks (%s). "
            "Falling back to the best transcript-scored window; voice cloning "
            "may be unreliable.",
            speaker, reason or "no details",
        )
        return best, "degraded"

    scored.sort(key=lambda pair: (-pair[0], pair[1].start))
    best = scored[0][1]
    tier = "target" if best.duration >= TARGET_MIN_SECONDS else "short"
    return best, tier


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_reference_audio(
    audio: np.ndarray,
    sr: int,
    start: float,
    end: float,
    out_path: str | Path,
    fade_ms: float = 10.0,
) -> tuple[str, float]:
    """
    Cut, condition and write a reference clip.

    Conditioning is deliberately timing-preserving: DC removal, edge fades and
    peak normalisation. Anything that changes the timeline (silence stripping in
    particular) would break the correspondence with `ref_text`, which transcribes
    the whole window.

    The clip is written at the source sample rate rather than resampled, so no
    high-frequency detail is lost; IndicF5 resamples internally and caches the
    result.
    """
    import soundfile as sf

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clip = _read_span(audio, sr, start, end).astype(np.float32, copy=True)
    if clip.size == 0:
        raise ValueError(f"Empty reference span {start:.2f}-{end:.2f}s")

    # Remove DC offset, which biases speaker embeddings.
    clip -= float(np.mean(clip))

    peak = float(np.max(np.abs(clip)))
    if peak > 0:
        clip *= (10 ** (-3 / 20)) / peak  # normalise to -3 dBFS

    fade = int(fade_ms / 1000.0 * sr)
    if fade > 0 and len(clip) > 2 * fade:
        clip[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        clip[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    sf.write(str(out_path), clip, sr, subtype="PCM_16")
    return str(out_path), len(clip) / float(sr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_speaker_references(
    segments: Sequence[dict],
    diar_segments: Sequence[dict],
    vocals_path: str | Path,
    output_dir: str | Path,
    min_speakers: int = 1,
) -> dict[str, SpeakerReference]:
    """
    Build one reference per diarized speaker.

    Called once per video, after speaker assignment, and reused for every
    segment. Returns a mapping keyed by speaker id; the values are plain
    dataclasses with `to_dict()`, so the caller can persist them for a resume
    without this module knowing anything about checkpointing.

    A speaker with no usable audio yields a `SpeakerReference` whose
    `reference_audio` is None. That is a deliberate, reported outcome — never a
    silent substitution of somebody else's voice.
    """
    import soundfile as sf

    speakers = sorted({s.get("speaker") for s in diar_segments if s.get("speaker")})
    if not speakers:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio = None
    sr = 0
    try:
        audio, sr = sf.read(str(vocals_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
    except Exception as exc:
        logger.error(
            "Could not read vocals for reference extraction (%s). Speakers will "
            "fall back to generic voices.", exc,
        )

    results: dict[str, SpeakerReference] = {}
    for speaker in speakers:
        candidate, tier = select_reference(segments, diar_segments, speaker, audio, sr)

        if candidate is None:
            logger.warning(
                "Speaker %s has no usable speech for a reference; it will use a "
                "generic voice. (Its dialogue is still translated and dubbed.)",
                speaker,
            )
            results[speaker] = SpeakerReference(
                speaker=speaker, tier="none",
                reason="no candidate speech found for this speaker",
            )
            continue

        if audio is None or sr <= 0:
            results[speaker] = SpeakerReference(
                speaker=speaker, tier="none",
                reason="vocals unavailable; could not extract reference audio",
                candidate=candidate,
            )
            continue

        out_path = output_dir / f"{speaker}_reference.wav"
        try:
            path, actual = extract_reference_audio(
                audio, sr, candidate.start, candidate.end, out_path
            )
        except Exception as exc:
            logger.error("Reference extraction failed for %s: %s", speaker, exc)
            results[speaker] = SpeakerReference(
                speaker=speaker, tier="none",
                reason=f"extraction failed: {exc}", candidate=candidate,
            )
            continue

        quality_note = ""
        if candidate.quality and candidate.quality.acceptable:
            quality_note = f", silence {candidate.quality.silence_ratio:.0%}"
        logger.info(
            "Reference for %s: %.2fs @%.0fHz (tier=%s, score=%.2f, "
            "overlap %.0f%%%s) -> %s",
            speaker, actual, sr, tier, candidate.score,
            candidate.overlap_ratio * 100, quality_note, out_path.name,
        )
        if tier != "target":
            logger.warning(
                "Speaker %s is using a %s reference (%.1fs). Cloning quality may "
                "be lower than for speakers with clean 6-12s clips.",
                speaker, tier, actual,
            )
        if not candidate.text.strip():
            logger.warning(
                "Speaker %s has a reference clip but no transcript for it. "
                "IndicF5 will be asked to continue without text conditioning.",
                speaker,
            )

        results[speaker] = SpeakerReference(
            speaker=speaker,
            reference_audio=path,
            reference_text=candidate.text.strip(),
            duration=actual,
            quality_score=candidate.score,
            tier=tier,
            reason=candidate.rejected_reason or candidate.quality.reason if candidate.quality else tier,
            candidate=candidate,
        )

    return results


def summarize_references(references: dict[str, SpeakerReference]) -> str:
    """One-line-per-speaker summary for the pipeline log."""
    if not references:
        return "No speakers to profile."
    lines = []
    for speaker in sorted(references):
        ref = references[speaker]
        if ref.has_audio:
            text = ref.reference_text[:60] + ("..." if len(ref.reference_text) > 60 else "")
            lines.append(
                f"  {speaker}: {ref.duration:.1f}s [{ref.tier}] "
                f"text={text!r}" if text else f"  {speaker}: {ref.duration:.1f}s [{ref.tier}] no text"
            )
        else:
            lines.append(f"  {speaker}: no reference ({ref.reason})")
    return "\n".join(lines)
