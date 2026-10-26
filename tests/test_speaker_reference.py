"""
Tests for automatic per-speaker reference selection (Phase 3).

The property that matters most here is that a speaker is voiced from *their own*
audio and *their own* transcript, for the whole video. A mistake in this module
is not a one-scene glitch — it mis-voices a character in every scene — so the
tests lean on contamination and stability rather than on scoring arithmetic.
"""

import numpy as np
import pytest
import soundfile as sf

from src.audio import speaker_reference as SR


SR_RATE = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _word(text, start, end, p=0.95):
    return {"word": text, "start": start, "end": end, "probability": p}


def _seg(speaker, start, end, text, words=None):
    return {
        "speaker_id": speaker,
        "start": start,
        "end": end,
        "text": text,
        "words": words if words is not None else [],
    }


def _speech(start, end, step=0.4, p=0.95, text="word"):
    """A run of words covering [start, end)."""
    words = []
    t = start
    while t + step <= end + 1e-9:
        words.append(_word(text, round(t, 3), round(t + step, 3), p))
        t += step
    return words


def _tone(seconds, sr=SR_RATE, freq=180.0, amp=0.35):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write(tmp_path, name, wave, sr=SR_RATE):
    path = tmp_path / name
    sf.write(str(path), wave, sr)
    return path


def _alternating_dialogue(n_turns=4, turn=7.0, gap=1.5):
    """
    A / B / A / B ... dialogue. Returns (segments, diar_segments, speakers).

    Each turn is a solid block of confident words, so every speaker has a
    perfectly good reference available — the interesting question is only
    whether each one gets their OWN.
    """
    segments, diar = [], []
    speakers = ["SPEAKER_00", "SPEAKER_01"]
    t = 1.0
    for i in range(n_turns):
        spk = speakers[i % 2]
        start, end = t, t + turn
        segments.append(_seg(spk, start, end, f"{spk} speech", _speech(start, end)))
        diar.append({"start": start, "end": end, "speaker": spk})
        t = end + gap
    return segments, diar, speakers


# ---------------------------------------------------------------------------
# Overlap analysis
# ---------------------------------------------------------------------------
class TestOverlapAnalysis:
    def test_no_other_speaker_is_zero(self):
        diar = [{"start": 0, "end": 10, "speaker": "A"}]
        assert SR.other_speaker_overlap(1, 5, diar, "A") == 0.0

    def test_full_overlap_is_one(self):
        diar = [
            {"start": 0, "end": 10, "speaker": "A"},
            {"start": 0, "end": 10, "speaker": "B"},
        ]
        assert SR.other_speaker_overlap(0, 10, diar, "A") == pytest.approx(1.0)

    def test_partial_overlap_is_the_covered_fraction(self):
        diar = [
            {"start": 0, "end": 10, "speaker": "A"},
            {"start": 5, "end": 10, "speaker": "B"},
        ]
        # Half of [0, 10) is B.
        assert SR.other_speaker_overlap(0, 10, diar, "A") == pytest.approx(0.5)

    def test_overlapping_turns_are_not_double_counted(self):
        diar = [
            {"start": 0, "end": 10, "speaker": "A"},
            {"start": 2, "end": 6, "speaker": "B"},
            {"start": 4, "end": 8, "speaker": "C"},
        ]
        # B and C together cover [2, 8) once, i.e. 0.6 of the window.
        assert SR.other_speaker_overlap(0, 10, diar, "A") == pytest.approx(0.6)

    def test_own_turns_never_count_as_overlap(self):
        diar = [
            {"start": 0, "end": 5, "speaker": "A"},
            {"start": 5, "end": 10, "speaker": "A"},
        ]
        assert SR.other_speaker_overlap(0, 10, diar, "A") == 0.0

    def test_zero_length_window(self):
        assert SR.other_speaker_overlap(5, 5, [], "A") == 0.0

    def test_speaker_intervals_are_merged(self):
        diar = [
            {"start": 0, "end": 5, "speaker": "A"},
            {"start": 4, "end": 9, "speaker": "A"},
            {"start": 20, "end": 25, "speaker": "B"},
        ]
        assert SR.speaker_intervals(diar, "A") == [(0.0, 9.0)]


# ---------------------------------------------------------------------------
# Candidate construction and scoring
# ---------------------------------------------------------------------------
class TestCandidateWindows:
    def test_finds_a_window_for_a_single_speaker(self):
        segments = [_seg("A", 0, 8, "hello", _speech(0, 8))]
        cands = SR.candidate_windows(segments, [{"start": 0, "end": 8, "speaker": "A"}], "A")
        assert cands
        assert cands[0].duration >= SR.HARD_MIN_SECONDS
        assert cands[0].text

    def test_only_uses_the_requested_speakers_words(self):
        segments = [
            _seg("A", 0, 8, "alpha", _speech(0, 8, text="alpha")),
            _seg("B", 8, 16, "beta", _speech(8, 16, text="beta")),
        ]
        cands = SR.candidate_windows(segments, [], "A")
        assert cands
        for cand in cands:
            assert "beta" not in cand.text
            assert cand.end <= 8.0

    def test_no_words_and_no_segments_yields_nothing(self):
        assert SR.candidate_windows([], [], "A") == []

    def test_segment_fallback_when_no_word_timings(self):
        segments = [_seg("A", 0, 8, "no word timings here")]
        cands = SR.candidate_windows(segments, [], "A")
        assert len(cands) == 1
        assert cands[0].source == "segments"
        assert cands[0].text == "no word timings here"

    def test_windows_never_span_a_long_pause(self):
        # Two 6s blocks separated by a 5s silence. No window may straddle it.
        words = _speech(0, 6) + _speech(11, 17)
        segments = [_seg("A", 0, 17, "split", words)]
        cands = SR.candidate_windows(segments, [], "A")
        assert cands
        for cand in cands:
            assert not (cand.start < 6 and cand.end > 11), \
                f"window {cand.start}-{cand.end} straddles the pause"

    def test_candidates_respect_the_maximum_duration(self):
        segments = [_seg("A", 0, 40, "long", _speech(0, 40))]
        cands = SR.candidate_windows(segments, [], "A")
        assert cands
        for cand in cands:
            assert cand.duration <= SR.TARGET_MAX_SECONDS + 1e-6

    def test_drops_windows_shorter_than_the_hard_minimum(self):
        segments = [_seg("A", 0, 1.0, "tiny", _speech(0, 1.0, step=0.25))]
        assert SR.candidate_windows(segments, [], "A") == []

    def test_low_confidence_words_score_worse(self):
        good = [_seg("A", 0, 8, "good", _speech(0, 8, p=0.98))]
        bad = [_seg("A", 0, 8, "bad", _speech(0, 8, p=0.30))]
        good_score = SR.candidate_windows(good, [], "A")[0].score
        bad_score = SR.candidate_windows(bad, [], "A")[0].score
        assert good_score > bad_score

    def test_overlap_lowers_the_score(self):
        segments = [_seg("A", 0, 8, "solo", _speech(0, 8))]
        solo = SR.candidate_windows(segments, [{"start": 0, "end": 8, "speaker": "A"}], "A")[0]
        duet = SR.candidate_windows(
            segments,
            [
                {"start": 0, "end": 8, "speaker": "A"},
                {"start": 0, "end": 8, "speaker": "B"},
            ],
            "A",
        )[0]
        assert duet.overlap_ratio > solo.overlap_ratio
        assert duet.score < solo.score

    def test_intro_position_is_penalised(self):
        early = [_seg("A", 0, 8, "early", _speech(0, 8))]
        late = [_seg("A", 30, 38, "late", _speech(30, 38))]
        early_score = SR.candidate_windows(early, [], "A")[0].score
        late_score = SR.candidate_windows(late, [], "A")[0].score
        assert late_score > early_score

    def test_target_band_window_beats_a_shorter_confident_one(self):
        """
        A clean 6-12s window must outrank a shorter, more confident one.

        On real VoxConverse clips the duration reward used to be smaller than
        the confidence spread between windows, so a ~3.5s clip at 0.99 beat a
        clean 10s clip at 0.84 and most references fell outside the band.
        """
        segments = [
            _seg("A", 30, 33.5, "short", _speech(30, 33.5, p=0.99)),
            _seg("A", 50, 60, "long", _speech(50, 60, p=0.84)),
        ]
        best = SR.candidate_windows(segments, [], "A")[0]
        assert best.duration >= SR.TARGET_MIN_SECONDS
        assert best.duration <= SR.TARGET_MAX_SECONDS

    def test_near_duplicate_windows_are_collapsed(self):
        segments = [_seg("A", 0, 12, "one long run", _speech(0, 12))]
        cands = SR.candidate_windows(segments, [], "A")
        for i, a in enumerate(cands):
            for b in cands[i + 1:]:
                assert not SR._heavy_overlap(a, b), f"{a.start}-{a.end} vs {b.start}-{b.end}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_equal_scores_pick_the_earliest_window(self):
        # Two identical, separated blocks. Nothing distinguishes them but time.
        words = _speech(5, 13) + _speech(20, 28)
        segments = [_seg("A", 5, 28, "twice", words)]
        cands = SR.candidate_windows(segments, [], "A")
        assert cands[0].start == pytest.approx(5.0)
        assert cands[0].score == pytest.approx(
            max(c.score for c in cands if c.start > 15), abs=1e-9
        )

    def test_repeated_selection_is_stable(self):
        segments, diar, _ = _alternating_dialogue()
        runs = [SR.select_reference(segments, diar, "SPEAKER_00")[0] for _ in range(5)]
        assert all(r.start == runs[0].start and r.end == runs[0].end for r in runs)

    def test_same_speaker_keeps_one_reference_across_many_segments(self, tmp_path):
        # Speaker A speaks 6 separate times; they must all resolve to one clip.
        segments, diar = [], []
        t = 1.0
        for i in range(6):
            segments.append(_seg("A", t, t + 7, f"line {i}", _speech(t, t + 7)))
            diar.append({"start": t, "end": t + 7, "speaker": "A"})
            t += 9
        # Plus a second speaker so it is a genuine multi-speaker file.
        segments.append(_seg("B", t, t + 7, "other", _speech(t, t + 7)))
        diar.append({"start": t, "end": t + 7, "speaker": "B"})

        audio = _tone(t + 10)
        refs = SR.build_speaker_references(
            segments, diar, _write(tmp_path, "v.wav", audio), tmp_path / "out"
        )
        assert refs["A"].reference_audio
        chosen = refs["A"].candidate
        # Every A-segment overlaps the chosen window? No — but the chosen window
        # must be one of A's own turns, not B's.
        assert chosen.speaker == "A"
        assert not any(
            d["start"] < chosen.end and d["end"] > chosen.start and d["speaker"] == "B"
            for d in diar
        )


# ---------------------------------------------------------------------------
# Audio quality
# ---------------------------------------------------------------------------
class TestAudioQuality:
    def test_clean_tone_is_acceptable(self):
        q = SR.measure_audio_quality(_tone(8.0), SR_RATE)
        assert q.acceptable
        assert q.clipping_ratio == 0.0
        assert q.silence_ratio < 0.2

    def test_silence_is_rejected(self):
        q = SR.measure_audio_quality(np.zeros(SR_RATE * 8, dtype=np.float32), SR_RATE)
        assert not q.acceptable
        assert "quiet" in q.reason or "silence" in q.reason

    def test_clipping_is_rejected(self):
        wave = np.ones(SR_RATE * 8, dtype=np.float32)
        q = SR.measure_audio_quality(wave, SR_RATE)
        assert not q.acceptable
        assert "clipping" in q.reason

    def test_mostly_silent_clip_is_rejected(self):
        wave = np.zeros(SR_RATE * 8, dtype=np.float32)
        wave[: int(SR_RATE * 0.5)] = _tone(0.5)
        q = SR.measure_audio_quality(wave, SR_RATE)
        assert not q.acceptable

    def test_too_short_is_rejected(self):
        q = SR.measure_audio_quality(_tone(1.0), SR_RATE)
        assert not q.acceptable
        assert "short" in q.reason

    def test_empty_is_rejected(self):
        q = SR.measure_audio_quality(np.zeros(0, dtype=np.float32), SR_RATE)
        assert not q.acceptable
        assert q.reason == "empty"

    def test_quiet_but_clean_recording_is_judged_relative_to_itself(self):
        # A softly mastered film must not be rejected wholesale; silence is
        # measured against the clip's own peak.
        q = SR.measure_audio_quality(_tone(8.0, amp=0.05), SR_RATE)
        assert q.silence_ratio < 0.2


# ---------------------------------------------------------------------------
# Selection tiers and fallbacks
# ---------------------------------------------------------------------------
class TestSelectionTiers:
    def test_long_clean_speech_is_a_target_reference(self):
        segments = [_seg("A", 1, 12, "hi", _speech(1, 12))]
        diar = [{"start": 1, "end": 12, "speaker": "A"}]
        cand, tier = SR.select_reference(segments, diar, "A", _tone(15), SR_RATE)
        assert tier == "target"
        assert cand.duration >= SR.TARGET_MIN_SECONDS

    def test_short_but_clean_speech_is_marked_short(self):
        segments = [_seg("A", 1, 4, "hi", _speech(1, 4, step=0.3))]
        diar = [{"start": 1, "end": 4, "speaker": "A"}]
        cand, tier = SR.select_reference(segments, diar, "A", _tone(6), SR_RATE)
        assert tier == "short"
        assert cand.duration < SR.TARGET_MIN_SECONDS

    def test_speaker_with_no_speech_returns_none(self):
        cand, tier = SR.select_reference([], [], "GHOST", _tone(5), SR_RATE)
        assert cand is None and tier == "none"

    def test_all_overlapped_falls_back_to_degraded_not_silence(self):
        segments = [_seg("A", 0, 8, "duet", _speech(0, 8))]
        diar = [
            {"start": 0, "end": 8, "speaker": "A"},
            {"start": 0, "end": 8, "speaker": "B"},
        ]
        cand, tier = SR.select_reference(segments, diar, "A", _tone(9), SR_RATE)
        assert tier == "degraded"
        assert cand is not None
        assert cand.overlap_ratio == pytest.approx(1.0)

    def test_unusable_audio_is_degraded(self):
        segments = [_seg("A", 0, 8, "quiet", _speech(0, 8))]
        diar = [{"start": 0, "end": 8, "speaker": "A"}]
        cand, tier = SR.select_reference(
            segments, diar, "A", np.zeros(SR_RATE * 9, dtype=np.float32), SR_RATE
        )
        assert tier == "degraded"
        assert cand is not None

    def test_missing_waveform_falls_back_to_transcript_ranking(self):
        segments = [_seg("A", 1, 12, "hi", _speech(1, 12))]
        diar = [{"start": 1, "end": 12, "speaker": "A"}]
        cand, tier = SR.select_reference(segments, diar, "A", None, 0)
        assert cand is not None
        assert tier == "target"

    def test_overlapped_candidate_is_preferred_against_by_a_clean_one(self):
        """
        The core promise of the overlap rule: when a clean window exists, it is
        chosen over a longer but contaminated one.
        """
        words = _speech(0, 6) + _speech(10, 20)  # short clean, long overlapped
        segments = [_seg("A", 0, 20, "both", words)]
        diar = [
            {"start": 0, "end": 6, "speaker": "A"},
            {"start": 10, "end": 20, "speaker": "A"},
            {"start": 10, "end": 20, "speaker": "B"},
        ]
        cand, tier = SR.select_reference(segments, diar, "A", _tone(22), SR_RATE)
        # "short" is correct here: the clean run is ~5.6s, under the 6s target.
        # The point is that it won over the longer, overlapped alternative.
        assert tier in ("target", "short")
        assert cand.overlap_ratio <= SR.MAX_OVERLAP_RATIO
        assert cand.end <= 6.5
        assert cand.start < 6.0


# ---------------------------------------------------------------------------
# Extraction and end-to-end profile building
# ---------------------------------------------------------------------------
class TestBuildSpeakerReferences:
    def test_two_speakers_get_different_audio_and_matching_text(self, tmp_path):
        segments, diar, _ = _alternating_dialogue()
        audio = _tone(60)
        refs = SR.build_speaker_references(
            segments, diar, _write(tmp_path, "movie.wav", audio), tmp_path / "out"
        )

        a, b = refs["SPEAKER_00"], refs["SPEAKER_01"]
        assert a.has_audio and b.has_audio
        assert a.reference_audio != b.reference_audio

        # Text must describe the audio it is paired with: check the clip really
        # sits inside that speaker's own turns.
        for ref in (a, b):
            start, end = ref.candidate.start, ref.candidate.end
            owner = [
                d["speaker"] for d in diar
                if d["start"] < end - 0.01 and d["end"] > start + 0.01
            ]
            assert owner == [ref.speaker], f"{ref.speaker} clip spans {owner}"

    def test_reference_text_comes_from_the_selected_window(self, tmp_path):
        words = _speech(2, 10, text="alpha")
        segments = [_seg("A", 2, 10, "alpha alpha", words)]
        diar = [{"start": 2, "end": 10, "speaker": "A"}]
        refs = SR.build_speaker_references(
            segments, diar, _write(tmp_path, "m.wav", _tone(12)), tmp_path / "out"
        )
        ref = refs["A"]
        assert ref.has_text
        # The transcript is the words inside the chosen window, not the segment.
        assert "alpha" in ref.reference_text
        assert ref.reference_text.count("alpha") == ref.candidate.word_count

    def test_written_clip_duration_matches_the_record(self, tmp_path):
        segments, diar, _ = _alternating_dialogue()
        refs = SR.build_speaker_references(
            segments, diar, _write(tmp_path, "m.wav", _tone(60)), tmp_path / "out"
        )
        for ref in refs.values():
            if ref.has_audio:
                info = sf.info(ref.reference_audio)
                assert info.duration == pytest.approx(ref.duration, abs=0.02)

    def test_no_speakers_yields_empty_mapping(self, tmp_path):
        assert SR.build_speaker_references([], [], _write(tmp_path, "m.wav", _tone(3)), tmp_path / "o") == {}

    def test_speaker_with_no_speech_has_no_audio_but_is_reported(self, tmp_path):
        # B is diarized but never actually says anything transcribable.
        segments = [_seg("A", 0, 10, "solo", _speech(0, 10))]
        diar = [
            {"start": 0, "end": 10, "speaker": "A"},
            {"start": 11, "end": 12, "speaker": "B"},
        ]
        refs = SR.build_speaker_references(
            segments, diar, _write(tmp_path, "m.wav", _tone(15)), tmp_path / "out"
        )
        assert refs["A"].has_audio
        assert refs["B"].reference_audio is None
        assert refs["B"].tier == "none"
        assert refs["B"].reason

    def test_missing_vocals_does_not_raise(self, tmp_path):
        segments, diar, _ = _alternating_dialogue()
        refs = SR.build_speaker_references(
            segments, diar, tmp_path / "does_not_exist.wav", tmp_path / "out"
        )
        assert set(refs) == {"SPEAKER_00", "SPEAKER_01"}
        assert all(not r.has_audio for r in refs.values())

    def test_references_are_reused_not_rebuilt_per_segment(self, tmp_path, monkeypatch):
        """
        Extraction must happen once per speaker, not once per dialogue segment.
        A two-hour movie has thousands of segments; re-cutting the reference for
        each would dominate the run.
        """
        segments, diar, _ = _alternating_dialogue(n_turns=20)
        calls = {"n": 0}
        original = SR.extract_reference_audio

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(SR, "extract_reference_audio", counting)
        refs = SR.build_speaker_references(
            segments, diar, _write(tmp_path, "m.wav", _tone(400)), tmp_path / "out"
        )
        assert len(refs) == 2
        assert calls["n"] == 2, f"expected 2 extractions, got {calls['n']}"

    def test_no_cross_speaker_contamination(self, tmp_path):
        """
        The central invariant. Each speaker's transcript must contain only their
        own words, and their clip must not overlap another speaker's turn.
        """
        segments = [
            _seg("A", 0, 8, "alpha", _speech(0, 8, text="alpha")),
            _seg("B", 10, 18, "beta", _speech(10, 18, text="beta")),
        ]
        diar = [
            {"start": 0, "end": 8, "speaker": "A"},
            {"start": 10, "end": 18, "speaker": "B"},
        ]
        refs = SR.build_speaker_references(
            segments, diar, _write(tmp_path, "m.wav", _tone(20)), tmp_path / "out"
        )
        assert "beta" not in refs["A"].reference_text
        assert "alpha" not in refs["B"].reference_text
        assert refs["A"].reference_text.strip()
        assert refs["B"].reference_text.strip()


class TestExtractReferenceAudio:
    def test_clip_length_matches_the_requested_span(self, tmp_path):
        audio = _tone(10)
        path, duration = SR.extract_reference_audio(audio, SR_RATE, 2.0, 9.0, tmp_path / "r.wav")
        assert duration == pytest.approx(7.0, abs=0.01)
        assert sf.info(path).duration == pytest.approx(7.0, abs=0.01)

    def test_clip_is_normalised_and_dc_free(self, tmp_path):
        audio = _tone(10, amp=0.02) + 0.1  # quiet, with a DC offset
        path, _ = SR.extract_reference_audio(audio, SR_RATE, 1.0, 9.0, tmp_path / "r.wav")
        data, _ = sf.read(path, dtype="float32")
        assert abs(float(np.mean(data))) < 0.01
        assert float(np.max(np.abs(data))) == pytest.approx(10 ** (-3 / 20), abs=0.02)

    def test_edges_are_faded(self, tmp_path):
        audio = _tone(10)
        path, _ = SR.extract_reference_audio(audio, SR_RATE, 1.0, 9.0, tmp_path / "r.wav")
        data, _ = sf.read(path, dtype="float32")
        assert abs(float(data[0])) < 1e-3
        assert abs(float(data[-1])) < 1e-3

    def test_empty_span_raises(self, tmp_path):
        with pytest.raises(ValueError):
            SR.extract_reference_audio(_tone(5), SR_RATE, 9.0, 9.0, tmp_path / "r.wav")

    def test_clip_does_not_change_the_timeline(self, tmp_path):
        # A constant-amplitude tone must stay constant throughout: no internal
        # silence stripping, which would desynchronise the clip from its text.
        audio = _tone(10, amp=0.3)
        path, duration = SR.extract_reference_audio(audio, SR_RATE, 1.0, 9.0, tmp_path / "r.wav")
        data, _ = sf.read(path, dtype="float32")

        assert duration == pytest.approx(8.0, abs=0.01)

        # Compare energy in each half-second block across the interior; a
        # time-preserving cut leaves them all roughly equal.
        frame = int(0.5 * SR_RATE)
        blocks = [
            float(np.sqrt(np.mean(data[i:i + frame] ** 2)))
            for i in range(int(0.2 * SR_RATE), len(data) - frame, frame)
        ]
        assert len(blocks) >= 10
        assert min(blocks) > 0.9 * max(blocks)


class TestSummarize:
    def test_summary_mentions_each_speaker(self, tmp_path):
        segments, diar, _ = _alternating_dialogue()
        refs = SR.build_speaker_references(
            segments, diar, _write(tmp_path, "m.wav", _tone(60)), tmp_path / "out"
        )
        text = SR.summarize_references(refs)
        assert "SPEAKER_00" in text and "SPEAKER_01" in text

    def test_summary_handles_no_speakers(self):
        assert "No speakers" in SR.summarize_references({})


class TestAdjacentSpeakerContamination:
    """
    The harder contamination case: two speakers back-to-back with a gap small
    enough that a naive window builder would happily splice them into one
    "reference". Selecting on speaker_id is what prevents it, and this test
    fails if that check is ever dropped.
    """

    def _adjacent(self):
        # A speaks 0-8s, B speaks 8.2-16.2s: only a 0.2s gap between them.
        segments = [
            _seg("A", 0.0, 8.0, "alpha alpha", _speech(0.0, 8.0, text="alpha")),
            _seg("B", 8.2, 16.2, "beta beta", _speech(8.2, 16.2, text="beta")),
        ]
        diar = [
            {"start": 0.0, "end": 8.0, "speaker": "A"},
            {"start": 8.2, "end": 16.2, "speaker": "B"},
        ]
        return segments, diar

    def test_no_window_mixes_two_speakers(self):
        segments, diar = self._adjacent()
        for speaker, own, other in (("A", "alpha", "beta"), ("B", "beta", "alpha")):
            for cand in SR.candidate_windows(segments, diar, speaker):
                assert other not in cand.text, \
                    f"{speaker} candidate {cand.start}-{cand.end} contains {other!r}"
                assert own in cand.text

    def test_reference_clip_stays_inside_one_speakers_turn(self, tmp_path):
        segments, diar = self._adjacent()
        refs = SR.build_speaker_references(
            segments, diar, _write(tmp_path, "m.wav", _tone(18)), tmp_path / "out"
        )
        for speaker, own, other in (("A", "alpha", "beta"), ("B", "beta", "alpha")):
            ref = refs[speaker]
            assert ref.has_audio
            assert other not in ref.reference_text
            assert own in ref.reference_text
            # And the audio window itself must not cross into the other's turn.
            owner = [
                d["speaker"] for d in diar
                if d["start"] < ref.candidate.end - 0.01
                and d["end"] > ref.candidate.start + 0.01
            ]
            assert owner == [speaker], f"{speaker} window covers {owner}"


class TestDoesNotSimplyPickTheLongest:
    """
    The explicit requirement: selection must not degenerate to "longest segment
    wins". A long, poorly-transcribed stretch is worse conditioning than a
    shorter, confident one, so the score has to be able to override length.
    """

    def _long_poor_vs_short_good(self):
        # 12s of low-confidence speech (the longest available window) ...
        poor = _speech(3.0, 15.0, p=0.35)
        # ... versus 7s of confident speech later in the file.
        good = _speech(20.0, 27.0, p=0.98)
        segments = [_seg("A", 3.0, 27.0, "mixed", poor + good)]
        diar = [
            {"start": 3.0, "end": 15.0, "speaker": "A"},
            {"start": 20.0, "end": 27.0, "speaker": "A"},
        ]
        return segments, diar

    def test_shorter_confident_window_beats_longer_poor_one(self):
        segments, diar = self._long_poor_vs_short_good()
        cands = SR.candidate_windows(segments, diar, "A")
        assert cands

        longest = max(cands, key=lambda c: c.duration)
        best = cands[0]

        assert best.score > longest.score, "the longest window scored best"
        assert best.duration < longest.duration, \
            "selection collapsed to the longest window"
        assert best.word_confidence > longest.word_confidence

    def test_selected_reference_is_not_the_longest(self):
        segments, diar = self._long_poor_vs_short_good()
        cand, tier = SR.select_reference(segments, diar, "A", _tone(30), SR_RATE)
        assert cand.word_confidence > 0.9
        assert cand.duration < 12.0

    def test_clean_short_wins_over_overlapped_long(self):
        # 10s overlapped by another speaker vs 6s clean.
        overlapped = _speech(2.0, 12.0)
        clean = _speech(16.0, 22.0)
        segments = [_seg("A", 2.0, 22.0, "mixed", overlapped + clean)]
        diar = [
            {"start": 2.0, "end": 12.0, "speaker": "A"},
            {"start": 2.0, "end": 12.0, "speaker": "B"},
            {"start": 16.0, "end": 22.0, "speaker": "A"},
        ]
        cand, _ = SR.select_reference(segments, diar, "A", _tone(24), SR_RATE)
        assert cand.overlap_ratio <= SR.MAX_OVERLAP_RATIO
        assert cand.start >= 15.0, f"picked the overlapped window at {cand.start}"
