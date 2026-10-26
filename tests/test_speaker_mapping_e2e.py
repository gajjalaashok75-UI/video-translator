"""
End-to-end speaker-mapping validation (Phase 3).

This is the integration check for the promise the whole phase exists to make:

    A -> reference A, B -> reference B, A -> reference A, B -> reference B

The TTS backend is replaced with an instrumented stand-in, so what we verify is
the exact `(speaker_wav, ref_text)` pair each segment would hand to the model.
That is the whole contract. Whether IndicF5 then renders a *similar-sounding*
voice needs the real 1.4 GB checkpoint and is marked `requires_models`.
"""

import numpy as np
import pytest
import soundfile as sf

from src.core.video_translator import VideoTranslator

RATE = 16000


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------
class RecordingTTS:
    """
    Stands in for TTSEngine. Records every task instead of synthesizing.

    Deliberately implements the real routing/validation surface the pipeline
    calls, so the code under test follows its normal path.
    """

    def __init__(self):
        self.tasks = []
        self.available = ["Auto"]

    def route_model(self, target_lang_code, requested_model=None):
        return requested_model or "edge"

    def validate_reference(self, wav_path, model_name=None):
        return bool(wav_path)

    def get_available_voices(self, model, language_code):
        return self.available

    def generate_batch(self, tasks, model="edge", check_cancel_callback=None):
        self.tasks.extend(tasks)
        results = []
        for t in tasks:
            path = t["output_path"]
            # Write a real, tiny file so the pipeline's existence/size checks
            # behave as they would in production.
            sf.write(str(path), np.zeros(int(RATE * 0.5), dtype=np.float32), RATE)
            results.append(str(path))
        return results

    def unload_all(self):
        pass


def _word(text, start, end, p=0.95):
    return {"word": text, "start": start, "end": end, "probability": p}


def _speech(start, end, step=0.4, text="word"):
    words, t = [], start
    while t + step <= end + 1e-9:
        words.append(_word(text, round(t, 3), round(t + step, 3)))
        t += step
    return words


def _tone(seconds, sr=RATE, freq=180.0, amp=0.3):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def translator(tmp_path):
    """A VideoTranslator with heavy collaborators replaced by stand-ins."""
    vt = VideoTranslator.__new__(VideoTranslator)  # skip __init__'s model loads

    from src.core.session import SessionContext
    vt.session = SessionContext()
    vt.cancel_requested = False
    vt.tts_engine = RecordingTTS()
    vt.device = "cpu"
    return vt


# ---------------------------------------------------------------------------
# A / B / A / B
# ---------------------------------------------------------------------------
class TestAlternatingSpeakers:
    def _build(self, vt, tmp_path, n_turns=4, turn=7.0):
        """A/B/A/B dialogue plus the matching diarization and audio."""
        segments, diar = [], []
        speakers = ["SPEAKER_00", "SPEAKER_01"]
        t = 1.0
        for i in range(n_turns):
            spk = speakers[i % 2]
            segments.append({
                "speaker_id": spk,
                "start": t,
                "end": t + turn,
                "text": f"{spk} line {i}",
                "translated_text": f"translated {spk} line {i}",
                "words": _speech(t, t + turn),
            })
            diar.append({"start": t, "end": t + turn, "speaker": spk})
            t = t + turn + 1.5

        total = t + 2
        vocals = tmp_path / "vocals.wav"
        sf.write(str(vocals), _tone(total), RATE)
        return segments, diar, vocals, total

    def _run_reference_stage(self, vt, segments, diar, vocals, tmp_path):
        """Run exactly the Stage 4b logic from _process_pipeline."""
        from src.audio import speaker_reference as SR
        from src.utils import config

        refs = SR.build_speaker_references(
            segments=segments,
            diar_segments=diar,
            vocals_path=vocals,
            output_dir=tmp_path / "profiles",
        )
        speaker_profiles, speaker_ref_texts, speaker_ref_audio = {}, {}, {}
        for spk, ref in refs.items():
            if ref.has_audio:
                speaker_profiles[spk] = ref.reference_audio
                speaker_ref_audio[spk] = ref.reference_audio
                if ref.has_text:
                    speaker_ref_texts[spk] = ref.reference_text
        return speaker_profiles, speaker_ref_texts, speaker_ref_audio

    def test_each_segment_gets_its_own_speakers_reference(self, translator, tmp_path):
        segments, diar, vocals, _ = self._build(translator, tmp_path)
        profiles, ref_texts, ref_audio = self._run_reference_stage(
            translator, segments, diar, vocals, tmp_path
        )

        assert set(profiles) == {"SPEAKER_00", "SPEAKER_01"}
        assert profiles["SPEAKER_00"] != profiles["SPEAKER_01"]
        assert ref_texts["SPEAKER_00"] and ref_texts["SPEAKER_01"]

        # Drive the real TTS task builder.
        speaker_map = {"SPEAKER_00": "Female", "SPEAKER_01": "Male"}
        seg_dir = tmp_path / "tts"
        seg_dir.mkdir()

        list(translator._step_tts_generator(
            segments, tmp_path / "v.mp4", "te", "en",
            "edge", "Auto", False,
            True, diar, speaker_map, profiles, vocals,
            speaker_ref_texts=ref_texts,
            speaker_ref_audio=ref_audio,
        ))

        tasks = translator.tts_engine.tasks
        assert len(tasks) == len(segments)

        # THE core assertion: A/B/A/B -> A/B/A/B, with no drift.
        for i, task in enumerate(tasks):
            expected_speaker = segments[i]["speaker_id"]
            assert task["speaker_wav"] == profiles[expected_speaker], \
                f"segment {i} ({expected_speaker}) got the wrong reference audio"
            assert task["ref_text"] == ref_texts[expected_speaker], \
                f"segment {i} ({expected_speaker}) got the wrong ref_text"
            assert task["speaker_id"] == expected_speaker

        # And the mapping really did alternate rather than collapsing to one.
        wavs = [t["speaker_wav"] for t in tasks]
        assert wavs[0] == wavs[2] and wavs[1] == wavs[3]
        assert wavs[0] != wavs[1]

    def test_ref_text_always_accompanies_its_own_audio(self, translator, tmp_path):
        """
        No task may pair one speaker's audio with another's transcript. This is
        the failure that would corrupt IndicF5's duration model.
        """
        segments, diar, vocals, _ = self._build(translator, tmp_path, n_turns=8)
        profiles, ref_texts, ref_audio = self._run_reference_stage(
            translator, segments, diar, vocals, tmp_path
        )
        seg_dir = tmp_path / "tts"
        seg_dir.mkdir()

        list(translator._step_tts_generator(
            segments, tmp_path / "v.mp4", "te", "en",
            "edge", "Auto", False, True, diar, {}, profiles, vocals,
            speaker_ref_texts=ref_texts, speaker_ref_audio=ref_audio,
        ))

        audio_to_text = {profiles[s]: ref_texts[s] for s in profiles}
        for task in translator.tts_engine.tasks:
            assert task["ref_text"] == audio_to_text[task["speaker_wav"]]

    def test_many_segments_reuse_one_reference_per_speaker(self, translator, tmp_path):
        """Caching: 20 segments, 2 speakers, 2 distinct references."""
        segments, diar, vocals, _ = self._build(translator, tmp_path, n_turns=20)
        profiles, ref_texts, ref_audio = self._run_reference_stage(
            translator, segments, diar, vocals, tmp_path
        )
        seg_dir = tmp_path / "tts"
        seg_dir.mkdir()

        list(translator._step_tts_generator(
            segments, tmp_path / "v.mp4", "te", "en",
            "edge", "Auto", False, True, diar, {}, profiles, vocals,
            speaker_ref_texts=ref_texts, speaker_ref_audio=ref_audio,
        ))

        tasks = translator.tts_engine.tasks
        assert len(tasks) == 20
        assert len({t["speaker_wav"] for t in tasks}) == 2

    def test_reference_stage_is_source_language_independent(self, translator, tmp_path):
        """
        The source language must not matter: the reference is always source
        audio plus its own source-language transcript, whatever that is.
        """
        segments, diar, vocals, _ = self._build(translator, tmp_path, n_turns=4)
        # Simulate a non-English source: Devanagari words and text.
        for seg in segments:
            seg["text"] = "नमस्ते यह एक पंक्ति है"
            seg["words"] = [
                {**w, "word": "नमस्ते"} for w in seg["words"]
            ]

        profiles, ref_texts, ref_audio = self._run_reference_stage(
            translator, segments, diar, vocals, tmp_path
        )
        for spk, text in ref_texts.items():
            assert text.strip(), f"{spk} has no reference transcript"
            # The reference transcript stays in the SOURCE language; it is never
            # translated. Translated text is a separate field on the task.
            assert any("\u0900" <= ch <= "\u097f" for ch in text)

        seg_dir = tmp_path / "tts"
        seg_dir.mkdir()
        list(translator._step_tts_generator(
            segments, tmp_path / "v.mp4", "ta", "hi",
            "edge", "Auto", False, True, diar, {}, profiles, vocals,
            speaker_ref_texts=ref_texts, speaker_ref_audio=ref_audio,
        ))
        for task in translator.tts_engine.tasks:
            # ref_text is Hindi source speech; text is the Tamil translation.
            assert task["ref_text"] in ref_texts.values()
            assert task["text"].startswith("translated")

    def test_indic_ref_text_survives_sanitisation(self):
        """
        The reference transcript reaches IndicF5 as-is. If the sanitiser rejected
        Indic script, every reference would be dropped and cloning would degrade
        silently.
        """
        from src.synthesis.tts import TTSEngine

        for text in ["నమస్కారం ఇది ఒక పరీక్ష", "नमस्ते", "வணக்கம்", "ನಮಸ್ಕಾರ"]:
            assert TTSEngine()._sanitize_text(text) == text


# ---------------------------------------------------------------------------
# Cross-speaker contamination guard at the task level
# ---------------------------------------------------------------------------
class TestFallbackDoesNotMismatch:
    def test_substituted_audio_drops_the_transcript(self, translator, tmp_path):
        """
        When the fallback chain swaps in a different clip, the original
        speaker's ref_text must NOT travel with it — that pairing is the one
        that breaks IndicF5's timing.
        """
        selected = tmp_path / "selected.wav"
        substitute = tmp_path / "substitute.wav"
        sf.write(str(selected), _tone(5), RATE)
        sf.write(str(substitute), _tone(5), RATE)

        ref_texts = {"SPEAKER_00": "this is speaker zero speaking"}
        ref_audio = {"SPEAKER_00": str(selected)}

        seg = {"ref_text": None}

        # Normal case: audio is the selected one, so the text is kept.
        assert VideoTranslator._resolve_speaker_ref_text(
            "SPEAKER_00", seg, ref_texts,
            resolved_wav=selected, speaker_ref_audio=ref_audio,
        ) == "this is speaker zero speaking"

        # Substituted audio: the text must be dropped rather than mismatched.
        assert VideoTranslator._resolve_speaker_ref_text(
            "SPEAKER_00", seg, ref_texts,
            resolved_wav=substitute, speaker_ref_audio=ref_audio,
        ) is None

    def test_unknown_speaker_gets_no_transcript(self):
        assert VideoTranslator._resolve_speaker_ref_text(
            "SPEAKER_99", {"ref_text": None}, {"SPEAKER_00": "hi"},
        ) is None

    def test_missing_transcript_is_none_not_an_error(self):
        assert VideoTranslator._resolve_speaker_ref_text(
            "SPEAKER_00", {"ref_text": None}, {},
        ) is None

    def test_segment_level_ref_text_wins(self):
        assert VideoTranslator._resolve_speaker_ref_text(
            "SPEAKER_00", {"ref_text": "explicit"}, {"SPEAKER_00": "speaker"},
        ) == "explicit"


# ---------------------------------------------------------------------------
# Real model inference
# ---------------------------------------------------------------------------
@pytest.mark.requires_models
class TestIndicF5ReceivesTheReference:
    """
    Full inference: Telugu output from an English source speaker.

    Needs the gated IndicF5 checkpoint and an accepted licence, so it is skipped
    without HF_TOKEN. Everything above this class runs without any model.
    """

    def test_speaker_like_telugu_is_generated(self, tmp_path):
        import os

        from src.synthesis import indicf5_engine as E
        from src.synthesis.backends.indicf5 import IndicF5Backend

        ok, reason = E.engine_available()
        if not ok:
            pytest.skip(reason)
        if not os.getenv("HF_TOKEN"):
            pytest.skip("IndicF5 is a gated repo; set HF_TOKEN to run this test.")

        ref = tmp_path / "speaker00.wav"
        sf.write(str(ref), _tone(8, sr=24000, freq=160), 24000)

        backend = IndicF5Backend()
        out = backend.generate(
            "నమస్కారం, ఇది ఒక పరీక్ష వాక్యం.",
            tmp_path / "out.wav",
            language="te",
            speaker_wav=ref,
            ref_text="this is the english reference line",
        )
        assert out and sf.info(out).samplerate == 24000
