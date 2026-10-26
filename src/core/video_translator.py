from src.utils import config
import torch
import gc
import inspect
import logging
import os
import time
from pathlib import Path
import soundfile as sf
from src.core.session import SessionContext

# Component imports
from src.audio.separator import AudioSeparator
from src.audio.transcription import Transcriber
from src.translation.text_translator import Translator
from src.synthesis.tts import TTSEngine
from src.processing.synchronization import AudioSynchronizer
from src.processing.timing import (
    SegmentTiming, TimingDiagnostics, build_timings,
)
from src.core import jobs as joblib
from src.core.cache import ArtifactCache, reference_identity, tts_key
from src.core.manifest import (
    Manifest, STAGE_COMPLETE, STAGE_FAILED, STAGE_PARTIAL, STAGE_PENDING,
    STAGE_RUNNING, fingerprint, software_versions,
)
from src.core.retry import run_with_retry
from src.processing.duration_model import (
    DurationModel, measure_reference_durations, plan_native_fit,
)
from src.translation.duration_aware import SpeechShortener, should_shorten
from src.synthesis.capabilities import honours_duration
from src.processing.video import VideoProcessor
from src.audio.diarization import Diarizer
from src.audio import speaker_reference

from src.processing.lipsync import LipSyncer
from src.translation.visual_translator import VisualTranslator
from src.processing.voice_enhancement import VoiceEnhancer
from typing import Generator, Literal, Any, Optional

from src.utils import patches
from src.utils.chunker import VideoChunker

logger = logging.getLogger(__name__)


class CancelledError(Exception):
    """Raised when the pipeline is cancelled by the user."""
    pass


class VideoTranslator:
    """
    Central controller for the video translation pipeline.
    Enforces strict VRAM management by ensuring only one heavy model is loaded at a time.
    """

    # Durable job state, declared on the class so an instance built without
    # __init__ (which tests do, to exercise one stage in isolation) still reads
    # these as "no job" instead of raising AttributeError.
    job = None
    manifest = None
    cache = None
    # Set while the chunked path is processing one chunk. Chunked runs share one
    # manifest, so every stage fingerprint and artifact path must be namespaced
    # by chunk or chunk 2 would silently reuse chunk 1's transcript.
    _chunk_index = None
    # Subtitle files written by the subtitles stage, held so the final mux can
    # embed or burn them without re-deriving the path. Empty until that stage
    # runs, and reset per render.
    _pending_subtitles: dict = {}

    def __init__(self,
                 separator: AudioSeparator | None = None,
                 transcriber: Transcriber | None = None,
                 translator: Translator | None = None,
                 tts_engine: TTSEngine | None = None,
                 synchronizer: AudioSynchronizer | None = None,
                 processor: VideoProcessor | None = None,
                 diarizer: Diarizer | None = None,
                 lipsyncer: LipSyncer | None = None,
                 visual_translator: VisualTranslator | None = None,
                 voice_enhancer: VoiceEnhancer | None = None,
                 live_portrait_mode: str = "ort") -> None:
        
        self.separator = separator or AudioSeparator()
        self.transcriber = transcriber or Transcriber()
        self.translator = translator or Translator()
        self.tts_engine = tts_engine or TTSEngine()
        self.synchronizer = synchronizer or AudioSynchronizer()
        # Per-run timing report, replaced at the start of every render.
        self.timing_diagnostics = TimingDiagnostics()
        self.processor = processor or VideoProcessor()
        self.diarizer = diarizer or Diarizer()
        self.lipsyncer = lipsyncer or LipSyncer(acceleration=live_portrait_mode)
        self.visual_translator = visual_translator or VisualTranslator()
        self.voice_enhancer = voice_enhancer or VoiceEnhancer()
        
        # Track currently loaded model to avoid redundant unloads/loads
        self.current_model = None
        
        # Session state
        self.session = SessionContext()
        
        # Abort flag
        self.cancel_requested = False

        # Durable job state (Phase 6). Set by process_video; None means the
        # caller asked for a plain, non-resumable run.
        self.job = None
        self.manifest = None
        self.cache = None

    def abort(self):
        """Signals the pipeline to stop at the next safe cancellation point."""
        logger.warning("Pipeline cancel requested.")
        self.cancel_requested = True
        
    def _check_cancel(self):
        """Raises CancelledError if cancellation is requested."""
        if self.cancel_requested:
            # Mark the job as stopped rather than failed. A cancelled job is
            # resumable by design: that is the difference between "the user
            # stopped it" and "it broke".
            if self.manifest:
                self.manifest.status = "CANCELLED"
                self.manifest.note("Cancelled by user")
                for name, rec in self.manifest.stages.items():
                    if rec.status == STAGE_RUNNING:
                        rec.status = STAGE_PARTIAL
                        rec.finished_at = time.time()
                self.manifest.save()
            raise CancelledError("Processing Cancelled by User")

    def _get_assigned_voice(self, speaker_id, available_voices):
        """
        Deterministically assigns an unused voice to a speaker ID for the session.
        """
        # Check existing assignment
        existing = self.session.get_voice(speaker_id)
        if existing:
            return existing
            
        # Find first unused voice
        for voice in available_voices:
            if not self.session.is_voice_used(voice):
                self.session.assign_voice(speaker_id, voice)
                return voice
                
        # If all used, cycle using modulo (fallback)
        idx = len(self.session.speaker_voice_map) % len(available_voices)
        return available_voices[idx]

    def _resolve_speaker_reference(self, speaker_id: Optional[str], speaker_profiles: dict, vocals_path: str | Path, tts_model_name: str) -> tuple[Optional[str], bool]:
        """
        Determines the best reference audio for a speaker.
        Returns: (path_to_wav, is_fallback)
        """
        # Case 1: No speaker ID (Single speaker mode)
        if not speaker_id:
             return vocals_path, False

        # Case 2: Speaker has a clean profile
        if speaker_id in speaker_profiles:
            profile_path = speaker_profiles[speaker_id]
            # Use public validation API
            if self.tts_engine.validate_reference(profile_path, model_name=tts_model_name):
                return profile_path, False
            else:
                logger.warning(f"Profile validation failed for {speaker_id}. Fallback to generic voice.")
                return None, False

        # Case 3: Speaker identified but no profile extracted
        logger.warning(f"No clean profile for {speaker_id}. Fallback to generic voice.")
        return None, False  # Return None to signal use of generic voice, NOT full vocals

    def _extract_fallback_reference(self, vocals_path: str | Path) -> Optional[str]:
        """
        Extracts a valid speech clip from the first 30 seconds of the audio
        to use as a desperate fallback when no other reference is found.
        """
        fallback_path = self._scratch_path(f"fallback_ref_0_30_{Path(vocals_path).stem}.wav")
        if fallback_path.exists():
            return str(fallback_path)
            
        try:
            logger.info("Comparing audio... Attempting to extract fallback reference from first 30s...")
            # We can use VAD or just simple slicing if VAD isn't desired/loaded.
            # But we have 'self.transcriber.vad' usually if initialized.
            
            # Simple approach: Slice 0-30s
            data, sr = sf.read(str(vocals_path))
            
            # Limit to 30s
            max_samples = 30 * sr
            if len(data) > max_samples:
                data = data[:max_samples]
                
            # If we have VAD loaded, try to find a speech segment
            # But 'self.transcriber' might not be loaded if we are in a different stage?
            # Actually process_video controls the flow, so Transcriber is initialized but maybe 'model' unloaded.
            # VAD is lightweight.
            
            # For robustness, let's just use the first 5-10 seconds of NON-SILENCE if possible.
            # Or just save the cropped 30s if it's long enough.
            
            if len(data) < 2 * sr: # Less than 2s
                logger.warning("First 30s too short for fallback.")
                return None
                
            sf.write(str(fallback_path), data, sr)
            
            # Check if valid (using TTS engine's validator)
            if self.tts_engine._check_reference_audio(str(fallback_path), min_duration=2.0):
                 logger.info(f"Created fallback reference: {fallback_path.name}")
                 return str(fallback_path)
            else:
                 logger.warning("Extracted fallback 30s failed validation.")
                 return None
                 
        except CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to extract fallback reference: {e}")
            return None

    def unload_all_models(self):
        """
        Force unloads all known models and clears CUDA cache.
        """
        logger.info("Unloading all models and clearing CUDA cache...")
        config.debug_log("Unloading all models...")
        
        config.debug_log("Unloading Separator...")
        self.separator.unload_model()
            
        config.debug_log("Unloading Transcriber...")
        self.transcriber.unload_model()

        config.debug_log("Unloading Translator (LLM)...")
        self.translator.unload_model()
             
        config.debug_log("Unloading TTS...")
        self.tts_engine.unload_model()
            
        config.debug_log("Unloading Diarizer...")
        self.diarizer.unload_model()
            
        config.debug_log("Unloading Lipsyncer...")
        self.lipsyncer.unload_model()
            
        config.debug_log("Unloading Visual Translator...")
        self.visual_translator.unload_model()
            
        config.debug_log("Unloading Voice Enhancer...")
        self.voice_enhancer.unload_model()
        
        self.current_model = None
        
        # Force garbage collection
        config.debug_log("Running GC...")
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            config.debug_log("Emptying CUDA Cache...")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        logger.info("VRAM cleared.")
        config.debug_log("Unload Complete.")

    def load_model(self, model_type, **kwargs):
        """
        Generic loader that enforces the 'one heavy model' rule.
        """
        if self.current_model == model_type:
            return # Already loaded
            
        # Unload everything before loading a new heavy model
        self.unload_all_models()
        
        logger.info(f"Loading model: {model_type}")
        self.current_model = model_type
        
        # In the future, we can call specific load methods here if needed
        pass



    def get_available_tts_voices(self, model_name, language_code):
        """Wrapper to get available voices from TTSEngine."""
        return self.tts_engine.get_available_voices(model_name, language_code)

    def process_video(self,
                      video_path: str | Path,
                      source_lang: str = "auto",
                      target_lang: str = "en",
                      audio_model_name: str = "Torchaudio HDemucs (Recommended)",
                      tts_model_name: str = "edge",
                      translation_model_name: str = "Google Translate (Online, Fast)",
                      context_model_name: str = "Tencent HY-MT1.5 (Local, Better Context)",
                      transcription_model_name: str = "Large v3 Turbo (Fast)",
                      optimize_translation: bool = False,
                      enable_diarization: bool = False,
                      diarization_model: str = "pyannote/SpeechBrain (Default)",
                      min_speakers: int = 1,
                      max_speakers: int = 10,
                      enable_time_stretch: bool = False,
                      enable_vad: bool = False,
                      vad_min_silence_duration_ms: int = 1000,
                      enable_lipsync: bool = False,
                      lipsync_model_name: str = "Wav2Lip-GAN (Low Quality - Fast)",
                      live_portrait_mode: str = "ort",
                      enable_visual_translation: bool = False,
                      ocr_model_name: str = "PaddleOCR",
                      tts_voice: str | None = None,
                      transcription_beam_size: int = 7,
                      tts_enable_cfg: bool = False,
                      enable_audio_enhancement: bool = False,
                      chunk_duration: int = 300,
                      subtitle_mode: str = config.DEFAULT_SUBTITLE_MODE,
                      hf_token: str | None = None,
                      job_id: str | None = None,
                      resume: bool = True,
                      quality_preset: str | None = None) -> Generator[tuple[Literal["log", "progress", "result"], Any] | list, None, None]:
        """
        Orchestrates the full pipeline, handling chunking if necessary.

        :param quality_preset: Optional internal preset name ("FAST", "BALANCED",
            "HIGH"). Explicitly passing it overrides the quality-related
            arguments (lip-sync, enhancement); passing None keeps whatever the
            caller supplied, which is the default and leaves behaviour
            unchanged.
        :param job_id: Reopen this job's workspace instead of creating a new one.
        :param resume: When False, always start a fresh job even if `job_id`
                       matches an existing one. Tests use this to get the old
                       behaviour (no durable state on disk).
        """
        if quality_preset is not None:
            overrides = config.preset_config(quality_preset)
            enable_lipsync = overrides.get("enable_lipsync", enable_lipsync)
            lipsync_model_name = overrides.get("lipsync_model_name", lipsync_model_name)
            live_portrait_mode = overrides.get("live_portrait_mode", live_portrait_mode)
            enable_audio_enhancement = overrides.get(
                "enable_audio_enhancement", enable_audio_enhancement)
        video_path = config.validate_path(video_path, must_exist=False)

        # Durable workspace (Phase 6). A render is a job with a stable id; its
        # intermediates survive a restart, so a crash at hour three does not
        # cost hours one and two.
        if resume:
            self.job = joblib.Job.open_or_create(job_id, prefix=Path(str(video_path)).stem)
            request_config = self._run_config(locals())
            self.manifest = Manifest.load_or_create(
                self.job,
                source_path=video_path,
                source_lang=source_lang,
                target_lang=target_lang,
                run_config=request_config,
                software=software_versions(),
            )
            self.cache = ArtifactCache(self.job)
            # Hand the translator the job-scoped, content-addressed store so
            # translations are persisted to the job and reused across a restart.
            # This is the single authoritative production translation cache; the
            # translator's own temp file is only used when there is no job.
            self.translator.job_cache = self.cache
            self.manifest.job_id = self.job.job_id
            self.manifest.save()
            # Fail safe on configuration drift. Reopening a job keeps the
            # *stored* configuration, and fingerprints are computed from it, so
            # calling this with different settings would recompute the old
            # fingerprints, match, and silently hand back the previous run's
            # output. `resume_job` is unaffected: it replays the recorded config,
            # so the two always agree.
            #
            # The one exception is the re-renderable set: flipping lip-sync or
            # enhancement changes no cached feature (ASR, diarization,
            # references, translation, TTS), so those stages are invalidated and
            # re-run while everything expensive stays put. Any other difference
            # is still refused, because the cached output was made under the old
            # settings.
            drifted = self.manifest.differs_from(request_config)
            if drifted and set(drifted) <= set(config.RE_RENDERABLE_CONFIG_KEYS):
                report = self.manifest.reconcile(
                    request_config,
                    reason="re-rendering lip-sync/enhancement only")
                yield ("log",
                       f"Re-rendering {', '.join(drifted)} without redoing "
                       f"upstream work (invalidated: "
                       f"{', '.join(report['invalidated']) or 'none'}).")
            elif drifted:
                raise joblib.JobError(
                    f"Job {self.job.job_id} was created with different settings "
                    f"({', '.join(drifted)}); resume it with resume_job(job_id) "
                    "to continue it as it was, or start a new job with a new "
                    "job_id.")
            yield ("log", f"Job {self.job.job_id} at {self.job.root}")
            yield from self._emit_job_progress()
        else:
            self.job = None
            self.manifest = None
            self.cache = None
            self.translator.job_cache = None
        
        # Only attempt chunking if the file actually exists (prevents failures in tests with mocked paths)
        if hasattr(video_path, 'exists') and video_path.exists() and video_path.stat().st_size > 0:
            chunker = VideoChunker(max_duration_sec=int(chunk_duration))
            if chunker.should_chunk(video_path):
                yield ("log", f"Video is long (> {chunker.max_duration}s). Switching to Chunked Processing...")
                yield from self._process_chunked(video_path, chunker,
                    source_lang=source_lang, target_lang=target_lang,
                    audio_model_name=audio_model_name, tts_model_name=tts_model_name,
                    translation_model_name=translation_model_name, context_model_name=context_model_name,
                    transcription_model_name=transcription_model_name, optimize_translation=optimize_translation,
                    enable_diarization=enable_diarization, diarization_model=diarization_model,
                    min_speakers=min_speakers, max_speakers=max_speakers,
                    enable_time_stretch=enable_time_stretch, enable_vad=enable_vad,
                    vad_min_silence_duration_ms=vad_min_silence_duration_ms,
                    enable_lipsync=enable_lipsync, lipsync_model_name=lipsync_model_name,
                    live_portrait_mode=live_portrait_mode,
                    enable_visual_translation=enable_visual_translation, ocr_model_name=ocr_model_name,
                    tts_voice=tts_voice, transcription_beam_size=transcription_beam_size,
                    tts_enable_cfg=tts_enable_cfg, enable_audio_enhancement=enable_audio_enhancement,
                    subtitle_mode=subtitle_mode,
                    hf_token=hf_token)
                return

        # Reset cancel flag
        self.cancel_requested = False
        self._pending_subtitles = {}

        # Normal pipeline
        yield from self._process_pipeline(
            video_path=video_path, source_lang=source_lang, target_lang=target_lang,
            audio_model_name=audio_model_name, tts_model_name=tts_model_name,
            translation_model_name=translation_model_name, context_model_name=context_model_name,
            transcription_model_name=transcription_model_name, optimize_translation=optimize_translation,
            enable_diarization=enable_diarization, diarization_model=diarization_model,
            min_speakers=min_speakers, max_speakers=max_speakers,
            enable_time_stretch=enable_time_stretch, enable_vad=enable_vad,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
            enable_lipsync=enable_lipsync, lipsync_model_name=lipsync_model_name,
            live_portrait_mode=live_portrait_mode,
            enable_visual_translation=enable_visual_translation, ocr_model_name=ocr_model_name,
            tts_voice=tts_voice, transcription_beam_size=transcription_beam_size,
            tts_enable_cfg=tts_enable_cfg, enable_audio_enhancement=enable_audio_enhancement,
            subtitle_mode=subtitle_mode,
            hf_token=hf_token
        )

    def resume_job(self, job_id: str, hf_token: str | None = None):
        """
        Continue a previous render from where it stopped.

        Everything needed to restart is already on disk: the manifest holds the
        source, the configuration, and each stage's status, and the job
        directory holds the artifacts. So a resume is the same pipeline run
        again with the same job id, and the stage fingerprints decide what is
        skipped. Nothing is special-cased per stage here, which is why a stage
        added later resumes correctly without being told about this method.

        `hf_token` is not persisted (a credential has no business in a
        manifest), so a gated model needs it passed again.

        :param job_id: The job to continue.
        :param hf_token: Token for gated models, if the run needs one.
        :raises joblib.JobError: If the job or its manifest is not usable.
        """
        job = joblib.Job.load(job_id)
        if job is None:
            raise joblib.JobError(f"No job {job_id!r} on disk")

        manifest = Manifest.load(job)
        if manifest is None:
            raise joblib.JobError(f"Job {job_id!r} has no readable manifest")

        source_path = (manifest.source or {}).get("path")
        if not source_path:
            raise joblib.JobError(f"Job {job_id!r} does not record its source video")
        if not Path(source_path).exists():
            raise joblib.JobError(
                f"Job {job_id!r} was created from {source_path}, which is gone; "
                "restore the file or start a new job")

        if manifest.status == "COMPLETE":
            final = manifest.artifacts.get("final_output")
            if final and Path(str(final)).exists():
                yield ("log", f"Job {job_id} is already complete; nothing to resume.")
                yield ("result", str(final))
                return
            # The recorded output is gone. Re-running the missing stages is the
            # only way to give the caller a usable file; the stage fingerprints
            # make every surviving stage a no-op.
            yield ("log", f"Job {job_id} is marked complete but its output is "
                          "missing; re-running the missing stages.")

        # Only pass arguments the pipeline actually accepts, so a manifest
        # written by a different version cannot crash the resume with an
        # unexpected keyword.
        # Against the class's signature, not the instance's: a wrapper or test
        # double on the instance would hide the real parameter names.
        accepted = set(inspect.signature(type(self).process_video).parameters)
        kwargs = {k: v for k, v in (manifest.config or {}).items()
                  if k in accepted and k not in ("job_id", "resume", "hf_token")}

        yield ("log", f"Resuming job {job_id} from {manifest.render_progress_text()}")
        yield from self.process_video(
            source_path, job_id=job_id, resume=True, hf_token=hf_token, **kwargs)

    def _process_chunked(self, video_path: Path, chunker: VideoChunker,
                         source_lang: str, target_lang: str,
                         audio_model_name: str, tts_model_name: str,
                         translation_model_name: str, context_model_name: str,
                         transcription_model_name: str, optimize_translation: bool,
                         enable_diarization: bool, diarization_model: str,
                         min_speakers: int, max_speakers: int,
                         enable_time_stretch: bool, enable_vad: bool,
                         vad_min_silence_duration_ms: int,
                         enable_lipsync: bool, lipsync_model_name: str,
                         live_portrait_mode: str,
                         enable_visual_translation: bool, ocr_model_name: str,
                         tts_voice: str | None, transcription_beam_size: int,
                         tts_enable_cfg: bool, enable_audio_enhancement: bool,
                         subtitle_mode: str = config.DEFAULT_SUBTITLE_MODE,
                         hf_token: str | None = None):
        """
        Handles splitting, processing, and merging for long videos.
        """
        # 1. Global Diarization (if enabled)
        # We run this on the Full Audio to ensure speaker IDs are consistent across chunks.
        precomputed_diarization = None
        full_audio = None
        if enable_diarization:
             yield ("progress", 0.05, "Global Diarization (Full Audio)...")
             yield ("log", "Running global diarization on full audio for consistency...")
             
             # Extract full audio temporarily
             full_audio = self._step_extraction(video_path)
             
             # Run Diarization Step
             self.load_model("diarization")
             # Use same params as pipeline
             precomputed_diarization = self._step_diarization(
                 full_audio, video_path, 
                 diarization_model, min_speakers, max_speakers,
                 hf_token=hf_token
             )
             self.unload_all_models()
             yield ("log", "Global diarization complete.")

        # 2. Split Video & Audio
        yield ("log", "Splitting video into chunks...")
        video_chunks = chunker.split_video(video_path)
        
        processed_chunks = []

        # Timing report for the whole video, not just the final chunk.
        run_diagnostics = TimingDiagnostics()
        
        for i, chunk in enumerate(video_chunks):
            yield ("log", f"Refining Chunk {i+1}/{len(video_chunks)}...")
            yield ("progress", (i / len(video_chunks)), f"Processing Chunk {i+1}/{len(video_chunks)}")
            
            # Resume: a chunk that already produced a valid result is not
            # reprocessed. This is what makes a crash on chunk 9 of 12 cost one
            # chunk instead of nine, which on a long movie is the whole point.
            #
            # The reuse check has to come BEFORE the PENDING reset: resetting
            # first would clear the very COMPLETE record the check looks for, so
            # every chunk would be redone and chunk-level resume would be a
            # no-op that still looked implemented.
            durable_chunk = self._valid_artifact(
                "final", f"chunk_{i:03d}.mp4", kind="video")
            if durable_chunk is not None and self._chunk_done(i, len(video_chunks)):
                processed_chunks.append(durable_chunk)
                yield ("log", f"  [Chunk {i+1}] Reusing finished chunk from a previous run.")
                continue
            if self.manifest:
                # Only now that we have committed to redoing it.
                self.manifest.upsert_chunk(i, status=STAGE_PENDING)

            # Each chunk gets its own stage namespace, so chunk 3's Demucs
            # result and chunk 4's do not overwrite each other's fingerprints.
            self._chunk_index = i
            chunk_result = None
            
            iterator = self._process_pipeline(
                video_path=chunk, source_lang=source_lang, target_lang=target_lang,
                audio_model_name=audio_model_name, tts_model_name=tts_model_name,
                translation_model_name=translation_model_name, context_model_name=context_model_name,
                transcription_model_name=transcription_model_name, optimize_translation=optimize_translation,
                enable_diarization=enable_diarization, diarization_model=diarization_model,
                min_speakers=min_speakers, max_speakers=max_speakers,
                enable_time_stretch=enable_time_stretch, enable_vad=enable_vad,
                vad_min_silence_duration_ms=vad_min_silence_duration_ms,
                enable_lipsync=enable_lipsync, lipsync_model_name=lipsync_model_name,
                live_portrait_mode=live_portrait_mode,
                enable_visual_translation=enable_visual_translation, ocr_model_name=ocr_model_name,
                tts_voice=tts_voice, transcription_beam_size=transcription_beam_size,
                tts_enable_cfg=tts_enable_cfg, enable_audio_enhancement=enable_audio_enhancement,
                precomputed_diarization=precomputed_diarization,
                subtitle_mode=subtitle_mode,
                hf_token=hf_token
            )
            for item in iterator:
                if isinstance(item, tuple):
                    if item[0] == "result":
                        chunk_result = item[1]
                    elif item[0] == "log":
                        # Indent logs for clarity
                        yield ("log", f"  [Chunk {i+1}] {item[1]}")
                    # Skip progress updates from sub-pipeline to avoid jumping 0-100
                elif isinstance(item, list):
                    # TTS segments list, ignore
                    pass
            
            if chunk_result:
                # Keep a durable copy of the finished chunk so a later crash
                # does not have to reprocess it.
                durable = self._stage_output("final", f"chunk_{i:03d}.mp4")
                if durable is not None:
                    import shutil as _shutil
                    try:
                        _shutil.copy2(str(chunk_result), str(durable))
                        processed_chunks.append(durable)
                    except OSError as exc:
                        logger.warning("Could not persist chunk %d: %s", i + 1, exc)
                        processed_chunks.append(Path(chunk_result))
                else:
                    processed_chunks.append(Path(chunk_result))
                if self.manifest:
                    self.manifest.upsert_chunk(i, status=STAGE_COMPLETE)
                    self.manifest.save()
            else:
                yield ("log", f"Error: Chunk {i+1} failed to produce result.")
                if self.manifest:
                    self.manifest.upsert_chunk(i, status=STAGE_FAILED,
                                               error="chunk produced no result")
                    self.manifest.record_error("final", "chunk produced no result",
                                               segment_index=i)
                    self.manifest.save()
                # A chunk that failed is not the end of the job: the chunks
                # already finished are on disk and a resume picks up here. Only
                # raise when nothing at all was produced, because then there is
                # no partial result worth keeping.
                if not processed_chunks:
                    raise Exception(f"Chunk {i+1} processing failed.")
                yield ("log", f"  [Chunk {i+1}] Failed; continuing with "
                              f"{len(processed_chunks)} finished chunk(s).")

            # Fold this chunk's timing records into the run-level report.
            run_diagnostics.merge(self.timing_diagnostics)

            # Critical: Clear VRAM between chunks
            self.unload_all_models()
            
        # 3. Merge Results
        yield ("progress", 0.95, "Merging Chunks...")
        yield ("log", "Merging processed chunks...")
        
        final_output = config.OUTPUT_DIR / f"translated_{video_path.name}"
        chunker.merge_videos(processed_chunks, final_output)

        self.timing_diagnostics = run_diagnostics
        self.session.timing_diagnostics = run_diagnostics.summary()
        logger.info("Timing summary (all chunks): %s", self.session.timing_diagnostics)

        if self.manifest:
            self.manifest.diagnostics["timing"] = self.session.timing_diagnostics
            counts = self.manifest.chunk_counts()
            self.manifest.diagnostics["chunks"] = counts
            self.manifest.artifacts["final_output"] = str(final_output)
            if counts["failed"]:
                self.manifest.partial_stage("final", {"chunks": counts})
            else:
                self._finish_stage("final", {"chunks": counts})
            self.manifest.status = "COMPLETE"
            self.manifest.save()

        yield ("result", str(final_output))


    def _process_pipeline(self, 
                      video_path: str | Path, 
                      source_lang: str, 
                      target_lang: str, 
                      audio_model_name: str, 
                      tts_model_name: str, 
                      translation_model_name: str, 
                      context_model_name: str,
                      transcription_model_name: str, 
                      optimize_translation: bool, 
                      enable_diarization: bool, 
                      enable_time_stretch: bool, 
                      enable_vad: bool,
                      enable_lipsync: bool,
                      enable_visual_translation: bool,
                      enable_audio_enhancement: bool = False,
                      vad_min_silence_duration_ms: int = 1000,
                      transcription_beam_size: int = 5,
                      tts_enable_cfg: bool = False,
                      diarization_model: str = "pyannote/SpeechBrain (Default)",
                      min_speakers: int = 1,
                      max_speakers: int = 10,
                      ocr_model_name: str = "PaddleOCR",
                      tts_voice: str | None = None,
                      lipsync_model_name: str | None = "wav2lip",
                      live_portrait_mode: str = "ort",
                      precomputed_diarization: tuple | None = None,
                      precomputed_audio: str | None = None,
                      subtitle_mode: str = config.DEFAULT_SUBTITLE_MODE,
                      hf_token: str | None = None) -> Generator[tuple[Literal["log", "progress", "result"], Any] | list, None, None]:
        """
        Internal pipeline logic (renamed from process_video).
        Supports precomputed_diarization for chunked execution.
        """
        
        # 0. Setup and Validation
        logger.info(f"DEBUG: VideoTranslator received live_portrait_mode='{live_portrait_mode}'")
        video_path = config.validate_path(video_path, must_exist=True)
        
        # ---------------------------------------------------------------------
        # 1. Audio Extraction Stage
        #    - Uses FFmpeg to extract audio track from video
        #    - Skip if precomputed_audio is provided (chunked mode)
        # ---------------------------------------------------------------------
        config.debug_log("Stage 1: Audio Extraction")
        if precomputed_audio:
            extracted_path = precomputed_audio
            yield ("log", "Using precomputed audio.")
        else:
            yield ("progress", 0.1, "Extracting Audio...")
            # Resume: extraction is pure ffmpeg on an unchanged source, so a
            # validated result from a previous run is reused as-is.
            extracted_path = self._valid_artifact("extraction", "audio.wav", kind="wav")
            if extracted_path is not None and self._stage_done("extraction"):
                yield ("log", "Reusing extracted audio from a previous run.")
            else:
                self._begin_stage("extraction")
                try:
                    extracted_path = self._run_stage_with_retry(
                        lambda: self._step_extraction(video_path), "audio extraction")
                except Exception as exc:
                    self._fail_stage("extraction", exc)
                    raise
                durable = self._stage_output("extraction", "audio.wav")
                if durable is not None:
                    import shutil as _shutil
                    try:
                        _shutil.copy2(str(extracted_path), str(durable))
                        extracted_path = str(durable)
                    except OSError as exc:
                        logger.warning("Could not persist extracted audio: %s", exc)
                self._finish_stage("extraction", {"path": str(extracted_path)})
                yield ("log", "Audio extracted.")
             
        # ---------------------------------------------------------------------
        # 2. Vocal Separation Stage
        #    - Uses Demucs to separate Vocals vs Background (Accompaniment)
        # ---------------------------------------------------------------------
        config.debug_log("Stage 2: Vocal Separation")
        self.load_model("demucs") 
        yield ("progress", 0.2, "Separating Vocals...")
        self._check_cancel()
        # Resume: Demucs is the single most expensive stage on a long movie.
        # Both stems must validate before the stage is considered done.
        cached_vocals = self._valid_artifact("separation", "vocals.wav", kind="wav")
        cached_bg = self._valid_artifact("separation", "accompaniment.wav", kind="wav")
        if (cached_vocals is not None and cached_bg is not None
                and self._stage_done("separation")):
            vocals_path, bg_path = str(cached_vocals), str(cached_bg)
            yield ("log", "Reusing separated stems from a previous run.")
        else:
            self._begin_stage("separation")
            try:
                vocals_path, bg_path = self._run_stage_with_retry(
                    lambda: self._step_separation(extracted_path, audio_model_name),
                    "vocal separation")
            except Exception as exc:
                self._fail_stage("separation", exc)
                raise
            for stem, produced in (("vocals.wav", vocals_path),
                                   ("accompaniment.wav", bg_path)):
                durable = self._stage_output("separation", stem)
                if durable is not None:
                    import shutil as _shutil
                    try:
                        _shutil.copy2(str(produced), str(durable))
                    except OSError as exc:
                        logger.warning("Could not persist %s: %s", stem, exc)
            if self.job:
                vocals_path = str(self.job.stage_dir("separation") / "vocals.wav")
                bg_path = str(self.job.stage_dir("separation") / "accompaniment.wav")
            self._finish_stage("separation", {"vocals": str(vocals_path),
                                             "background": str(bg_path)})
            yield ("log", f"Separation complete. Vocals: {Path(vocals_path).name}")
        
        # ---------------------------------------------------------------------
        # 3. Speaker Diarization Stage
        #    - Identifies distinct speakers and extracts voice profiles
        # ---------------------------------------------------------------------
        # Reset session state for new video
        self.session.reset()
        
        diarization_segments, speaker_map, speaker_profiles = [], {}, {}
        # Per-speaker reference audio and transcript, keyed by speaker id.
        # Populated by the reference-selection stage below (Stage 4b) and
        # consumed when the TTS tasks are built.
        speaker_ref_texts: dict = {}
        speaker_ref_audio: dict = {}
        if enable_diarization:
            if precomputed_diarization:
                 yield ("log", "Using precomputed global diarization...")
                 diarization_segments, speaker_map, speaker_profiles = precomputed_diarization
            else:
                config.debug_log("Stage 3: Diarization")
                self.load_model("diarization")
                yield ("progress", 0.25, "Diarizing...")
                self._check_cancel()
                # Resume: diarization output is durable JSON plus per-speaker
                # profile WAVs, so a resumed run does not re-run pyannote.
                cached_diar = self._load_stage_json("diarization", "segments.json")
                if cached_diar and self._stage_done("diarization"):
                    diarization_segments = cached_diar.get("segments", [])
                    speaker_map = cached_diar.get("speaker_map", {})
                    speaker_profiles = {
                        spk: path for spk, path in (cached_diar.get("profiles") or {}).items()
                        if joblib.validate_wav(Path(path))
                    }
                    yield ("log", "Reusing diarization from a previous run.")
                else:
                    self._begin_stage("diarization")
                    try:
                        diarization_segments, speaker_map, speaker_profiles = self._run_stage_with_retry(
                            lambda: self._step_diarization(
                                vocals_path, video_path, diarization_model,
                                min_speakers, max_speakers, hf_token=hf_token),
                            "diarization")
                    except Exception as exc:
                        self._fail_stage("diarization", exc)
                        raise
                    self._save_stage_json("diarization", "segments.json", {
                        "segments": diarization_segments,
                        "speaker_map": speaker_map,
                        "profiles": speaker_profiles,
                    })
                    self._finish_stage("diarization", {
                        "speakers": len(speaker_map) if speaker_map else 0})
            
        # ---------------------------------------------------------------------
        # 4. Transcription Stage
        #    - Converts speech to text (ASR) using Whisper
        #    - Incorporates VAD for clean segmentation
        # ---------------------------------------------------------------------
        config.debug_log("Stage 4: Transcription")
        self.load_model("whisper")
        yield ("progress", 0.3, "Transcribing...")
        # Resolve source/target codes
        source_code = config.get_language_code(source_lang)
        target_code = config.get_language_code(target_lang)
        
        # Resume: ASR is expensive and its output is a plain segment list.
        cached_asr = self._load_stage_json("transcription", "segments.json")
        if cached_asr and self._stage_done("transcription"):
            segments = cached_asr.get("segments", [])
            detected_lang = cached_asr.get("detected_lang", source_code)
            yield ("log", f"Reusing transcription from a previous run. {len(segments)} segments.")
        else:
            self._begin_stage("transcription")
            try:
                segments, detected_lang = self._run_stage_with_retry(
                    lambda: self._step_transcription(
                        vocals_path, source_code, transcription_model_name,
                        enable_vad, transcription_beam_size, vad_min_silence_duration_ms),
                    "transcription")
            except Exception as exc:
                self._fail_stage("transcription", exc)
                raise
            yield ("log", f"Transcription complete. {len(segments)} segments (Lang: {detected_lang}).")
            self._save_stage_json("transcription", "segments.json", {
                "segments": segments, "detected_lang": detected_lang})
            self._finish_stage("transcription", {"segments": len(segments),
                                                 "detected_lang": detected_lang})
        
        # Update source lang if auto
        if source_code == "auto" or source_code is None:
            logger.info(f"Updating source language from 'auto' to '{detected_lang}'")
            source_code = detected_lang
            if target_code == "auto": target_code = "en"
            
        # Split transcription segments by speaker if diarization is enabled
        if enable_diarization and diarization_segments:
             logger.info("Splitting transcription segments by speaker changes...")
             split_segments = []
             for seg in segments:
                 split_segments.extend(self._split_segment_by_speaker(seg, diarization_segments))
             segments = split_segments
             
        # Merge short segments
        logger.info(f"Merging short segments (min_dur={config.MERGE_MIN_DURATION}s, max_gap={config.MERGE_MAX_GAP}s)...")
        before_count = len(segments)
        segments = self.transcriber.merge_short_segments(segments, min_duration=config.MERGE_MIN_DURATION, max_gap=config.MERGE_MAX_GAP)
        logger.info(f"Merged segments: {before_count} -> {len(segments)}")

        # ---------------------------------------------------------------------
        # 4b. Per-speaker reference selection
        #     Runs HERE, not during diarization, because it needs the transcript:
        #     a cloning reference is an (audio, text) pair, and the text can only
        #     come from ASR. Built once for the whole video and reused by every
        #     segment, so a two-hour movie extracts each voice once.
        # ---------------------------------------------------------------------
        if enable_diarization and diarization_segments:
            yield ("progress", 0.38, "Selecting speaker references...")
            yield ("log", "Selecting a clean reference clip per speaker...")
            # Resume: references are an (audio, text) pair per speaker, derived
            # from the transcript. Persisting them means a resume does not
            # re-extract every voice, and it is also what lets a later change be
            # scoped to one speaker.
            cached_refs = self._load_stage_json("references", "references.json")
            reusable = None
            if cached_refs and self._stage_done("references"):
                reusable = {}
                for spk, rec in (cached_refs.get("speakers") or {}).items():
                    audio = rec.get("reference_audio")
                    if audio and joblib.validate_wav(Path(audio)):
                        reusable[spk] = rec
                if not reusable:
                    reusable = None
            if reusable is not None:
                for spk, rec in reusable.items():
                    speaker_profiles[spk] = rec["reference_audio"]
                    speaker_ref_audio[spk] = rec["reference_audio"]
                    if rec.get("reference_text"):
                        speaker_ref_texts[spk] = rec["reference_text"]
                self.session.speaker_references = reusable
                yield ("log", f"Reusing speaker references from a previous run "
                              f"({len(reusable)} speaker(s)).")
            else:
                self._begin_stage("references")
                try:
                    refs = speaker_reference.build_speaker_references(
                        segments=segments,
                        diar_segments=diarization_segments,
                        vocals_path=vocals_path,
                        output_dir=self._cached_path(
                            "speaker_refs", "profiles",
                            self._scratch_path(f"{video_path.stem}_profiles")),
                    )
                    for spk, ref in refs.items():
                        if ref.has_audio:
                            # Override the diarization-time profile, which was cut
                            # without knowing what was actually said in it.
                            speaker_profiles[spk] = ref.reference_audio
                            speaker_ref_audio[spk] = ref.reference_audio
                            if ref.has_text:
                                speaker_ref_texts[spk] = ref.reference_text
                    self.session.speaker_references = {
                        spk: ref.to_dict() for spk, ref in refs.items()
                    }
                    logger.info("Speaker references:\n%s", speaker_reference.summarize_references(refs))
                    yield ("log", f"Speaker references ready for {len(refs)} speaker(s).")
                    self._save_stage_json("references", "references.json", {
                        "speakers": {spk: ref.to_dict() for spk, ref in refs.items()}})
                    self._finish_stage("references", {"speakers": len(refs)})
                except Exception as exc:
                    # Never let reference selection sink the run: the existing
                    # validation + fallback chain still applies per segment.
                    self._fail_stage("references", exc)
                    logger.error("Speaker reference selection failed: %s", exc, exc_info=True)
                    yield ("log", f"Speaker reference selection failed ({exc}); using existing profiles.")

        # ---------------------------------------------------------------------
        # 5. Translation Stage
        #    - Translates text segments to target language
        #    - Supports Context-Aware LLM translation
        # ---------------------------------------------------------------------
        config.debug_log("Stage 5: Translation")
        self.load_model("translation_llm") 
        yield ("progress", 0.4, "Translating...")
        
        # Resume: translation is the most expensive text stage and its output
        # is a plain segment list. The per-line cache inside the Translator is
        # keyed on text+lang+model, so a partial translation resumes line-wise.
        cached_translation = self._load_segments("translation")
        if cached_translation and self._stage_done("translation"):
            translated_segments = cached_translation
            yield ("log", f"Reusing translation from a previous run ({len(translated_segments)} segments).")
        else:
            self._begin_stage("translation")
            try:
                translated_segments = self._step_translation(
                    segments, source_code, target_code, target_lang,
                    translation_model_name, context_model_name, optimize_translation,
                    video_path,
                    hf_token=hf_token
                )
            except Exception as exc:
                self._fail_stage("translation", exc)
                raise
            # Success is "usable text came out", not "the file was written".
            # A backend that fails every line returns a full-length list of
            # empties; recording that as complete would dub nothing and still
            # report success. Fail here, at the responsible stage.
            translation_error = self._translation_invariant_error(
                segments, translated_segments)
            if translation_error:
                self._fail_stage("translation", translation_error, {
                    "segments": len(translated_segments),
                    "source_speech_segments": self._usable_text_count(segments, "text"),
                    "non_empty_translations": self._usable_text_count(
                        translated_segments, "translated_text"),
                })
                raise RuntimeError(translation_error)
            yield ("log", f"Translation complete for {target_lang}.")
            self._save_segments("translation", translated_segments)
            if self.cache:
                self.cache.flush_translations()
            self._finish_stage("translation", {"segments": len(translated_segments)})
        
        # 6. TTS
        config.debug_log("Stage 6: TTS")
        self.load_model("tts")
        # Note: Progress 0.5 is now yielded in _step_tts_generator to avoid duplicate
        # ---------------------------------------------------------------------
        # 5b. Duration-aware translation (Phase 5, layer 1)
        #     Rewrite only the lines predicted to badly overrun their window.
        #     Runs before TTS because text is the cheapest thing to change.
        # ---------------------------------------------------------------------
        translated_segments = self._step_duration_aware_text(
            translated_segments, target_code, source_code=source_code,
            hf_token=hf_token,
        )

        # Authoritative timing records: built once, carried through TTS and the
        # merge so no later stage re-derives a segment's window.
        timings = build_timings(translated_segments)
        self.timing_diagnostics = TimingDiagnostics()

        # Ask the backend for an exact duration only if it can honour one
        # (IndicF5). Everyone else is corrected later by measurement.
        duration_model = DurationModel(source_lang=source_code, target_lang=target_code)
        # IndicF5's fix_duration is the TOTAL (reference + generated) length, so
        # the reference durations must be known before a request is made.
        ref_durations = measure_reference_durations(speaker_ref_audio)
        native_fit = plan_native_fit(timings, duration_model,
                                     model_name=tts_model_name,
                                     ref_durations=ref_durations)
        if native_fit:
            yield ("log", f"Requesting native duration fit for {len(native_fit)} segment(s).")

        
        # Determine iterator for progress reporting
        gen_iterator = self._step_tts_generator(
            translated_segments, video_path, target_code, source_code,
            tts_model_name, tts_voice, tts_enable_cfg,
            enable_diarization, diarization_segments, speaker_map, speaker_profiles, vocals_path,
            speaker_ref_texts=speaker_ref_texts,
            speaker_ref_audio=speaker_ref_audio,
            timings=timings, native_fit=native_fit,
        )
        
        tts_segments = []
        for item in gen_iterator:
            if isinstance(item, tuple) and item[0] == "progress":
                 yield item
            elif isinstance(item, list):
                 tts_segments = item

        # Second line of defence: translated dialogue that produced no speech
        # must fail here rather than reach the mix as a silent dub. A legitimately
        # speechless source has nothing to synthesize and passes.
        tts_error = self._tts_invariant_error(translated_segments, tts_segments)
        if tts_error:
            self._fail_stage("tts", tts_error, {
                "required": self._usable_text_count(translated_segments, "translated_text"),
                "generated": len(tts_segments or []),
            })
            raise RuntimeError(tts_error)
        
        # EQ Matching
        if enable_audio_enhancement and vocals_path and Path(vocals_path).exists():
             yield ("log", "Applying EQ matching...")
             count = self._apply_eq_matching_batch(vocals_path, tts_segments)
             yield ("log", f"Applied EQ matching to {count} segments.")

        yield ("log", "TTS Generation complete.")

        # 7. Sync & Mix
        config.debug_log("Stage 7: Synchronizing")
        self.unload_all_models() 
        yield ("progress", 0.7, "Synchronizing...")
        
        # Calculate duration
        try:
             duration_sec = sf.info(str(extracted_path)).duration
        except:
             duration_sec = tts_segments[-1]['end'] + 2.0 if tts_segments else 10.0
             
        self._begin_stage("sync")
        try:
            merged_speech, final_mix = self._step_merge_mix(
                tts_segments, duration_sec, video_path, bg_path,
                enable_time_stretch, enable_audio_enhancement
            )
        except Exception as exc:
            self._fail_stage("sync", exc)
            raise
        # Persist the timeline (per-segment audio paths and windows) so a resume
        # that only needs to re-mix does not re-run TTS.
        self._save_segments("sync", tts_segments)
        self._finish_stage("sync", {"segments": len(tts_segments),
                                    "duration": float(duration_sec)})
        # Timing report: which segments needed stretching, which overflowed,
        # which failed. Kept on the session for the UI/editor and for auditing
        # a finished render.
        self.session.timing_diagnostics = self.timing_diagnostics.summary()
        logger.info("Timing summary: %s", self.session.timing_diagnostics)
        degraded = (self.timing_diagnostics.overflowed + self.timing_diagnostics.trimmed
                    + self.timing_diagnostics.failed)
        if degraded:
            yield ("log", f"Timing: {len(degraded)} segment(s) could not fit their window cleanly.")
        
        # 7b. Subtitles (SRT + WebVTT)
        # Runs here, after sync, for two reasons. The cue timings come from the
        # *dubbed* timeline (which sync has just built and persisted), so a
        # subtitle matches what is heard rather than what was said; and the
        # artefacts are durable under the job, so they survive a restart and are
        # offered through the API instead of only landing in the shared
        # `output/` folder where two jobs would overwrite each other.
        mode = self._subtitle_mode(subtitle_mode)
        subtitle_paths: dict = {}
        if mode == "none":
            # The user asked for no subtitles. The stage is recorded SKIPPED so
            # the manifest says the choice was honoured rather than that the
            # stage was forgotten, and so a later resume does not silently add
            # files the request excluded.
            self._begin_stage("subtitles")
            self._skip_stage("subtitles", "subtitle_mode=none")
            yield ("log", "Subtitle mode is 'none'; no subtitle files will be written.")
        else:
            yield ("progress", 0.75, "Writing subtitles...")
            self._begin_stage("subtitles")
            try:
                subtitle_paths = self._step_subtitles(
                    tts_segments, video_path, duration=duration_sec,
                )
            except Exception as exc:
                # Subtitles are a deliverable, not a dependency of the video, so
                # a failure is recorded but does not abort a render whose audio
                # and video are already correct.
                self._fail_stage("subtitles", exc)
                yield ("log", f"Subtitle export failed: {exc}")
            else:
                detail = {k: str(v) for k, v in subtitle_paths.items()}
                if subtitle_paths:
                    self._finish_stage("subtitles", detail)
                    yield ("log", f"Subtitles written: {', '.join(p.name for p in subtitle_paths.values())}")
                else:
                    # No translated text at all: nothing to write. Marking the
                    # stage SKIPPED (not COMPLETE) keeps the manifest honest —
                    # the stage ran and deliberately produced nothing.
                    self._skip_stage("subtitles", "no translated text to subtitle")
                    yield ("log", "No translated text; skipping subtitles.")
        # The muxed/burned track needs the SRT the stage just wrote. Kept on the
        # instance so the final stage can find it without re-deriving the path.
        self._pending_subtitles = subtitle_paths

        # The final audio deliverables are independent of subtitles, so they are
        # persisted here rather than inside the subtitle stage: with
        # `subtitle_mode=none` that stage deliberately writes nothing, and
        # hanging the audio mix off it would silently drop the required
        # `audio_mixed` artifact for every no-subtitle render.
        self._persist_mix_artifacts(merged_speech, bg_path, final_mix, video_path)

        # 8. Visual & LipSync
        if enable_visual_translation:
             config.debug_log("Stage 8: Visual Translation")
             yield ("progress", 0.8, "Translating Video Text...")
             self.load_model("visual")
             visual_out = self._scratch_path(f"{video_path.stem}_visual.mp4")
             try:
                 self.visual_translator.translate_video_text(
                     str(video_path), str(visual_out),
                     target_lang=target_code, source_lang=source_code,
                     ocr_engine=ocr_model_name, ocr_interval_sec=2.0
                 )
                 if visual_out.exists():
                     video_path = visual_out
                     yield ("log", "Visual translation complete.")
             except CancelledError:
                 raise
             except Exception as e:
                logger.error(f"Visual translation failed: {e}")
                config.debug_log(f"Visual translation failed: {e}")
                
        if enable_lipsync and lipsync_model_name:
            config.debug_log("Stage 9: Lipsync")
            yield ("progress", 0.9, f"Lip-Syncing ({lipsync_model_name})...")

            # A finished lip-sync is expensive and its result feeds no later
            # audio stage, so it is checkpointed: a resume after a crash in the
            # final mux reuses it instead of paying for the model again. The
            # cache is read before `_begin_stage` so the completed record is
            # still inspectable.
            cached = self._valid_artifact("lipsync", "lipsynced.mp4", kind="video")
            reuse = bool(cached) and self._stage_done("lipsync")
            self._begin_stage("lipsync")

            if reuse:
                video_path = Path(cached)
                self._finish_stage("lipsync", {"output": str(video_path), "cached": True})
                yield ("log", f"Lip-Sync reused from checkpoint: {video_path.name}")
            else:
                self.load_model("lipsync")

                # Propagate configuration to an already-built lipsyncer.
                if hasattr(self.lipsyncer, 'acceleration') and live_portrait_mode:
                     if self.lipsyncer.acceleration != live_portrait_mode:
                         logger.info(f"Switching LivePortrait acceleration from {self.lipsyncer.acceleration} to {live_portrait_mode}")
                         self.lipsyncer.acceleration = live_portrait_mode

                out_path = self._step_lipsync(video_path, merged_speech, lipsync_model_name)
                out_path = Path(out_path) if out_path else None

                if out_path and out_path.exists() and out_path.stat().st_size > 0:
                    # Keep the result in the job so a resume does not pay for
                    # the model again; the temp copy is a staging area.
                    durable = self._stage_output("lipsync", "lipsynced.mp4")
                    if durable is not None:
                        import shutil as _sh
                        try:
                            _sh.copy2(str(out_path), str(durable))
                            out_path = durable
                        except OSError as exc:
                            logger.warning("Could not keep a durable lip-sync copy: %s", exc)
                    video_path = out_path
                    self._finish_stage("lipsync", {"output": str(out_path)})
                    yield ("log", "Lip-Sync complete.")
                else:
                    # Lip-sync is optional: no usable output means this region
                    # keeps its original video, and the dubbed-audio pipeline is
                    # unaffected. The movie still finishes.
                    self._fail_stage("lipsync", "no usable lip-synced video produced")
                    yield ("log", "Lip-Sync produced no usable video; keeping the "
                                  "original video and continuing with dubbed audio.")
        
        # 9. Final Output
        config.debug_log("Stage 10: Final Merge")
        output_video = config.OUTPUT_DIR / f"translated_{video_path.name}"
        # Sticky: the mux takes whatever lip-sync produced as its input, and
        # whether that ran is not in the fingerprinted config. Re-muxing is
        # seconds; trusting a stale mux would hand back the wrong video.
        self._begin_stage("final", sticky=True)
        # With a job, the finished video is written into the job's `final/` dir
        # as well as the output folder, so a resume that only needs the final
        # mux does not have to redo the mix.
        try:
            result = self.processor.replace_audio(str(video_path), str(final_mix), str(output_video))
        except Exception as exc:
            self._fail_stage("final", exc)
            raise
        # A mux that produced nothing must fail the stage. Recording it would
        # mark the manifest COMPLETE and hand a missing/empty file to the user
        # and to `resume_job`. `replace_audio` raises on a real ffmpeg failure
        # and only ever returns a path on success, so a falsy result here means
        # a backend that signalled failure with None.
        if not result:
            exc = RuntimeError(f"Final mux produced no output video at {output_video}")
            self._fail_stage("final", exc)
            raise exc
        # Embed or burn the subtitle track, if the request asked for it. Done on
        # the finished mux so the track is timed against the dubbed audio, and
        # best-effort: a subtitle track is a bonus, so a failure leaves the
        # correct video in place rather than failing a render that already
        # produced the right picture and sound.
        result = self._apply_subtitle_track(result, mode)

        # The durable copy is taken *after* the subtitle pass, because it is the
        # file the user is handed: copying the pre-subtitle mux and recording
        # that as `final_output` would deliver a video without the track that
        # was just added. The name follows the result, so the plain mux and the
        # subtitled variant stay distinguishable in the job directory.
        durable = None
        if self.job:
            import shutil as _shutil
            durable = self.job.stage_dir("final") / Path(str(result)).name
            try:
                if Path(str(result)).resolve() != durable.resolve():
                    _shutil.copy2(str(result), str(durable))
            except OSError as exc:
                logger.warning("Could not keep a durable copy of the final video: %s", exc)
                durable = None

        # Defense-in-depth, before anything is recorded as finished: a mux on
        # disk proves ffmpeg ran, not that the job dubbed anything. If the
        # source had speech and no speech came out, fail instead of reporting a
        # false success. The earlier stages normally catch this first.
        final_error = self._final_invariant_error(segments, tts_segments)
        if final_error:
            self._fail_stage("final", final_error)
            raise RuntimeError(final_error)

        if self.manifest:
            self.manifest.status = "COMPLETE"
            self.manifest.diagnostics["timing"] = self.session.timing_diagnostics
            # Recorded as an artifact, not only in the stage detail, so a later
            # resume of an already-finished job can hand the video straight back
            # without re-running anything. Only a copy that actually landed is
            # recorded: pointing at a failed copy would make `resume_job` hand
            # back a path that is not there.
            if durable and Path(str(durable)).exists():
                self.manifest.artifacts["final_output"] = str(durable)
            else:
                self.manifest.artifacts["final_output"] = str(result)
            self._finish_stage("final", {"output": str(result)})
            self.manifest.save()
            if self.job:
                self.job.log("job complete")

        yield ("result", str(result))

    # --- Durable job helpers (Phase 6) ---

    @staticmethod
    def _run_config(local_vars: dict) -> dict:
        """
        The pipeline configuration that decides cached output.

        Recorded in the manifest so a resume can tell whether the run it is
        continuing is the same run. Keys are sorted by the writer.
        """
        keys = (
            "source_lang", "target_lang", "audio_model_name", "tts_model_name",
            "translation_model_name", "context_model_name",
            "transcription_model_name", "optimize_translation",
            "enable_diarization", "diarization_model", "min_speakers",
            "max_speakers", "enable_time_stretch", "enable_vad",
            "vad_min_silence_duration_ms", "enable_lipsync",
            "lipsync_model_name", "live_portrait_mode",
            "enable_visual_translation", "ocr_model_name", "tts_voice",
            "transcription_beam_size", "tts_enable_cfg",
            "enable_audio_enhancement", "chunk_duration", "subtitle_mode",
        )
        cfg = {}
        for key in keys:
            value = local_vars.get(key)
            # hf_token and progress objects are not configuration and must not
            # be persisted; only plain scalars survive.
            if isinstance(value, (str, int, float, bool)) or value is None:
                cfg[key] = value
        return cfg

    def _emit_job_progress(self):
        """Yield the durable progress block, rebuilt from the manifest."""
        if not self.manifest:
            return
        yield ("log", self.manifest.render_progress_text())

    def _stage_fingerprint(self, stage: str, **extra) -> str:
        """
        Fingerprint of a stage's inputs.

        Built from the manifest's source plus the run-configuration keys that
        this stage and its upstream stages depend on (see `STAGE_CONFIG_KEYS`),
        so a change to one stage's settings invalidates that stage and the ones
        after it but leaves earlier cached work intact. An unknown stage falls
        back to the whole config, which is conservative: it cannot under-
        invalidate.

        `extra` lets a caller add what the config does not carry (a chunk index,
        a per-segment identity).
        """
        stage_keys = config.STAGE_CONFIG_KEYS.get(stage)
        recorded = (self.manifest.config if self.manifest else {}) or {}
        if stage_keys is None:
            # An unknown stage cannot under-invalidate: it hashes the whole
            # recorded config, the pre-Phase-7 behaviour.
            config_subset = dict(recorded)
        else:
            config_subset = {k: recorded[k] for k in config.config_keys_through(stage)
                             if k in recorded}
        base = {
            "stage": stage,
            "source": self.manifest.source if self.manifest else {},
            "config": config_subset,
        }
        # A chunk is a different input for the same stage name, so it must
        # produce a different fingerprint.
        if self._chunk_index is not None:
            base["chunk"] = self._chunk_index
        base.update(extra)
        return fingerprint(base)

    def _stage_done(self, stage: str, **extra) -> bool:
        """Whether `stage` already completed for inputs matching `extra`."""
        if not self.manifest:
            return False
        return self.manifest.stage_is_valid(stage, self._stage_fingerprint(stage, **extra))

    def _chunk_done(self, index: int, total: int) -> bool:
        """
        Whether a chunk's durable result is already recorded as complete.

        `total` is accepted for symmetry and future use (a partially finished
        movie where every chunk but the last is done); only the recorded state
        matters today.
        """
        if not self.manifest:
            return False
        rec = self.manifest.chunks.get(index)
        return bool(rec and rec.status == STAGE_COMPLETE)

    def _run_stage_with_retry(self, fn, label: str, attempts=None):
        """
        Run one expensive stage operation, retrying transient failures.

        The biggest single-point-of-failure risk in a long render is a stage
        that fails once for a transient reason: a model that would not load, a
        subprocess that lost its pipe, a lock on an output file. Retrying those
        is cheap; re-running the whole job is not. Fatal failures are not
        retried, and cancellation is re-raised untouched.
        """
        outcome = run_with_retry(fn, attempts=attempts, label=label)
        if not outcome.ok:
            raise RuntimeError(f"{label} failed: {outcome.error}") from None
        return outcome.value

    def _begin_stage(self, stage: str, sticky: bool = False, **extra):
        """
        Mark a stage running, returning its fingerprint (or "" without a job).

        `sticky=True` always produces a fresh fingerprint, so the stage re-runs
        even when nothing in its configuration changed. It is for stages whose
        result is not captured by config alone — the final mux, which depends on
        whatever the lip-sync stage produced without that being in the
        fingerprinted config.
        """
        if not self.manifest:
            return ""
        fp = self._stage_fingerprint(stage, **extra)
        if sticky:
            fp = fingerprint({"sticky": fp, "nonce": time.time_ns()})
        self.manifest.start_stage(stage, fp)
        if self.job:
            self.job.log(f"stage {stage} start")
        return fp

    def _finish_stage(self, stage: str, detail: dict | None = None):
        if not self.manifest:
            return
        self.manifest.complete_stage(stage, detail)
        if self.job:
            self.job.log(f"stage {stage} complete {detail or ''}")

    @staticmethod
    def _subtitle_mode(mode) -> str:
        """
        The subtitle mode, normalised, or the default for an unknown value.

        An unknown mode falls back to `external` rather than raising: a client
        that sends a typo should get subtitles, not a failed render. `none` is
        only returned when it was asked for by name, so a typo can never
        silently produce a job with no subtitles.
        """
        value = str(mode or "").strip().lower()
        if value in config.SUBTITLE_MODES:
            return value
        return config.DEFAULT_SUBTITLE_MODE

    def _skip_stage(self, stage: str, reason: str = ""):
        """Record a stage that ran and deliberately produced nothing."""
        if not self.manifest:
            return
        self.manifest.skip_stage(stage, reason)
        if self.job:
            self.job.log(f"stage {stage} skipped {reason}")

    def _fail_stage(self, stage: str, error, detail: dict | None = None):
        if not self.manifest:
            return
        self.manifest.fail_stage(stage, str(error), detail)

    def _stage_output(self, stage: str, name: str) -> "Path | None":
        """
        Path to a stage's durable artifact, or None when there is no job.

        With a job, artifacts live under `jobs/<id>/<stage>/` and survive a
        restart. Without one, the legacy flat `temp/` layout is used, so callers
        that opted out of durability behave exactly as before.
        """
        if not self.job:
            return None
        stage_dir = self.job.stage_dir(stage)
        if self._chunk_index is not None:
            # Per-chunk subdirectory, so one chunk's intermediate never
            # overwrites another's.
            stage_dir = stage_dir / f"chunk_{self._chunk_index:03d}"
            stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir / joblib.safe_name(name, max_length=120)

    def _cached_path(self, stage: str, name: str, legacy: "Path") -> "Path":
        """Prefer the durable location when a job exists, else the legacy one."""
        durable = self._stage_output(stage, name)
        return durable if durable is not None else Path(legacy)

    def _scratch_path(self, name: str) -> "Path":
        """
        A working file that is neither durable nor shared.

        With a job this lives in the job's own `scratch/`, so two jobs never
        collide and a run can never overwrite a committed fixture under the
        repository's shared `temp/`. Without a job (the legacy, non-durable
        path) it falls back to `temp/` exactly as before, preserving the old
        behaviour for callers that opted out of durability.
        """
        if self.job is not None:
            return self.job.scratch_dir() / joblib.safe_name(name, max_length=120)
        return config.TEMP_DIR / name

    def _save_stage_json(self, stage: str, name: str, payload) -> None:
        """Persist a stage's structured output so a resume can reload it."""
        target = self._stage_output(stage, name)
        if target is None:
            return
        try:
            joblib.atomic_write_json(target, payload)
        except Exception as exc:  # noqa: BLE001 - never fail a render on a save
            logger.warning("Could not persist %s/%s: %s", stage, name, exc)

    def _load_stage_json(self, stage: str, name: str):
        """Reload a stage's structured output, or None if absent/unreadable."""
        target = self._stage_output(stage, name)
        if target is None:
            return None
        return joblib.read_json(target)

    def _save_segments(self, stage: str, segments) -> None:
        """
        Persist a segment list durably.

        Timings are floats, texts are strings and speaker ids are strings, so the
        list survives a round-trip through JSON without loss.
        """
        if not segments:
            return
        plain = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            row = {k: v for k, v in seg.items()
                   if isinstance(v, (str, int, float, bool)) or v is None}
            plain.append(row)
        self._save_stage_json(stage, "segments.json", plain)

    def _load_segments(self, stage: str):
        data = self._load_stage_json(stage, "segments.json")
        return data if isinstance(data, list) else None

    def _valid_artifact(self, stage: str, name: str, kind: str = "wav",
                        min_duration: float = 0.05) -> "Path | None":
        """
        The stage's artifact, if it exists and validates.

        Validation is the difference between "a file is there" and "a file is
        usable": a crash mid-write leaves a file, and treating that as a cache
        hit is how a dub ends up with a silent stretch in the middle.
        """
        target = self._stage_output(stage, name)
        if target is None or not target.exists():
            return None
        if kind == "wav" and not joblib.validate_wav(target, min_duration=min_duration):
            logger.warning("Discarding unusable %s artifact: %s", stage, target)
            return None
        if kind == "video" and not joblib.validate_video(target):
            logger.warning("Discarding unusable %s artifact: %s", stage, target)
            return None
        return target

    # --- Orchestration Steps ---

    def _step_extraction(self, video_path: Path) -> str:
        """
        Extracts full audio track from the video file using ffmpeg.
        
        :param video_path: Path to the input video.
        :return: Path to the extracted .wav audio file.
        :raises Exception: If extraction fails.
        """
        full_audio = self._scratch_path(f"{video_path.stem}_full.wav")
        extracted_path = self.processor.extract_audio(str(video_path), str(full_audio))
        if not extracted_path:
             raise Exception("Failed to extract audio")
        return extracted_path

    def _step_separation(self, extracted_path, model_name):
        """
        Separates audio into vocals and background (accompaniment).
        
        :param extracted_path: Path to the full audio file.
        :param model_name: Name/Type of the separation model (e.g., 'demucs', 'mdx').
        :return: Tuple (vocals_path, accompaniment_path).
        """
        return self.separator.separate(
            extracted_path, model_selection=model_name,
            output_dir=self._scratch_path("separation"))

    def _step_diarization(self, vocals_path, video_path, model_name, min_spk, max_spk, hf_token=None):
        """
        Performs speaker diarization to identify speakers and extract their profiles.
        
        :param vocals_path: Path to the isolated vocals audio.
        :param video_path: Path to the original video (used for naming).
        :param model_name: Name of the diarization model/backend.
        :param min_spk: Minimum number of speakers.
        :param max_spk: Maximum number of speakers.
        :return: Tuple (segments, speaker_map, speaker_profiles).
        """
        # Map UI name to backend key
        diar_backend = "speechbrain"
        if "NeMo" in model_name: diar_backend = "nemo"
        elif "Community" in model_name: diar_backend = "pyannote_community"
        elif "pyannote" in model_name: diar_backend = "pyannote"
            
        segs = self.diarizer.diarize(vocals_path, backend=diar_backend, min_speakers=min_spk, max_speakers=max_spk, hf_token=hf_token)
        spk_map = self.diarizer.detect_genders(vocals_path, segs)
        
        profiles_dir = self._scratch_path(f"{video_path.stem}_profiles")
        profiles = self.diarizer.extract_speaker_profiles(vocals_path, segs, profiles_dir)
        return segs, spk_map, profiles

    def _step_transcription(self, vocals_path, source_code, model_name, use_vad, beam_size, min_silence):
        """
        Transcribes the vocals track into text segments with timestamps.
        
        :param vocals_path: Path to isolated vocals.
        :param source_code: Language code of the source audio.
        :param model_name: Size of the Whisper model (e.g., 'tiny', 'large-v3').
        :return: List of transcription segments (dict with 'start', 'end', 'text').
        """
        return self.transcriber.transcribe(
            str(vocals_path), 
            language=source_code,
            beam_size=beam_size,
            use_vad=use_vad,
            model_size=model_name,
            min_silence_duration_ms=min_silence
        )

    @staticmethod
    def _resolve_translation_backend(model_name):
        """
        Map a UI model label to a backend key.

        Returns None for the auto option so the router decides from the language
        pair. Unknown labels keep the historical Google default.
        """
        if not model_name:
            return None
        if "Auto" in model_name: return None
        if "NLLB" in model_name: return "nllb"
        if "HY-MT" in model_name: return "hymt"
        if "Llama" in model_name: return "llama"
        if "ALMA" in model_name: return "alma"
        return "google"

    @staticmethod
    def _usable_text_count(segments, key: str) -> int:
        """Number of segments whose `key` text is non-empty after stripping."""
        count = 0
        for seg in segments or ():
            if not isinstance(seg, dict):
                continue
            if str(seg.get(key) or "").strip():
                count += 1
        return count

    @classmethod
    def _translation_invariant_error(cls, source_segments, translated_segments):
        """
        Translation's success condition, as an actionable message or None.

        A backend that fails every line (an outage, a rate limit, an unsupported
        pair) still returns a full-length list whose translations are all empty.
        Without this check that list is written to disk and the stage is marked
        complete, so the render goes on to dub nothing and still reports success.
        Zero usable translations from real speech is always fatal; individual
        failures among otherwise usable lines are not.
        """
        speech = cls._usable_text_count(source_segments, "text")
        if speech == 0:
            return None
        if cls._usable_text_count(translated_segments, "translated_text") == 0:
            return (
                f"Translation failed: {speech} speech segment(s) were provided "
                "but no usable translated text was produced. The translation "
                "backend returned no usable translations; check network/backend "
                "availability or select another translation backend.")
        return None

    @classmethod
    def _tts_invariant_error(cls, translated_segments, tts_segments):
        """Synthesis's success condition: dialogue in, speech out."""
        required = cls._usable_text_count(translated_segments, "translated_text")
        if required == 0:
            return None
        if not tts_segments:
            return (
                f"Speech synthesis failed: {required} translated segment(s) "
                "required synthesis but no usable speech was generated.")
        return None

    @classmethod
    def _final_invariant_error(cls, source_segments, tts_segments):
        """
        Last line of defence before a job is recorded COMPLETE.

        A finished mux on disk proves only that ffmpeg ran; it is not proof that
        anything was dubbed. If the source carried speech but no speech was
        synthesized, the output is a silent dub and must not be marked complete.
        """
        speech = cls._usable_text_count(source_segments, "text")
        if speech == 0:
            return None
        if not tts_segments:
            return (
                f"Refusing to mark the job complete: the source contains {speech} "
                "speech segment(s) but no dubbed speech was produced.")
        return None

    def _step_translation(self, segments, source_code, target_code, target_lang, 
                          model_name, context_model, optimize, video_path, hf_token=None):
        """
        Translates transcribed segments to the target language.
        Handles caching, optimization (context-aware), and SRT export.
        
        :param segments: List of source segments.
        :param source_code: Source language code.
        :param target_code: Target language code.
        :param model_name: Translation model key (e.g., 'google', 'gpt4').
        :param optimize: Whether to use context-aware translation.
        :return: List of segments with added 'translated_text' key.
        """
        # [Early Exit] Optimization
        if source_code == target_code:
            logger.info(f"Source matches Target ({source_code}). Skipping translation API.")
            translated = []
            for seg in segments:
                new_seg = seg.copy()
                new_seg["translated_text"] = seg["text"]
                translated.append(new_seg)
        else:
            effective_model = context_model if (optimize and context_model) else model_name
            trans_key = self._resolve_translation_backend(effective_model)
            # Pass cancel callback to translator. `trans_key=None` lets the router
            # pick a backend from the language pair (NLLB for Indic pairs).
            translated = self.translator.translate_segments(
                segments, target_code, model=trans_key,
                source_lang=source_code, optimize=optimize,
                check_cancel_callback=self._check_cancel,
                hf_token=hf_token
            )
            plan = getattr(self.translator, "last_plan", None)
            if plan is not None:
                logger.info("Translation route: %s via %s", plan.path, plan.backend)
            failures = getattr(self.translator, "last_failures", None)
            if failures:
                logger.warning(
                    "%d segment(s) could not be translated and were left empty.",
                    len(failures))


        # Subtitles are written by the dedicated `subtitles` stage, which runs
        # after sync so a cue can be re-timed to the dubbed audio rather than the
        # source. Writing them here as well would produce two files that
        # disagree, and would put a file-system side effect inside a translation
        # method that is otherwise pure.
        return translated

    def _step_duration_aware_text(self, translated_segments, target_code,
                                  source_code=None, hf_token=None):
        """
        Phase 5 layer 1: shorten only the lines predicted to badly overrun.

        Text is the cheapest correction available, so it is attempted before any
        audio is generated. NLLB cannot be instructed to be concise, so this is
        a separate rewrite pass, not a prompt. Anything the validator rejects is
        discarded and the original translation is kept, so the worst case is
        "no improvement" rather than wrong dialogue.
        """
        if not translated_segments:
            return translated_segments

        shortener = SpeechShortener()
        changed = 0
        for seg in translated_segments:
            text = seg.get("translated_text") or ""
            start = float(seg.get("start") or 0.0)
            end = float(seg.get("end") or start)
            window = end - start
            source_text = seg.get("source_text") or seg.get("text") or ""

            # Duration-aware shortening is an optimisation. If the language
            # registry cannot rate this pair, or anything else goes wrong, the
            # translation is used as-is rather than failing the render.
            try:
                if not should_shorten(text, target_code, window,
                                      source_text=source_text,
                                      source_lang=source_code):
                    continue
                new_text, method = shortener.shorten(
                    text, target_code, window, token=hf_token,
                    source_text=source_text, source_lang=source_code)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Duration-aware shortening skipped for segment: %s", exc)
                continue
            if method == "none" or new_text == text:
                continue

            # Keep the original available: the dub used the short version, but
            # subtitles and any later re-edit want the full one.
            meta = seg.setdefault("translation", {})
            meta["original_translated_text"] = text
            meta["shortened_by"] = method
            seg["translated_text"] = new_text
            changed += 1

        if changed:
            logger.info("Duration-aware shortening rewrote %d/%d segment(s).",
                        changed, len(translated_segments))
        return translated_segments

    def _step_tts_generator(self, translated_segments, video_path, target_code, source_code,
                            model_name, tts_voice, enable_cfg, enable_diarization, 
                            diarization_segments, speaker_map, speaker_profiles, vocals_path,
                            speaker_ref_texts=None, speaker_ref_audio=None,
                            timings=None, native_fit=None):
        """
        Generator that handles the loop for TTS segments.
        Now uses BATCH generation for performance.
        
        :param translated_segments: List of text segments to synthesize.
        :param video_path: Source video path.
        :param target_code: Target language code.
        :param source_code: Source language code.
        :param model_name: TTS model to use.
        :param tts_voice: Preferred generic voice.
        :param enable_cfg: Enable Classifier-Free Guidance.
        :param enable_diarization: Whether to use speaker profiles.
        :param diarization_segments: Diarization data.
        :param speaker_map: Map of speaker IDs to genders.
        :param speaker_profiles: Map of speaker IDs to reference wavs.
        :param vocals_path: Original vocals for fallback.
        :yields: Progress updates or log messages.
        :return: List of generated TTS segments (yielded as final item).
        """
        # With a job, segment audio lives in the job workspace so it survives a
        # restart; without one, the legacy flat temp layout is used unchanged.
        seg_dir = self._cached_path("tts", "segments", config.TEMP_DIR / video_path.stem)
        seg_dir.mkdir(parents=True, exist_ok=True)
        tts_segments = []
        timing_by_id = {t.segment_id: t for t in (timings or [])}
        native_fit = native_fit or {}

        # Segments already synthesized by an earlier run, keyed by segment
        # index. Filled in below, per segment, from the *resolved* reference:
        # the reference a segment ends up using depends on the fallback chain,
        # so it cannot be predicted before that chain has run.
        cached_entries: dict = {}
        
        # Route the backend by TARGET LANGUAGE. A language with a dedicated
        # reference-cloning engine (IndicF5 for the 11 Indian languages) uses
        # it; everything else keeps the requested general backend. An explicit
        # request for a backend that cannot speak the language falls back with
        # a warning instead of failing mid-render.
        routed_model = self.tts_engine.route_model(target_code, model_name)
        if routed_model != model_name:
            yield ("log", f"TTS backend for {target_code}: {model_name} -> {routed_model}")
            model_name = routed_model

        # Prepare Batch Tasks
        tasks = []
        original_indices = []
        
        total_segs = len(translated_segments)
        
        for i, seg in enumerate(translated_segments):
             if i % 5 == 0: self._check_cancel()
             text = seg["translated_text"]
             if not text: continue
             
             ext = ".wav" if model_name in ["piper", "xtts"] else ".mp3"
             # Index alone is not an identity: re-rendering segment 3 with
             # different text must not land on the previous run's file.
             seg_out = seg_dir / f"seg_{i}_{self._seg_digest(seg, i, model_name)}{ext}"
             
             gender, best_speaker, speaker_wav = "Female", None, None
             use_force_cloning = False
             
             # Resolve Speaker
             if enable_diarization:
                 best_speaker = seg.get('speaker_id')
                 if not best_speaker and diarization_segments:
                     best_speaker = self._find_overlapping_speaker(seg, diarization_segments)
                 if best_speaker:
                     gender = speaker_map.get(best_speaker, "Female")
                     speaker_wav, use_force_cloning = self._resolve_speaker_reference(
                         best_speaker, speaker_profiles, vocals_path, model_name
                     )
                 else:
                     # logger.warning(f"No specific speaker detected for segment {i}.")
                     pass
             else:
                  # If no diarization, generic or fallback
                  speaker_wav = vocals_path 
                  
             # Fallback Logic
             speaker_wav, use_force_cloning = self._apply_reference_fallback(
                 speaker_wav, vocals_path, model_name, i,
                 speaker_id=best_speaker, enable_diarization=enable_diarization
             )

             # Native duration request (Phase 5 layer 2). Only a backend that
             # can honour an exact duration gets one; the rest ignore the key
             # and are corrected later by measurement + a bounded stretch.
             seg_timing = timing_by_id.get(seg.get('segment_id', i))
             request = None
             if seg_timing is not None and native_fit:
                 request = native_fit.get(seg_timing.segment_id)
             if request is not None and seg_timing is not None:
                 seg_timing.duration_honoured = honours_duration(model_name)

             # Create Task
             task = {
                'text': text,
                'output_path': seg_out,
                'fix_duration': request,
                'language': target_code,
                'speaker_wav': speaker_wav,
                # Transcript of the reference clip. Reference-conditioned
                # backends that condition on it (IndicF5) require it; the
                # others ignore it. Resolved per speaker from Stage 4b, and
                # withheld when the fallback chain substituted different audio.
                'ref_text': self._resolve_speaker_ref_text(
                    best_speaker, seg, speaker_ref_texts,
                    resolved_wav=speaker_wav, speaker_ref_audio=speaker_ref_audio,
                ),
                'gender': gender,
                'speaker_id': best_speaker if enable_diarization else None,
                'guidance_scale': 1.3 if enable_cfg else None,
                'force_cloning': use_force_cloning,
                'voice_selector': self._get_assigned_voice,
                'source_lang': source_code,
                'preferred_voice': tts_voice
             }
             # Content identity of this line's audio. If a previous run
             # produced audio for exactly this text, voice, model and duration
             # request, reuse it and do not re-synthesize.
             key = tts_key(
                 text=text, language=target_code, model=model_name,
                 reference_text=task.get('ref_text'),
                 fix_duration=task.get('fix_duration'),
                 settings={'cfg': bool(enable_cfg)},
                 reference_identity=reference_identity(speaker_wav),
             )
             task['cache_key'] = key
             cached_audio = self.cache.get_tts(key) if self.cache else None
             if cached_audio is not None:
                 seg_meta = translated_segments[i]
                 timing = timing_by_id.get(seg_meta.get('segment_id', i))
                 entry = {
                     'audio_path': str(cached_audio),
                     'start': seg_meta['start'],
                     'end': seg_meta['end'],
                     'segment_id': seg_meta.get('segment_id', i),
                     'speaker_id': seg_meta.get('speaker_id'),
                 }
                 if timing is not None:
                     entry['timing'] = timing
                 cached_entries[i] = entry
                 if self.manifest:
                     self.manifest.upsert_segment(
                         i, start=seg_meta.get('start', 0.0), end=seg_meta.get('end', 0.0),
                         speaker_id=seg_meta.get('speaker_id'),
                         source_text=seg_meta.get('text', ''),
                         translated_text=text, status=STAGE_COMPLETE,
                         audio_path=str(cached_audio), cache_key=key)
                 continue

             tasks.append(task)
             original_indices.append(i) # Needed to map back to segments

        if cached_entries:
            yield ("log", f"Reusing {len(cached_entries)} cached TTS segment(s) "
                          f"from a previous run.")
        yield ("log", f"Batch generating {len(tasks)} TTS segment(s)...")
        
        # Execute Batch
        yield ("progress", 0.5, "Generating Speech (Batch)...")
        
        self._check_cancel()
        # Retry the whole batch once on a transient failure. Per-segment
        # failures inside the engine already fall back individually; this
        # catches the case where the batch call itself dies (a model that
        # failed to load, a lost subprocess).
        generated_paths = self._run_stage_with_retry(
            lambda: self.tts_engine.generate_batch(
                tasks, model=model_name, check_cancel_callback=self._check_cancel),
            "tts batch")
        
        # Start the result list from the cached entries so ordering by segment
        # index is preserved regardless of which lines were reused.
        produced: dict = dict(cached_entries)
        failures: list = []
        msg_count = 0
        for idx, (orig_i, path) in enumerate(zip(original_indices, generated_paths)):
             seg = translated_segments[orig_i]
             timing = timing_by_id.get(seg.get('segment_id', orig_i))
             task = tasks[idx] if idx < len(tasks) else {}
             key = task.get('cache_key', '')

             if path and Path(path).exists() and Path(path).stat().st_size > 100:
                audio_path = str(path)
                # Cache the artifact so a later resume reuses it. Only a valid
                # file is cached; caching a truncated one would make the
                # failure permanent.
                if self.cache and key:
                    stored = self.cache.put_tts(key, path)
                    if stored is not None:
                        audio_path = str(stored)
                entry = {
                    'audio_path': audio_path,
                    'start': seg['start'],
                    'end': seg['end'],
                    # Identity travels with the audio so the merge stage can
                    # report per speaker and per segment.
                    'segment_id': seg.get('segment_id', orig_i),
                    'speaker_id': seg.get('speaker_id'),
                }
                if timing is not None:
                    entry['timing'] = timing
                produced[orig_i] = entry
                if self.manifest:
                    self.manifest.upsert_segment(
                        orig_i, start=seg.get('start', 0.0), end=seg.get('end', 0.0),
                        speaker_id=seg.get('speaker_id'),
                        source_text=seg.get('text', ''),
                        translated_text=seg.get('translated_text', ''),
                        status=STAGE_COMPLETE, audio_path=audio_path, cache_key=key)
             else:
                # One segment failing must not sink the run: record it, keep the
                # rest, and let the merge lay silence in its place.
                error = "no usable audio produced"
                failures.append(orig_i)
                logger.warning("TTS segment %d produced no usable audio.", orig_i)
                if self.manifest:
                    self.manifest.upsert_segment(
                        orig_i, start=seg.get('start', 0.0), end=seg.get('end', 0.0),
                        speaker_id=seg.get('speaker_id'),
                        source_text=seg.get('text', ''),
                        translated_text=seg.get('translated_text', ''),
                        status=STAGE_FAILED, error=error)
                    self.manifest.record_error("tts", error, segment_index=orig_i)

             # Checkpoint periodically: a crash from here on loses at most the
             # segments after this point, not the whole stage.
             msg_count += 1
             if self.manifest and msg_count % 50 == 0:
                 self.manifest.save()

        if failures and len(failures) == len(original_indices) and not cached_entries:
            raise RuntimeError(
                f"Every one of {len(failures)} TTS segments failed to synthesize. "
                "Stopping rather than producing a silent dub.")

        tts_segments = [produced[i] for i in sorted(produced)]
        if failures:
            yield ("log", f"TTS: {len(failures)} segment(s) produced no audio; "
                          f"silence will be laid in their place.")

        if self.manifest:
            counts = self.manifest.segment_counts()
            if failures:
                self.manifest.partial_stage("tts", {"failed": len(failures), **counts})
            else:
                self.manifest.complete_stage("tts", counts)
            self.manifest.save()

        yield ("progress", 0.65, "TTS Generation Complete.")

        # Return the final list
        yield tts_segments

    @staticmethod
    def _seg_digest(seg: dict, index: int, model_name: str) -> str:
        """
        A short, stable suffix that makes a segment's output file name unique.

        Derived from the text and speaker, so re-rendering the same line reuses
        the same file (and the same cache entry) while a changed line cannot
        collide with a stale one.
        """
        return fingerprint({
            "index": index,
            "text": seg.get('translated_text', ''),
            "speaker": seg.get('speaker_id'),
            "model": model_name,
        })

    @staticmethod
    def _resolve_speaker_ref_text(speaker_id, seg, speaker_ref_texts,
                                  resolved_wav=None, speaker_ref_audio=None):
        """
        Resolve the reference transcript for a segment.

        Preference order:
          1. the segment's own ref_text, when a caller attached one
          2. the per-speaker reference transcript, but ONLY if the audio that
             will actually be sent is that same speaker's selected reference
          3. None, which reference-conditioned backends treat as "absent"

        Step 2's guard matters. `_apply_reference_fallback` may substitute a
        last-valid or 0-30s clip when a speaker's own reference fails
        validation. Pairing that substituted audio with the original speaker's
        transcript would tell the model the clip says something it does not,
        which corrupts its duration model and produces rushed, robotic speech.
        A missing transcript degrades voice quality; a wrong one degrades the
        timing as well, so we drop it instead.
        """
        if seg.get('ref_text'):
            return seg['ref_text']
        if not speaker_id or not speaker_ref_texts:
            return None

        text = speaker_ref_texts.get(speaker_id)
        if not text:
            return None

        if resolved_wav is not None and speaker_ref_audio:
            expected = speaker_ref_audio.get(speaker_id)
            if expected and Path(resolved_wav) != Path(expected):
                logger.debug(
                    "Speaker %s is using substituted reference audio (%s != %s); "
                    "dropping its ref_text to avoid an audio/transcript mismatch.",
                    speaker_id, Path(resolved_wav).name, Path(expected).name,
                )
                return None
        return text

    def _find_overlapping_speaker(self, seg, diar_segments):
         """
         Identifies the speaker with the most overlap for a given segment.
         
         :param seg: The text segment {'start': float, 'end': float}.
         :param diar_segments: List of diarization segments with speaker info.
         :return: The speaker ID (e.g., 'SPEAKER_01') or None.
         """
         seg_start, seg_end = seg['start'], seg['end']
         max_overlap = 0
         best = None
         for d_seg in diar_segments:
              ov_start = max(seg_start, d_seg['start'])
              ov_end = min(seg_end, d_seg['end'])
              overlap = max(0, ov_end - ov_start)
              if overlap > max_overlap:
                  max_overlap = overlap
                  best = d_seg['speaker']
         return best

    def _split_segment_by_speaker(self, seg: dict, diar_segments: list) -> list[dict]:
        """
        Splits a single transcription segment into multiple sub-segments 
        whenever the speaker shifts, using word-level timestamps.
        """
        if not seg.get('words') or not diar_segments:
             # Fallback if no word-level timestamps: assign speaker to whole segment
             best_spk = self._find_overlapping_speaker(seg, diar_segments)
             new_seg = seg.copy()
             new_seg['speaker_id'] = best_spk
             return [new_seg]

        # Assign speaker to each word
        for w in seg['words']:
             w_start, w_end = w['start'], w['end']
             best_spk = None
             max_overlap = 0.0
             for d_seg in diar_segments:
                  ov_start = max(w_start, d_seg['start'])
                  ov_end = min(w_end, d_seg['end'])
                  overlap = max(0.0, ov_end - ov_start)
                  if overlap > max_overlap:
                       max_overlap = overlap
                       best_spk = d_seg['speaker']
             
             if not best_spk:
                  # Fallback: assign speaker of nearest diarization segment
                  min_dist = float('inf')
                  for d_seg in diar_segments:
                       dist = max(0.0, d_seg['start'] - w_end) if d_seg['start'] >= w_end else max(0.0, w_start - d_seg['end'])
                       if dist < min_dist:
                            min_dist = dist
                            best_spk = d_seg['speaker']
                            
             w['speaker_id'] = best_spk or 'SPEAKER_00'

        # Group words into continuous speaker sub-segments
        sub_segments = []
        current_words = []
        current_speaker = None
        
        for w in seg['words']:
             spk = w['speaker_id']
             if current_speaker is None:
                  current_speaker = spk
                  current_words.append(w)
             elif spk == current_speaker:
                  current_words.append(w)
             else:
                  # Create a sub-segment
                  sub_text = "".join(x['word'] for x in current_words).strip()
                  sub_segments.append({
                      'start': current_words[0]['start'],
                      'end': current_words[-1]['end'],
                      'text': sub_text,
                      'words': current_words,
                      'speaker_id': current_speaker
                  })
                  current_speaker = spk
                  current_words = [w]
                  
        if current_words:
             sub_text = "".join(x['word'] for x in current_words).strip()
             sub_segments.append({
                 'start': current_words[0]['start'],
                 'end': current_words[-1]['end'],
                 'text': sub_text,
                 'words': current_words,
                 'speaker_id': current_speaker
             })
             
        return sub_segments

    def _apply_reference_fallback(self, speaker_wav, vocals_path, model_name, index, speaker_id=None, enable_diarization=False):
         """
         Applies cascading fallback logic for speaker reference audio.
         1. Validates current reference.
         2. Tries 'Last Valid Reference'.
         3. Tries '0-30s Clip' from original vocals.
         
         :param speaker_wav: Current candidate for speaker reference.
         :param vocals_path: Path to full vocals (source for 30s clip).
         :param model_name: TTS model name (for validation rules).
         :param index: Segment index for logging.
         :param speaker_id: The speaker ID for tracking speaker-specific voice clones.
         :param enable_diarization: Whether multi-speaker diarization is enabled.
         :return: Tuple (resolved_wav_path, force_cloning_flag).
         """
         # 1. Update tracking
         if speaker_wav:
             if self.tts_engine.validate_reference(speaker_wav, model_name=model_name):
                 self.session.last_valid_reference_wav = speaker_wav
                 if enable_diarization and speaker_id:
                     if not hasattr(self.session, 'speaker_last_valid_reference'):
                         self.session.speaker_last_valid_reference = {}
                     self.session.speaker_last_valid_reference[speaker_id] = speaker_wav
             else:
                 speaker_wav = None
         
         use_force = False
         
         # 2. Last Valid Fallback
         if not speaker_wav:
             if enable_diarization and speaker_id:
                 # Check speaker-specific last valid reference first
                 if hasattr(self.session, 'speaker_last_valid_reference'):
                     speaker_wav = self.session.speaker_last_valid_reference.get(speaker_id)
                     if speaker_wav:
                         logger.info(f"Segment {index}: Using last valid reference for speaker {speaker_id}.")
             else:
                 # Fallback to global last valid reference (for single speaker or unidentified speaker)
                 if self.session.last_valid_reference_wav:
                     logger.info(f"Segment {index}: Using LAST VALID reference.")
                     speaker_wav = self.session.last_valid_reference_wav
         
         # 3. 0-30s Fallback
         if not speaker_wav:
             fallback = self._extract_fallback_reference(vocals_path)
             if fallback:
                  logger.info(f"Segment {index}: Using 0-30s FALLBACK.")
                  speaker_wav = fallback
                  self.session.last_valid_reference_wav = fallback
                  if enable_diarization and speaker_id:
                      if not hasattr(self.session, 'speaker_last_valid_reference'):
                          self.session.speaker_last_valid_reference = {}
                      self.session.speaker_last_valid_reference[speaker_id] = fallback
         
         return speaker_wav, use_force

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _apply_eq_matching_batch(self, vocals_path, tts_segments):
         """
         Applies EQ matching to generated TTS segments to match the tone of the original vocals.
         
         :param vocals_path: Path to original vocals (reference).
         :param tts_segments: List of TTS segment dicts (modified in-place).
         :return: Count of successfully modified segments.
         """
         from src.audio.eq_matching import apply_eq_matching
         count = 0
         for tts_seg in tts_segments:
             out_path = str(tts_seg['audio_path']).replace('.wav', '_eq.wav').replace('.mp3', '_eq.wav')
             apply_eq_matching(str(vocals_path), str(tts_seg['audio_path']), out_path, strength=0.7)
             if Path(out_path).exists():
                 tts_seg['audio_path'] = out_path
                 count += 1
         return count

    def _subtitle_segments(self, tts_segments):
        """
        The cue list for the subtitle files, taken from the dubbed timeline.

        `tts_segments` (the synced timeline) carries the *final* window each line
        occupies — after stretching and trimming — but not its text, because the
        text lives with the translation. So the text is looked up from the
        durable translation by segment id, and the timing comes from the synced
        record. That is what makes a cue line up with the audio rather than with
        the original speech.

        A line whose text is missing is dropped rather than emitted empty: an
        empty cue is noise in a player and a puzzle in an editor.
        """
        translations = self._load_segments("translation") or []
        text_by_id: dict = {}
        for seg in translations:
            if not isinstance(seg, dict):
                continue
            key = seg.get("segment_id", seg.get("id"))
            text = (seg.get("translated_text") or "").strip()
            if key is not None and text:
                text_by_id[key] = text

        cues = []
        for index, seg in enumerate(tts_segments or []):
            if not isinstance(seg, dict):
                continue
            key = seg.get("segment_id", seg.get("id", index))
            text = text_by_id.get(key)
            if not text:
                # Fall back to the synced record's own text, if a backend put it
                # there; the translation lookup is the normal path.
                text = (seg.get("translated_text") or "").strip()
            if not text:
                continue
            cues.append({
                "start": float(seg.get("start") or 0.0),
                "end": float(seg.get("end") or 0.0),
                "translated_text": text,
                "speaker_id": seg.get("speaker_id"),
            })
        cues.sort(key=lambda c: (c["start"], c["end"]))
        return cues

    def _step_subtitles(self, tts_segments, video_path, duration=None):
        """
        Write the subtitle deliverables for a job and record them as artifacts.

        Returns a mapping of kind ("srt"/"vtt") to path, empty when there is
        nothing to write. The files go into the job's `subtitles/` directory so
        they are durable and per-job. Without a job the legacy flat `output/`
        directory is used instead, so the two layouts never disagree about the
        same render and two jobs never overwrite each other.

        The audio deliverables are deliberately *not* written here: with
        `subtitle_mode=none` this method is not called at all, and they must
        exist regardless of the subtitle choice.
        """
        cues = self._subtitle_segments(tts_segments)
        if not cues:
            return {}

        from src.utils.srt_generator import generate_srt, generate_vtt

        if self.job:
            out_dir = self.job.stage_dir("subtitles")
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = config.OUTPUT_DIR
            out_dir.mkdir(parents=True, exist_ok=True)

        stem = joblib.safe_name(Path(str(video_path)).stem, max_length=80)
        written: dict = {}
        for kind, suffix, writer in (
            ("srt", ".srt", generate_srt),
            ("vtt", ".vtt", generate_vtt),
        ):
            target = out_dir / f"{stem}{suffix}"
            writer(cues, str(target))
            written[kind] = target

        if self.manifest:
            # Recorded as artifacts so a resume hands them back without rewriting
            # them, and so the API can offer them by name.
            self.manifest.artifacts["subtitles_srt"] = str(written["srt"])
            self.manifest.artifacts["subtitles_vtt"] = str(written["vtt"])
            self.manifest.artifacts["subtitle_count"] = len(cues)

        return written

    def _apply_subtitle_track(self, result, mode):
        """
        Embed or burn the subtitle track into a finished mux.

        Returns the video to record as the job's output: the subtitle-bearing
        file when the operation succeeded, otherwise the mux unchanged. `mode`
        of `external` or `none` is a no-op, because the sidecar files already
        cover those cases.

        The new file is written into the job's `final/` directory (or beside the
        mux when there is no job) so it is durable and gets recorded in the
        manifest — the artifact list resolves `final_video` from that record,
        and pointing it at a temp file would hand back a path that disappears.
        """
        if mode not in ("embedded", "burned"):
            return result
        srt = (self._pending_subtitles or {}).get("srt")
        if not srt or not Path(str(srt)).exists():
            logger.warning("Subtitle mode %r requested but no SRT was written; "
                           "keeping the video without a subtitle track.", mode)
            return result

        target_dir = (self.job.stage_dir("final") if self.job
                      else Path(str(result)).parent)
        target_dir.mkdir(parents=True, exist_ok=True)
        # A distinct name: the plain mux stays on disk as the fallback, and the
        # two files must not be confused if the subtitle pass is re-run.
        target = target_dir / f"{Path(str(result)).stem}_subtitled{Path(str(result)).suffix}"
        try:
            self.processor.add_subtitles(str(result), str(srt), str(target),
                                         burn=(mode == "burned"))
        except Exception as exc:
            logger.error("Could not %s subtitles: %s", mode, exc)
            return result
        if not Path(str(target)).exists() or Path(str(target)).stat().st_size == 0:
            logger.error("Subtitle %s produced no output; keeping the plain mux.", mode)
            return result
        return target

    def _persist_mix_artifacts(self, dialogue, background, mixed, video_path):
        """
        Keep the mixed, dialogue-only and background tracks under the job.

        Best-effort and additive: a copy failure is logged, never raised. These
        are deliverables, but a render whose video is already correct must not
        fail because a copy of a secondary track did not land. Nothing is
        written without a job — the legacy flat layout has no per-job home.

        Each track is copied from the file the mix stage actually produced, so
        the artifact is the audible result itself and never a second mixdown.
        """
        if not self.job:
            return
        import shutil as _shutil

        final_dir = self.job.stage_dir("final")
        final_dir.mkdir(parents=True, exist_ok=True)
        stem = joblib.safe_name(Path(str(video_path)).stem, max_length=80)
        wanted = (
            ("audio_mixed", mixed, f"{stem}_audio_mixed.wav"),
            ("audio_dialogue", dialogue, f"{stem}_audio_dialogue.wav"),
            ("audio_background", background, f"{stem}_audio_background.wav"),
        )
        recorded = False
        for key, source, filename in wanted:
            if not source:
                continue
            source_path = Path(str(source))
            if not source_path.is_file():
                continue
            target = final_dir / filename
            try:
                _shutil.copy2(str(source_path), str(target))
            except OSError as exc:
                logger.warning("Could not keep %s: %s", key, exc)
                continue
            if self.manifest:
                self.manifest.artifacts[key] = str(target)
                recorded = True
        if recorded:
            # Written out now rather than at the end of the render: these are
            # requested deliverables, and a crash between here and the final mux
            # would otherwise lose the record while the files sit on disk.
            self.manifest.save()

    def _step_merge_mix(self, tts_segments, duration, video_path, bg_path, time_stretch, enhance_audio):
        """
        Merges individual TTS segments into a single track and mixes with background audio.
        
        :param tts_segments: List of generated TTS segments.
        :param duration: Total duration of the detailed audio.
        :param video_path: Original video path.
        :param bg_path: Background music/noise path.
        :param time_stretch: Whether to stretch audio to fit time.
        :param enhance_audio: Whether to run VoiceFixer.
        :return: Tuple (merged_speech_path, final_mix_path).
        """
        merged_speech = self._scratch_path(f"{video_path.stem}_merged_speech.wav")
        
        if not self.synchronizer.merge_segments(
                tts_segments, total_duration=duration, output_path=str(merged_speech),
                enable_time_stretch=time_stretch,
                diagnostics=self.timing_diagnostics):
            raise Exception("Merging speech segments failed")
            
        if enhance_audio:
             self.load_model("voice_enhancer")
             enhanced = self._scratch_path(f"{video_path.stem}_enhanced_speech.wav")
             try:
                 self.voice_enhancer.enhance_audio(merged_speech, enhanced)
                 if enhanced.exists(): merged_speech = enhanced
             except CancelledError:
                 raise
             except Exception as e:
                 logger.error(f"VoiceFixer failed: {e}")
                 
        final_mix = self._scratch_path(f"{video_path.stem}_final_mix.wav")
        self.processor.mix_tracks(str(merged_speech), bg_path, str(final_mix))
        return merged_speech, final_mix
        
    def _step_lipsync(self, video_path, audio_path, model_name):
         """
         Runs lip-syncing on the video using the generated audio.
         
         :param video_path: Path to the video file.
         :param audio_path: Path to the new audio file.
         :param model_name: Name of the lipsync model from UI.
         :return: Path to the lipsynced video or None if failed.
         """
         lipsync_out = self._scratch_path(f"{video_path.stem}_lipsync.mp4")
         try:
             # Resolve engine and enhancement
             engine_key = "wav2lip"
             enhance = "GFPGAN" in (model_name or "")
             
             if "LivePortrait" in (model_name or ""):
                 engine_key = "live_portrait"
             
             logger.info(f"Using lip-sync engine: {engine_key} (UI selected: {model_name})")
             
             self.lipsyncer.sync_lips(
                 str(video_path), 
                 str(audio_path), 
                 str(lipsync_out), 
                 model_name=engine_key,
                 enhance_face=enhance
             )
             return str(lipsync_out) if lipsync_out.exists() else None
         except CancelledError:
             raise
         except Exception as e:
             logger.error(f"Lip-Sync failed: {e}")
             return None