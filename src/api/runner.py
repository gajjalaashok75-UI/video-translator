"""
One job's execution: exactly one `VideoTranslator`, one worker thread, one
event buffer.

The isolation rule for F2 is that a running job owns its pipeline instance
outright. The legacy Gradio UI builds a single module-level `VideoTranslator`
and reuses it for every render, which is fine when only one render can happen at
a time and wrong the moment two can: the instance carries the loaded model, the
session, the manifest and the cancel flag, so two jobs sharing one would cancel
each other and overwrite each other's durable state. Each `JobRunner` therefore
constructs its own instance, and nothing outside the runner ever holds a
reference to it.

The generator is consumed on a dedicated thread, never on the event loop.
Whisper, Demucs and TTS are blocking CPU/GPU work that would otherwise freeze
every other request — including `/api/health`, which must keep answering while a
render runs. The thread's only contact with asyncio is through
`EventBuffer.publish`, which marshals to the loop with `call_soon_threadsafe`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from src.api.events import EventBuffer, EventType
from src.core import jobs as joblib
from src.core.manifest import Manifest

logger = logging.getLogger(__name__)


def is_cancellation(exc: BaseException) -> bool:
    """
    Whether `exc` is the pipeline's cancellation signal.

    Checked by class name rather than `isinstance`, because importing
    `src.core.video_translator` at module scope would pull in torch and
    transformers before bootstrap has run — the exact thing F1 works to avoid.
    `src/core/retry.py` makes the same check the same way, so this is the
    established convention in this codebase rather than a new trick.
    """
    return type(exc).__name__ == "CancelledError"


@dataclass
class JobConfig:
    """
    The subset of `process_video` arguments the API accepts.

    Deliberately explicit rather than `**kwargs`: the contract is what the API
    promises, and a client must not be able to reach an argument that was never
    meant to be exposed. `job_id`, `resume` and `hf_token` are absent because
    the API supplies them itself — the first two from the job it just created,
    the last from the server environment.
    """

    source_lang: str = "auto"
    target_lang: str = "en"
    audio_model_name: str = "Torchaudio HDemucs (Recommended)"
    tts_model_name: str = "edge"
    translation_model_name: str = "Google Translate (Online, Fast)"
    context_model_name: str = "Tencent HY-MT1.5 (Local, Better Context)"
    transcription_model_name: str = "Large v3 Turbo (Fast)"
    optimize_translation: bool = False
    enable_diarization: bool = False
    diarization_model: str = "pyannote/SpeechBrain (Default)"
    min_speakers: int = 1
    max_speakers: int = 10
    enable_time_stretch: bool = False
    enable_vad: bool = False
    vad_min_silence_duration_ms: int = 1000
    enable_lipsync: bool = False
    lipsync_model_name: str = "Wav2Lip-GAN (Low Quality - Fast)"
    live_portrait_mode: str = "ort"
    enable_visual_translation: bool = False
    ocr_model_name: str = "PaddleOCR"
    tts_voice: Optional[str] = None
    transcription_beam_size: int = 7
    tts_enable_cfg: bool = False
    enable_audio_enhancement: bool = False
    chunk_duration: int = 300
    subtitle_mode: str = "external"
    quality_preset: Optional[str] = None

    def to_pipeline_kwargs(self) -> dict:
        """The keyword arguments for `VideoTranslator.process_video`."""
        return {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "audio_model_name": self.audio_model_name,
            "tts_model_name": self.tts_model_name,
            "translation_model_name": self.translation_model_name,
            "context_model_name": self.context_model_name,
            "transcription_model_name": self.transcription_model_name,
            "optimize_translation": self.optimize_translation,
            "enable_diarization": self.enable_diarization,
            "diarization_model": self.diarization_model,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "enable_time_stretch": self.enable_time_stretch,
            "enable_vad": self.enable_vad,
            "vad_min_silence_duration_ms": self.vad_min_silence_duration_ms,
            "enable_lipsync": self.enable_lipsync,
            "lipsync_model_name": self.lipsync_model_name,
            "live_portrait_mode": self.live_portrait_mode,
            "enable_visual_translation": self.enable_visual_translation,
            "ocr_model_name": self.ocr_model_name,
            "tts_voice": self.tts_voice,
            "transcription_beam_size": self.transcription_beam_size,
            "tts_enable_cfg": self.tts_enable_cfg,
            "enable_audio_enhancement": self.enable_audio_enhancement,
            "chunk_duration": self.chunk_duration,
            "subtitle_mode": self.subtitle_mode,
            "quality_preset": self.quality_preset,
        }


class RunMode(str, Enum):
    """
    Why a runner exists.

    An explicit mode rather than an inference from the filesystem: whether a job
    is new or resumed is a fact the caller knows and a directory cannot be
    trusted to reveal. A new job whose source happens to sit next to a stale
    manifest is still new; a resumed job whose output is missing is still a
    resume. The mode decides which pipeline entry point is called, so getting it
    wrong would mean running the wrong thing, not merely reporting the wrong
    thing.
    """

    NEW = "new"
    RESUME = "resume"


class JobRunner:
    """
    The process-local execution of one job.

    Lifecycle: `queued` -> `running` -> one of `completed`, `failed`,
    `cancelled`. A runner is created when the job is accepted (so the event
    buffer exists from the first `queued` event) but its worker thread starts
    only when the manager grants a slot.

    `mode` selects the pipeline entry point. `NEW` calls `process_video` with the
    configuration this process holds; `RESUME` calls `resume_job`, which replays
    the configuration recorded in the manifest and lets the stage fingerprints
    decide what is redone. A resume therefore needs no `JobConfig` at all — the
    manifest is the authority on how the job was configured, and reconstructing
    it in the API would risk disagreeing with what actually ran.
    """

    def __init__(self, job_id: str, source_path: Path,
                 config: Optional[JobConfig] = None,
                 loop=None, hf_token: Optional[str] = None,
                 on_finished: Optional[Callable[[str], None]] = None,
                 mode: RunMode | str = RunMode.NEW):
        self.job_id = job_id
        self.source_path = Path(source_path)
        self.config = config
        self.mode = RunMode(mode)
        if self.mode == RunMode.NEW and self.config is None:
            raise ValueError("a NEW runner requires a JobConfig")
        self.hf_token = hf_token
        self.events = EventBuffer(job_id, loop)
        # Called exactly once when the worker reaches a terminal state, so the
        # manager can free the slot and start the next queued job. Kept as a
        # callback rather than a manager reference to avoid a circular import and
        # to keep the runner independently testable.
        self._on_finished = on_finished
        self._finished_notified = False

        self.state = "queued"
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

        # Live progress from the generator, which is *not* durable state. It is
        # reset by a restart and never written to the manifest: the manifest is
        # the truth about completed stages, this is only what the bar shows.
        self.progress_fraction: float = 0.0
        self.progress_description: str = ""
        self.error: Optional[str] = None

        self._translator: Any = None
        self._thread: Optional[threading.Thread] = None
        self._cancel_requested = threading.Event()
        # Guards `state` and the pipeline instance against the cancel route
        # racing the worker's own transitions.
        self._lock = threading.Lock()

    # -- introspection -----------------------------------------------------
    @property
    def is_finished(self) -> bool:
        with self._lock:
            return self.state in ("completed", "failed", "cancelled")

    @property
    def translator(self) -> Any:
        """The pipeline instance, for tests and cancellation. Never shared."""
        return self._translator

    def status_dict(self) -> dict:
        """The runner's own live state, merged into the job response."""
        with self._lock:
            return {
                "state": self.state,
                "run_mode": self.mode.value,
                "live_progress": (round(self.progress_fraction, 4)
                                  if self.state == "running" else None),
                "live_description": (self.progress_description
                                     if self.state == "running" else None),
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }

    # -- starting ----------------------------------------------------------
    def start(self) -> None:
        """Begin execution on a worker thread. Called once by the manager."""
        with self._lock:
            if self.state != "queued":
                return
            self.state = "running"
            self.started_at = time.time()
        self.events.publish(EventType.STARTED, {
            "message": ("Resume started." if self.mode == RunMode.RESUME
                        else "Render started."),
            "run_mode": self.mode.value,
            "source": self.source_path.name,
        })
        self._thread = threading.Thread(
            target=self._run, name=f"job-{self.job_id}", daemon=True)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- worker ------------------------------------------------------------
    def _build_translator(self):
        """
        Construct this job's `VideoTranslator`.

        Imported here rather than at module scope so that importing the API (and
        therefore `/api/health`) never loads torch. `bootstrap()` has already run
        by the time a job exists, so the import order `app.py` requires is
        preserved.
        """
        from src.core.video_translator import VideoTranslator
        return VideoTranslator()

    def _run(self) -> None:
        """The worker body. Never raises: it records the outcome and returns."""
        if self._cancel_requested.is_set():
            # Cancelled before a pipeline was ever built; never start one.
            self._finish_cancelled(never_started=True)
            return

        try:
            self._translator = self._build_translator()
        except Exception as exc:  # noqa: BLE001 - report, do not crash the thread
            self._finish_failed(
                "The pipeline could not be initialised on this machine.",
                exc)
            return

        if self._cancel_requested.is_set():
            # Cancelled while the pipeline was being constructed. `abort()` was
            # already recorded, but no generator has run, so this is still a
            # never-started cancellation.
            self._finish_cancelled(never_started=True)
            return

        try:
            iterator = self._pipeline_iterator()
            self._consume(iterator)
        except BaseException as exc:  # noqa: BLE001 - the worker owns its failure
            if is_cancellation(exc):
                self._finish_cancelled()
            else:
                self._finish_failed("The render failed.", exc)

    def _pipeline_iterator(self):
        """
        The generator to consume, chosen by mode.

        A resume does *not* get the current process's configuration: it calls
        `resume_job`, which reads the config the manifest recorded when the job
        was created. That is what makes a resume reproduce the original render
        rather than a differently-configured one, and it is why `JobConfig` is
        not needed to resume.
        """
        if self.mode == RunMode.RESUME:
            return self._translator.resume_job(
                self.job_id, hf_token=self.hf_token)
        return self._translator.process_video(
            video_path=self.source_path,
            job_id=self.job_id,
            resume=True,
            hf_token=self.hf_token,
            **self.config.to_pipeline_kwargs(),
        )

    def _consume(self, iterator) -> None:
        """
        Drive the pipeline generator to completion.

        The generator is what the Gradio UI iterates, so this is the same
        contract, read the same way. Each `("progress", fraction, description)`
        tuple updates the live view; each `("log", text)` becomes a `log` event;
        `("result", path)` records the output.
        """
        output_path: Optional[str] = None
        for item in iterator:
            if not isinstance(item, (tuple, list)) or not item:
                continue
            kind = item[0]
            if kind == "log":
                self._on_log(str(item[1]) if len(item) > 1 else "")
            elif kind == "progress":
                fraction = float(item[1]) if len(item) > 1 else 0.0
                description = str(item[2]) if len(item) > 2 else ""
                self._on_progress(fraction, description)
            elif kind == "result":
                output_path = str(item[1]) if len(item) > 1 else None
        self._finish_completed(output_path)

    def _on_log(self, message: str) -> None:
        self.events.publish(EventType.LOG, {"message": message})

    def _on_progress(self, fraction: float, description: str) -> None:
        # Clamp: the generator is trusted, but a stray value must not produce a
        # nonsensical bar for the client.
        fraction = min(1.0, max(0.0, float(fraction)))
        self.progress_fraction = fraction
        self.progress_description = description
        self.events.publish(EventType.PROGRESS, {
            "fraction": round(fraction, 4),
            "percent": round(fraction * 100, 1),
            "description": description,
        })

    # -- terminal states ---------------------------------------------------
    def _notify_finished(self) -> None:
        """
        Tell the manager this job is done, exactly once.

        Wrapped so a bookkeeping failure can never turn a finished job into a
        crashed worker thread, and guarded so the slot is released only once even
        if a terminal transition is somehow reached twice.

        Called with `self._lock` released: the callback takes the manager's lock,
        and holding this one across that call would order the two locks
        inconsistently with the manager's own path and risk a deadlock.
        """
        with self._lock:
            if self._finished_notified:
                return
            self._finished_notified = True
        if self._on_finished is None:
            return
        try:
            self._on_finished(self.job_id)
        except Exception:  # noqa: BLE001
            logger.exception("Job %s: on_finished callback failed", self.job_id)

    def _finish_completed(self, output_path: Optional[str]) -> None:
        output_available = bool(output_path) and Path(str(output_path)).exists()
        with self._lock:
            self.state = "completed"
            self.finished_at = time.time()
            self.progress_fraction = 1.0
        self.events.publish(EventType.RESULT, {
            "message": "Render complete.",
            # A boolean, not a path: the client fetches the artifact through the
            # API, and an absolute server path is not something to publish.
            "output_available": output_available,
            "duration_seconds": (round(self.finished_at - (self.started_at or 0), 2)
                                 if self.started_at else None),
        })
        self._notify_finished()

    def _finish_failed(self, public_message: str, exc: BaseException) -> None:
        """
        Record a failure.

        The public payload is a short, sanitised sentence; the traceback stays in
        the server log. An exception message can contain a token, a filesystem
        path or a model URL, none of which belong in an API response.
        """
        logger.exception("Job %s failed", self.job_id)
        detail = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self.state = "failed"
            self.finished_at = time.time()
            self.error = detail
        self._record_manifest_error(detail)
        self.events.publish(EventType.FAILED, {
            "message": public_message,
            "error_type": type(exc).__name__,
            "retryable": False,
        })
        self._notify_finished()

    def _finish_cancelled(self, never_started: bool = False) -> None:
        with self._lock:
            self.state = "cancelled"
            self.finished_at = time.time()
        # A job cancelled before it ever ran never reached the pipeline, so the
        # pipeline's own `CANCELLED` write never happened. Recording it here is
        # what keeps the durable manifest from claiming `RUNNING` forever, which
        # a later status read would report as `interrupted` — a materially
        # different and wrong answer for a deliberate cancellation.
        self._record_manifest_terminal("CANCELLED")
        self.events.publish(EventType.CANCELLED, {
            "message": ("Cancelled before it started."
                        if never_started else "Cancelled by request."),
            "never_started": never_started,
        })
        self._notify_finished()

    def _record_manifest_terminal(self, status: str) -> None:
        """
        Set a terminal status on the durable manifest.

        The manifest is loaded fresh and written back, so the runner never holds
        a stale copy of the pipeline's own state. Best-effort: a failure to write
        is logged, not raised, because the job has already ended.

        Only called for outcomes the pipeline itself does not record. A completed
        or failed render has already written its own status through the manifest
        it owns, and rewriting it here would race that write for no gain.
        """
        try:
            job = joblib.Job.load(self.job_id)
            manifest = Manifest.load(job)
            if manifest is None:
                return
            manifest.status = status
            manifest.save()
        except Exception:  # noqa: BLE001 - best-effort durability
            logger.warning("Job %s: could not record %s in manifest",
                           self.job_id, status, exc_info=True)

    def _record_manifest_error(self, detail: str) -> None:
        """
        Preserve the failure in the durable manifest.

        The pipeline records a failed *stage* itself, but a crash outside a stage
        (model init, a bug in the driver) would otherwise leave the manifest
        claiming `RUNNING`. The manifest is loaded fresh and written back, so the
        runner never holds a stale copy of the pipeline's own state.
        """
        try:
            job = joblib.Job.load(self.job_id)
            manifest = Manifest.load(job)
            if manifest is None:
                return
            manifest.status = "FAILED"
            manifest.record_error("job", detail)
            manifest.save()
        except Exception:  # noqa: BLE001 - best-effort durability
            logger.warning("Job %s: could not record failure in manifest",
                           self.job_id, exc_info=True)

    # -- cancellation ------------------------------------------------------
    def request_cancel(self) -> str:
        """
        Ask this job to stop. Idempotent.

        A queued job is cancelled outright and will never build a pipeline. A
        running job has `abort()` called on *its own* translator; the flag is
        honoured at the pipeline's next cancellation checkpoint, which is
        between stages, not inside a model call — see the F2 notes in AGENTS.md.
        The request is also recorded before the instance exists, so a job that is
        just starting cannot slip past it.
        """
        self._cancel_requested.set()
        with self._lock:
            state = self.state
            translator = self._translator
        if state in ("completed", "failed", "cancelled"):
            return state
        if state == "queued":
            # No thread has started, so this is the terminal transition.
            self._finish_cancelled(never_started=True)
            return "cancelled"
        if translator is not None:
            try:
                translator.abort()
            except Exception:  # noqa: BLE001 - abort must never break the route
                logger.warning("Job %s: abort() raised", self.job_id,
                               exc_info=True)
        return "cancelling"
