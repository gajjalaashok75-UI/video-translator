"""
Pydantic response models for the read-only API.

These are the public contract a TypeScript client will mirror, so field names
are stable and explicit, and status-like fields use enums rather than free-form
strings. Nothing here exposes an internal dictionary by accident: if a value
appears in a response, it has a name, a type and a documented meaning.

`Field(examples=[...])` values feed FastAPI's OpenAPI schema, which is how the
frontend team will see the shapes without reading Python.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ComponentState(str, Enum):
    """
    Availability of one optional runtime component.

    Deliberately richer than a boolean. "It doesn't work" is useless to a user
    who could fix it by installing a package, accepting a model licence, or
    downloading weights — and each of those is a different remedy, so each gets
    its own state.
    """

    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    MODEL_NOT_DOWNLOADED = "model_not_downloaded"
    TOKEN_REQUIRED = "token_required"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class ComponentStatus(BaseModel):
    """One component's availability plus a human-readable reason."""

    name: str = Field(description="Stable machine name, e.g. 'wav2lip'.")
    label: str = Field(description="Human-readable label for the UI.")
    state: ComponentState
    available: bool = Field(
        description="True only when the component can be used right now."
    )
    reason: str = Field(description="Why the component is in this state.")
    required_for: list[str] = Field(
        default_factory=list,
        description="Features that need this component, e.g. 'lip_sync'.",
    )
    detail: dict[str, str] = Field(
        default_factory=dict,
        description="Extra context (versions, expected paths). Never secrets.",
    )


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    api_version: str = Field(examples=["1.0.0"])
    uptime_seconds: float = Field(description="Seconds since the API process started.")


class SystemResponse(BaseModel):
    """
    Runtime environment. Every field is best-effort: a missing GPU, a missing
    FFmpeg binary or an unknown CUDA version yields null rather than an error.
    """

    api_version: str
    app_version: Optional[str] = Field(
        default=None, description="Version from pyproject.toml, when readable."
    )
    python_version: str = Field(examples=["3.12.14"])
    python_implementation: str = Field(examples=["CPython"])
    platform: str = Field(examples=["Linux"])
    platform_release: str
    machine: str = Field(examples=["x86_64"])
    processor: Optional[str] = None

    device: str = Field(description="Device the pipeline will use: 'cuda' or 'cpu'.")
    cuda_available: bool
    cuda_version: Optional[str] = Field(
        default=None, description="Torch's compiled CUDA version, when available."
    )
    cuda_runtime_version: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_count: int = 0
    gpu_memory_total_mb: Optional[int] = None
    torch_version: Optional[str] = None

    ffmpeg_available: bool
    ffmpeg_version: Optional[str] = None

    ffprobe_available: bool
    jobs_dir_writable: bool = Field(
        description="Whether the durable jobs directory can be created and written."
    )


class LanguageInfo(BaseModel):
    """
    One language, derived from `src/utils/languages.py` — never duplicated.

    The two cloning fields are deliberately separate. Whether a backend is
    *architecturally* capable of cloning a reference voice is a property of the
    language/engine pairing; whether the weights are actually present on this
    machine is a property of the installation. Conflating them would let the UI
    promise a cloned voice the machine cannot produce.
    """

    code: str = Field(examples=["hi"])
    name: str = Field(examples=["Hindi"])
    aliases: list[str] = Field(default_factory=list)
    script: Optional[str] = Field(
        default=None, description="Script from the flores-200 code, e.g. 'Deva'."
    )
    flores: Optional[str] = Field(default=None, examples=["hin_Deva"])

    source: bool = Field(description="Usable as a transcription source.")
    target: bool = Field(description="Usable as a dubbing target.")
    indic: bool = Field(description="Belongs to the Indic set the router treats as Indic.")
    priority_indic: bool = Field(
        description="One of the priority Indian target languages, shown first."
    )

    dedicated_engine: Optional[str] = Field(
        default=None,
        description="Reference-cloning engine that owns this language, if any.",
        examples=["indicf5"],
    )
    synth_lang: str = Field(
        description="Code handed to the TTS engine; differs from `code` for e.g. Urdu."
    )
    edge_supported: bool
    piper_supported: bool
    chars_per_second: float = Field(
        description="Rough natural speaking rate, used for duration sanity bounds."
    )

    cloning_supported_by_architecture: bool = Field(
        description="A backend that can clone this language's voice exists."
    )
    cloning_model_available_on_this_machine: bool = Field(
        description="That backend's weights are present and usable right now."
    )
    cloning_engine: Optional[str] = Field(
        default=None, description="Which cloning engine applies, if any."
    )


class LanguagesResponse(BaseModel):
    languages: list[LanguageInfo]
    sources: list[LanguageInfo] = Field(
        description="Languages usable as a source, including source-only entries."
    )
    targets: list[LanguageInfo] = Field(
        description="Languages usable as a target, priority Indic first."
    )
    auto_detect_code: str = Field(default="auto", examples=["auto"])


class BackendInfo(BaseModel):
    name: str
    label: str
    kind: str = Field(description="'translation' or 'tts'.")
    available: bool
    state: ComponentState
    reason: str
    detail: dict[str, str] = Field(default_factory=dict)


class PresetInfo(BaseModel):
    name: str = Field(examples=["BALANCED"])
    settings: dict[str, object] = Field(
        description="The quality configuration this preset applies."
    )


class RouteInfo(BaseModel):
    """A translation route the router can serve, for the language explorer."""

    source: str
    target: str
    backend: Optional[str] = Field(
        default=None, description="Preferred backend, when one is fixed for the pair."
    )
    pivots: list[str] = Field(
        default_factory=list, description="Intermediate languages if a pivot is needed."
    )
    is_passthrough: bool = False


class CapabilitiesResponse(BaseModel):
    translation_backends: list[BackendInfo]
    tts_backends: list[BackendInfo]
    cloning_engines: list[str] = Field(
        description="TTS backends that clone a reference voice by architecture."
    )
    quality_presets: list[PresetInfo]
    features: dict[str, bool] = Field(
        description="Optional pipeline features and whether they can run now."
    )
    sources: list[str] = Field(description="Language codes usable as a source.")
    targets: list[str] = Field(description="Language codes usable as a target.")
    priority_indic: list[str] = Field(
        description="Priority Indian targets, in display order."
    )
    routes: list[RouteInfo] = Field(
        description="Route matrix for the language explorer (bounded, see note)."
    )
    notes: list[str] = Field(
        default_factory=list, description="Caveats a client should surface."
    )


class OptionItem(BaseModel):
    value: str = Field(description="Value to send back to the API.")
    label: str = Field(description="Text to display.")
    description: Optional[str] = None
    available: bool = True
    state: ComponentState = ComponentState.AVAILABLE


class OptionsResponse(BaseModel):
    """
    Dropdown option lists.

    `source` records where each list came from, because during migration the
    Gradio literals in `app.py` are still the UI of record and these lists are
    reproduced here rather than shared.
    """

    translation_models: list[OptionItem]
    context_models: list[OptionItem]
    transcription_models: list[OptionItem]
    diarization_backends: list[OptionItem]
    lipsync_models: list[OptionItem]
    ocr_engines: list[OptionItem]
    audio_separators: list[OptionItem]
    live_portrait_modes: list[OptionItem]
    tts_models: list[OptionItem]
    subtitle_modes: list[OptionItem]
    source: dict[str, str] = Field(
        description="Per-list provenance, e.g. {'tts_models': 'languages.py'}."
    )


class VoicesResponse(BaseModel):
    tts_model: str
    language: str
    language_name: Optional[str] = None
    voices: list[str]
    cloning: bool = Field(
        description="True when this backend clones a reference voice instead of "
                    "selecting a named one."
    )
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# F6: bundled demo samples
# ---------------------------------------------------------------------------


class SampleInfo(BaseModel):
    """
    One allow-listed bundled sample.

    The API never lists a directory; a sample exists here only because it is in
    `src/api/samples.py`'s table. `available` is computed from the filesystem so
    a checkout that omitted the media reports it honestly instead of advertising
    a demo that would fail.
    """

    id: str = Field(examples=["hgf_scene"])
    title: str
    description: str
    filename: str = Field(description="Display name; resolved server-side.")
    duration_seconds: Optional[float] = None
    source_language: str = Field(examples=["en"])
    target_language: str = Field(examples=["hi"])
    credit: str = Field(description="Attribution for the bundled media.")
    size_bytes: Optional[int] = None
    media_type: Optional[str] = None
    available: bool = Field(
        description="True when the media is present and can be staged.")
    reason: Optional[str] = Field(
        default=None, description="Why the sample is unavailable, when it is.")


class SampleListResponse(BaseModel):
    """The bundled samples the demo page may offer."""

    samples: list[SampleInfo]
    total: int


# ---------------------------------------------------------------------------
# F2: uploads, jobs and execution
# ---------------------------------------------------------------------------


class JobState(str, Enum):
    """
    A job's lifecycle state as the API reports it.

    `queued` and `running` are *process-local*: they come from the runner that is
    executing the job. `completed`, `failed` and `cancelled` are also durable —
    the manifest records a terminal status — so a job that finished before a
    restart still reports its outcome, while one that was interrupted reports
    `interrupted` instead of pretending to still be running.
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


class RunMode(str, Enum):
    """Whether a runner is starting a job or continuing one."""

    NEW = "new"
    RESUME = "resume"


class MediaInfo(BaseModel):
    """
    What the server learned about an uploaded file.

    Informational: the authoritative check that a file is usable is the
    pipeline's own validation, which the upload endpoint runs. Fields are absent
    when ffprobe is unavailable rather than reported as zero.
    """

    container: Optional[str] = None
    duration: Optional[float] = Field(
        default=None, description="Duration in seconds.")
    width: Optional[int] = None
    height: Optional[int] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    has_video: Optional[bool] = None
    has_audio: Optional[bool] = None


class UploadResponse(BaseModel):
    """
    A staged upload.

    `upload_id` is the only handle the client needs; it is opaque and carries no
    client-supplied text, so it is safe to pass back. The original filename is
    returned for display only and has been sanitised.
    """

    upload_id: str = Field(examples=["up-1f2c3d4e5f60718293a4b5c6d7e8f901"])
    filename: str = Field(description="Sanitised display name of the upload.")
    bytes: int = Field(description="Size on disk in bytes.")
    media: MediaInfo = Field(default_factory=MediaInfo)
    status: str = Field(examples=["staged"])
    created_at: float
    claimed_by: Optional[str] = Field(
        default=None, description="Job id that claimed this upload, if any.")


class JobCreateRequest(BaseModel):
    """
    A request to render a staged upload.

    Every field is optional except `upload_id`, and each has the same default the
    Gradio UI uses, so a client that only wants "dub this into Hindi" can send
    two fields. Values are validated before a job directory is created: an
    unknown language or an impossible speaker range is a 422, not a job that
    fails a minute later inside a model call.
    """

    upload_id: str = Field(
        description="Id from POST /api/uploads, claimed by this job exactly once.")
    source_lang: str = Field(default="auto", examples=["auto", "en", "hi"])
    target_lang: str = Field(default="en", examples=["hi", "es", "en"])
    quality_preset: Optional[str] = Field(
        default=None, description="Named preset; overrides lip-sync settings.")

    audio_model_name: str = "Torchaudio HDemucs (Recommended)"
    tts_model_name: str = "edge"
    translation_model_name: str = "Google Translate (Online, Fast)"
    context_model_name: str = "Tencent HY-MT1.5 (Local, Better Context)"
    transcription_model_name: str = "Large v3 Turbo (Fast)"
    optimize_translation: bool = False

    enable_diarization: bool = False
    diarization_model: str = "pyannote/SpeechBrain (Default)"
    min_speakers: int = Field(default=1, ge=1, le=50)
    max_speakers: int = Field(default=10, ge=1, le=50)

    enable_time_stretch: bool = False
    enable_vad: bool = False
    vad_min_silence_duration_ms: int = Field(default=1000, ge=0, le=60000)

    enable_lipsync: bool = False
    lipsync_model_name: str = "Wav2Lip-GAN (Low Quality - Fast)"
    live_portrait_mode: str = "ort"

    enable_visual_translation: bool = False
    ocr_model_name: str = "PaddleOCR"

    tts_voice: Optional[str] = None
    transcription_beam_size: int = Field(default=7, ge=1, le=20)
    tts_enable_cfg: bool = False
    enable_audio_enhancement: bool = False
    chunk_duration: int = Field(default=300, ge=0, le=3600)
    subtitle_mode: str = Field(
        default="external",
        description=("How subtitles are delivered: 'none', 'external' (sidecar "
                     "files), 'embedded' (sidecar files plus a toggleable track "
                     "in the video), or 'burned' (rendered into the picture)."),
        examples=["external", "embedded", "burned", "none"],
    )


class StageStatus(BaseModel):
    """One pipeline stage's durable state, read from the manifest."""

    name: str
    status: str = Field(examples=["PENDING", "RUNNING", "COMPLETE", "FAILED"])
    error: Optional[str] = None
    attempts: int = 0


class SegmentCounts(BaseModel):
    """Per-segment progress, which is what the UI shows during TTS."""

    total: int = 0
    complete: int = 0
    pending: int = 0
    failed: int = 0
    partial: int = 0


class PreflightCheck(BaseModel):
    """
    One readiness check for a prospective render.

    `severity` is the whole point: `blocking` disables Start, `warning` does not,
    `ok` is a satisfied check. `field` names the request field at fault so the UI
    can point at the offending control rather than showing a bare message.
    """

    id: str = Field(description="Stable identifier, e.g. 'tts_language'.")
    label: str = Field(description="Short human label, e.g. 'TTS supports target'.")
    severity: str = Field(
        default="ok", examples=["ok", "warning", "blocking"],
        description="'ok', 'warning' or 'blocking'.")
    detail: str = Field(default="", description="Why the check has this severity.")
    field: Optional[str] = Field(
        default=None, description="The request field this check concerns, if any.")


class PreflightResponse(BaseModel):
    """
    The result of preflighting a render.

    `ready` is true when no check is `blocking`. It is advisory: the backend
    re-validates on `POST /api/jobs`, which is the authority. The panel exists so
    a user sees *why* Start is disabled, not to replace the check.
    """

    ready: bool
    checks: list[PreflightCheck] = Field(default_factory=list)
    blocking: list[PreflightCheck] = Field(default_factory=list)
    warnings: list[PreflightCheck] = Field(default_factory=list)


class JobSummary(BaseModel):
    """
    What a job did, from durable state.

    Every field is a recorded fact, so refreshing the result URL — or opening it
    on another machine — produces the same summary. Optional fields are None for
    a job that predates them rather than being guessed, and the UI renders an
    absent value as unknown.
    """

    source_language: Optional[str] = None
    source_language_name: Optional[str] = None
    target_language: Optional[str] = None
    target_language_name: Optional[str] = None
    translation_backend: Optional[str] = None
    context_backend: Optional[str] = None
    optimize_translation: bool = False
    voice_mode: str = Field(
        default="standard",
        description="'standard' or 'preserve_speaker' (per-speaker cloning).")
    tts_backend: Optional[str] = None
    tts_voice: Optional[str] = None
    subtitle_mode: str = "external"
    subtitle_count: int = Field(
        default=0, description="Cues written; 0 when none were requested.")
    lipsync: bool = False
    lipsync_model: Optional[str] = None
    diarization: bool = False
    speakers: int = Field(
        default=0, description="Speakers detected; 0 when diarization is off.")
    duration_seconds: Optional[float] = None
    status: str = Field(default="UNKNOWN", description="Raw manifest status.")
    state: str = Field(default="unknown", description="JobState as an enum value.")


class OutputInfo(BaseModel):
    """
    Where a job's finished video is, as URLs.

    Never a filesystem path: the client plays and downloads through the API, and
    an absolute path would disclose the server's layout. Both URLs are the same
    file route, which supports byte ranges: `media_url` is what a `<video>`
    element loads (media elements ignore `Content-Disposition`), and
    `download_url` is the same bytes offered as a named download. They are
    separate fields so a client does not have to know that.
    """

    available: bool = False
    filename: Optional[str] = None
    size_bytes: Optional[int] = None
    media_type: Optional[str] = Field(
        default=None, description="Content type the file is served with.")
    media_url: Optional[str] = Field(
        default=None, description="URL for inline playback (supports Range).")
    download_url: Optional[str] = Field(
        default=None, description="URL that prompts a download.")


class ResumeInfo(BaseModel):
    """
    Whether a job can be resumed, decided server-side.

    A client shows or hides a Resume control from this and does not re-derive the
    rules from `state`: the rules involve whether the source file still exists
    and whether the output is missing, which the client cannot see.
    """

    can_resume: bool
    reason: str = Field(
        description="Machine-readable token, e.g. 'interrupted', 'complete'.")
    message: str = Field(description="Sentence to show a person.")


class JobStatusResponse(BaseModel):
    """
    A job's current state, durable and live.

    Durable fields (stages, segments, status) come from the manifest on disk and
    are correct after a restart. Live fields (`live_progress`, `queue_position`)
    come from the runner in this process and are absent for a job this process is
    not running — which is not an error, just a different source of truth.
    """

    job_id: str
    state: JobState
    status: str = Field(description="Raw manifest status, e.g. RUNNING, COMPLETE.")
    run_mode: Optional[RunMode] = Field(
        default=None,
        description="Set while this process is running the job: 'new' or "
                    "'resume'. Null when no runner here owns it.")
    source: dict = Field(default_factory=dict)
    source_name: Optional[str] = Field(
        default=None, description="Original filename, for a job list to show.")
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    quality_preset: Optional[str] = None
    config: dict = Field(default_factory=dict)
    stages: list[StageStatus] = Field(default_factory=list)
    segments: SegmentCounts = Field(default_factory=SegmentCounts)
    progress_fraction: float = Field(
        default=0.0, description="Best available completion estimate, 0..1.")
    live_progress: Optional[float] = None
    live_description: Optional[str] = None
    queue_position: Optional[int] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    output_available: bool = False
    output: OutputInfo = Field(default_factory=OutputInfo)
    summary: JobSummary = Field(
        default_factory=JobSummary,
        description="Durable description of what this job did.")
    resume: ResumeInfo = Field(
        description="Whether this job can be resumed, and why.")
    error: Optional[str] = Field(
        default=None, description="Failure summary; never a traceback.")
    events_url: str = Field(
        description="Relative URL of this job's event stream.")
    logs_url: str = Field(
        default="", description="Relative URL of this job's durable log.")
    artifacts_url: str = Field(
        default="", description="Relative URL of this job's artifact list.")
    errors: list[dict] = Field(
        default_factory=list, description="Durable per-stage error records.")


class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]
    total: int
    active: int = Field(description="Jobs running in this process.")
    pending: int = Field(description="Jobs queued in this process.")


class CancelResponse(BaseModel):
    job_id: str
    state: JobState
    message: str


class ResumeResponse(BaseModel):
    """The outcome of asking a job to resume."""

    job_id: str
    state: JobState
    run_mode: RunMode = RunMode.RESUME
    message: str
    queue_position: Optional[int] = None


class ArtifactDescriptor(BaseModel):
    """
    One artifact the API is willing to hand out.

    `name` is the only handle a client uses; it resolves through a server-side
    allow-list, so no client string ever becomes a path. `available` is false for
    an artifact this render did not produce, with `reason` explaining why.
    """

    name: str = Field(examples=["subtitles"])
    type: str = Field(examples=["video", "subtitle", "transcript", "json"])
    label: str = Field(description="Human-readable name for a UI.")
    filename: Optional[str] = None
    available: bool
    size_bytes: Optional[int] = None
    media_type: Optional[str] = None
    download_url: Optional[str] = None
    description: str = ""
    reason: Optional[str] = Field(
        default=None, description="Why it is unavailable, when it is.")


class ArtifactListResponse(BaseModel):
    job_id: str
    artifacts: list[ArtifactDescriptor]
    count: int = Field(description="Number of artifacts listed.")
    available_count: int = Field(description="Number actually on disk.")


class LogsResponse(BaseModel):
    """
    A bounded window of a job's durable log.

    Plain text lines, because that is what the file is; a client renders them
    directly. They have been filtered of secrets and absolute paths, which
    `redacted` confirms, and `has_more` says whether earlier lines exist.
    """

    job_id: str
    lines: list[str]
    returned_lines: int
    total_lines: int
    has_more: bool = Field(
        description="True when the log has more lines than were returned.")
    truncated_lines: bool = Field(
        default=False, description="True when a line was shortened.")
    redacted: bool = True


class CleanupLevel(str, Enum):
    """
    How much of a deleted job's directory to remove.

    Mirrors the pipeline's own levels (`config.CLEANUP_*`) so the API does not
    invent a second vocabulary.
    """

    KEEP_ALL = "KEEP_ALL"
    KEEP_RESUMABLE = "KEEP_RESUMABLE"
    FINAL_ONLY = "FINAL_ONLY"
    EVERYTHING = "EVERYTHING"


class DeleteResponse(BaseModel):
    """What deleting a job actually did."""

    job_id: str
    deleted: bool
    level: CleanupLevel
    removed: list[str] = Field(
        default_factory=list,
        description="Subdirectories removed. Empty for KEEP_ALL.")
    message: str


class OutputResponse(BaseModel):
    """
    Where the finished video is.

    Deliberately not a filesystem path: the client is told whether an output
    exists and given the URL to download it, so the API never publishes an
    absolute server path. `media_url` is what a `<video>` element points at, and
    the download endpoint supports byte ranges so playback can seek.
    """

    job_id: str
    available: bool
    filename: Optional[str] = None
    download_url: Optional[str] = None
    media_url: Optional[str] = None
    media_type: Optional[str] = Field(
        default=None, description="Content type the file is served with.")
    size_bytes: Optional[int] = None
