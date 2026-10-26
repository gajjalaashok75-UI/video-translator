/**
 * TypeScript mirror of the FastAPI schemas in `src/api/schemas.py`.
 *
 * Field names, optionality and enum values are copied from the OpenAPI document
 * the backend emits, not invented. A field is optional here only when the
 * backend marks it nullable or defaulted, so a component cannot silently read a
 * value the server never sends.
 *
 * Kept as hand-written types rather than a generated client: the surface is
 * small, and a generator would add a build step whose output would still need
 * review. If the API grows, `openapi-typescript` can replace this file without
 * changing the callers.
 */

// ---------------------------------------------------------------------------
// Enums (string unions, matching the Pydantic str-enums)
// ---------------------------------------------------------------------------

export type ComponentState =
  | "available"
  | "not_installed"
  | "model_not_downloaded"
  | "token_required"
  | "unsupported_platform"
  | "unavailable"
  | "unknown";

export type JobState =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted"
  | "unknown";

export type RunMode = "new" | "resume";

/** Cleanup levels accepted by DELETE /api/jobs/{id}. */
export type CleanupLevel =
  | "KEEP_ALL"
  | "KEEP_RESUMABLE"
  | "FINAL_ONLY"
  | "EVERYTHING";

// ---------------------------------------------------------------------------
// System / discovery
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string;
  api_version: string;
  uptime_seconds: number;
}

export interface SystemResponse {
  api_version: string;
  app_version: string | null;
  python_version: string;
  python_implementation: string;
  platform: string;
  platform_release: string;
  machine: string;
  processor: string | null;
  device: string;
  cuda_available: boolean;
  cuda_version: string | null;
  cuda_runtime_version: string | null;
  gpu_name: string | null;
  gpu_count: number;
  gpu_memory_total_mb: number | null;
  torch_version: string | null;
  ffmpeg_available: boolean;
  ffmpeg_version: string | null;
  ffprobe_available: boolean;
  jobs_dir_writable: boolean;
}

export interface ComponentStatus {
  name: string;
  label: string;
  state: ComponentState;
  available: boolean;
  reason: string;
  required_for: string[];
  detail: Record<string, string>;
}

export interface LanguageInfo {
  code: string;
  name: string;
  aliases: string[];
  script: string | null;
  flores: string | null;
  source: boolean;
  target: boolean;
  indic: boolean;
  priority_indic: boolean;
  dedicated_engine: string | null;
  synth_lang: string;
  edge_supported: boolean;
  piper_supported: boolean;
  chars_per_second: number;
  cloning_supported_by_architecture: boolean;
  cloning_model_available_on_this_machine: boolean;
  cloning_engine: string | null;
}

export interface LanguagesResponse {
  languages: LanguageInfo[];
  sources: LanguageInfo[];
  targets: LanguageInfo[];
  auto_detect_code: string;
}

export interface BackendInfo {
  name: string;
  label: string;
  kind: string;
  available: boolean;
  state: ComponentState;
  reason: string;
  detail: Record<string, string>;
}

export interface PresetInfo {
  name: string;
  settings: Record<string, unknown>;
}

export interface RouteInfo {
  source: string;
  target: string;
  backend: string | null;
  pivots: string[];
  is_passthrough: boolean;
}

export interface CapabilitiesResponse {
  translation_backends: BackendInfo[];
  tts_backends: BackendInfo[];
  cloning_engines: string[];
  quality_presets: PresetInfo[];
  features: Record<string, boolean>;
  sources: string[];
  targets: string[];
  priority_indic: string[];
  routes: RouteInfo[];
  notes: string[];
}

export interface OptionItem {
  value: string;
  label: string;
  description: string | null;
  available: boolean;
  state: ComponentState;
}

export interface OptionsResponse {
  translation_models: OptionItem[];
  context_models: OptionItem[];
  transcription_models: OptionItem[];
  diarization_backends: OptionItem[];
  lipsync_models: OptionItem[];
  ocr_engines: OptionItem[];
  audio_separators: OptionItem[];
  live_portrait_modes: OptionItem[];
  tts_models: OptionItem[];
  subtitle_modes: OptionItem[];
  source: Record<string, string>;
}

export interface VoicesResponse {
  tts_model: string;
  language: string;
  language_name: string | null;
  voices: string[];
  cloning: boolean;
  note: string | null;
}

// ---------------------------------------------------------------------------
// Bundled demo samples
// ---------------------------------------------------------------------------

/**
 * One allow-listed bundled sample.
 *
 * `id` is the only handle a client uses; the server resolves it through its own
 * allow-list, so no path ever crosses the boundary. `available` is computed
 * server-side from the filesystem, so the demo page never has to guess whether
 * a sample can actually run.
 */
export interface SampleInfo {
  id: string;
  title: string;
  description: string;
  filename: string;
  duration_seconds: number | null;
  source_language: string;
  target_language: string;
  credit: string;
  size_bytes: number | null;
  media_type: string | null;
  available: boolean;
  reason: string | null;
}

export interface SampleListResponse {
  samples: SampleInfo[];
  total: number;
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

export interface StageStatus {
  name: string;
  status: string;
  error: string | null;
  attempts: number;
}

export interface SegmentCounts {
  total: number;
  complete: number;
  pending: number;
  failed: number;
  partial: number;
}

export interface OutputInfo {
  available: boolean;
  filename: string | null;
  size_bytes: number | null;
  media_type: string | null;
  media_url: string | null;
  download_url: string | null;
}

/**
 * Resume eligibility, decided server-side. The client renders `message` and
 * switches on `reason`; it must not re-derive the rules from `state`.
 */
export interface ResumeInfo {
  can_resume: boolean;
  reason: string;
  message: string;
}

/**
 * What a job did, from durable state. Every field is a recorded fact, so the
 * result page renders the same summary after a refresh or on another machine.
 * Optional fields are null for a job that predates them; the UI shows "unknown"
 * rather than guessing.
 */
export interface JobSummary {
  source_language: string | null;
  source_language_name: string | null;
  target_language: string | null;
  target_language_name: string | null;
  translation_backend: string | null;
  context_backend: string | null;
  optimize_translation: boolean;
  voice_mode: string;
  tts_backend: string | null;
  tts_voice: string | null;
  subtitle_mode: string;
  subtitle_count: number;
  lipsync: boolean;
  lipsync_model: string | null;
  diarization: boolean;
  speakers: number;
  duration_seconds: number | null;
  status: string;
  state: string;
}

export interface JobStatusResponse {
  job_id: string;
  state: JobState;
  status: string;
  run_mode: RunMode | null;
  source: Record<string, unknown>;
  source_name: string | null;
  source_language: string | null;
  target_language: string | null;
  quality_preset: string | null;
  config: Record<string, unknown>;
  stages: StageStatus[];
  segments: SegmentCounts;
  progress_fraction: number;
  live_progress: number | null;
  live_description: string | null;
  queue_position: number | null;
  created_at: number | null;
  updated_at: number | null;
  started_at: number | null;
  finished_at: number | null;
  output_available: boolean;
  output: OutputInfo;
  summary: JobSummary;
  resume: ResumeInfo;
  error: string | null;
  events_url: string;
  logs_url: string;
  artifacts_url: string;
  errors: Record<string, unknown>[];
}

export interface JobListResponse {
  jobs: JobStatusResponse[];
  total: number;
  active: number;
  pending: number;
}

// ---------------------------------------------------------------------------
// Uploads
// ---------------------------------------------------------------------------

/**
 * What the server learned about an uploaded file.
 *
 * Informational only: the authoritative check that a file is usable is the
 * pipeline's own validation, which the upload endpoint already ran. Every field
 * is optional because the server omits them when ffprobe is unavailable rather
 * than reporting a zero.
 */
export interface MediaInfo {
  container: string | null;
  duration: number | null;
  width: number | null;
  height: number | null;
  video_codec: string | null;
  audio_codec: string | null;
  has_video: boolean | null;
  has_audio: boolean | null;
}

export interface UploadResponse {
  upload_id: string;
  filename: string;
  bytes: number;
  media: MediaInfo;
  status: string;
  created_at: number;
  claimed_by: string | null;
}

/**
 * The body of `POST /api/jobs`.
 *
 * Mirrors `JobCreateRequest`. Every field is optional except `upload_id` and
 * each carries the same default the Gradio UI uses, so the client sends only
 * what the user actually changed. Values are validated server-side; the
 * frontend's own checks are a convenience, never the authority.
 */
export interface JobCreateRequest {
  upload_id: string;
  source_lang?: string;
  target_lang?: string;
  quality_preset?: string | null;
  audio_model_name?: string;
  tts_model_name?: string;
  translation_model_name?: string;
  context_model_name?: string;
  transcription_model_name?: string;
  optimize_translation?: boolean;
  enable_diarization?: boolean;
  diarization_model?: string;
  min_speakers?: number;
  max_speakers?: number;
  enable_time_stretch?: boolean;
  enable_vad?: boolean;
  vad_min_silence_duration_ms?: number;
  enable_lipsync?: boolean;
  lipsync_model_name?: string;
  live_portrait_mode?: string;
  enable_visual_translation?: boolean;
  ocr_model_name?: string;
  tts_voice?: string | null;
  transcription_beam_size?: number;
  tts_enable_cfg?: boolean;
  enable_audio_enhancement?: boolean;
  chunk_duration?: number;
  subtitle_mode?: string;
}

/**
 * One readiness check for a prospective render.
 *
 * `severity` drives the UI: `blocking` disables Start, `warning` does not,
 * `ok` is a satisfied check. `field` names the request field at fault so the
 * panel can point at the offending control.
 */
export interface PreflightCheck {
  id: string;
  label: string;
  severity: string;
  detail: string;
  field: string | null;
}

/**
 * The result of preflighting a render. Advisory: `POST /api/jobs` re-validates
 * everything, so this exists to explain a disabled Start button, not to replace
 * the server's check.
 */
export interface PreflightResponse {
  ready: boolean;
  checks: PreflightCheck[];
  blocking: PreflightCheck[];
  warnings: PreflightCheck[];
}

export interface CancelResponse {
  job_id: string;
  state: JobState;
  message: string;
}

export interface ResumeResponse {
  job_id: string;
  state: JobState;
  run_mode: RunMode;
  message: string;
  queue_position: number | null;
}

// ---------------------------------------------------------------------------
// Artifacts, logs and output
// ---------------------------------------------------------------------------

/**
 * One artifact the API is willing to hand out.
 *
 * `name` is the only handle the client uses; it resolves server-side through an
 * allow-list, so no client string ever becomes a path. `available` is false for
 * an artifact this render did not produce, with `reason` explaining why.
 */
export interface ArtifactDescriptor {
  name: string;
  type: string;
  label: string;
  filename: string | null;
  available: boolean;
  size_bytes: number | null;
  media_type: string | null;
  download_url: string | null;
  description: string;
  reason: string | null;
}

export interface ArtifactListResponse {
  job_id: string;
  artifacts: ArtifactDescriptor[];
  count: number;
  available_count: number;
}

/** A bounded window of a job's durable log, already redacted server-side. */
export interface LogsResponse {
  job_id: string;
  lines: string[];
  returned_lines: number;
  total_lines: number;
  has_more: boolean;
  truncated_lines: boolean;
  redacted: boolean;
}

export interface OutputResponse {
  job_id: string;
  available: boolean;
  filename: string | null;
  download_url: string | null;
  media_url: string | null;
  media_type: string | null;
  size_bytes: number | null;
}

export interface DeleteResponse {
  job_id: string;
  deleted: boolean;
  level: CleanupLevel;
  removed: string[];
  message: string;
}

// ---------------------------------------------------------------------------
// Error envelope
// ---------------------------------------------------------------------------

/**
 * The stable machine-readable error codes from `src/api/errors.py`. A client
 * switches on these; the `message` is for display only and may change.
 */
export type ApiErrorCode =
  | "INVALID_REQUEST"
  | "INVALID_SETTINGS"
  | "INVALID_JOB_ID"
  | "INVALID_UPLOAD"
  | "UPLOAD_TOO_LARGE"
  | "JOB_NOT_FOUND"
  | "UPLOAD_NOT_FOUND"
  | "ARTIFACT_UNAVAILABLE"
  | "OUTPUT_UNAVAILABLE"
  | "NOT_FOUND"
  | "JOB_ALREADY_ACTIVE"
  | "JOB_NOT_RESUMABLE"
  | "JOB_ACTIVE"
  | "UPLOAD_ALREADY_CLAIMED"
  | "METHOD_NOT_ALLOWED"
  | "TOO_MANY_SUBSCRIBERS"
  | "SERVICE_UNAVAILABLE"
  | "INTERNAL_ERROR"
  // Not from the server: raised by the client when the request never arrived.
  | "NETWORK_ERROR";

export interface ApiErrorBody {
  code: string;
  message: string;
  field?: string | null;
  details?: Record<string, unknown>;
}

export interface ApiErrorEnvelope {
  error: ApiErrorBody;
}
