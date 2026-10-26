/**
 * Test fixtures.
 *
 * These are trimmed copies of real backend responses, captured from the running
 * API. Using real shapes (rather than invented ones) is what makes the type
 * mirror meaningful: if a field is renamed in `schemas.py`, these fixtures and
 * the types disagree and the tests fail, instead of passing against a fiction.
 */

import type {
  ArtifactListResponse,
  CapabilitiesResponse,
  ComponentStatus,
  HealthResponse,
  JobListResponse,
  JobStatusResponse,
  LanguagesResponse,
  LogsResponse,
  MediaInfo,
  OutputResponse,
  SampleInfo,
  SampleListResponse,
  SystemResponse,
  UploadResponse,
} from "../api/types";

export const healthFixture: HealthResponse = {
  status: "ok",
  api_version: "0.3.0",
  uptime_seconds: 128.5,
};

export const systemFixture: SystemResponse = {
  api_version: "0.3.0",
  app_version: "1.4.0",
  python_version: "3.12.14",
  python_implementation: "CPython",
  platform: "Linux",
  platform_release: "6.8.0",
  machine: "x86_64",
  processor: "x86_64",
  device: "cpu",
  cuda_available: false,
  cuda_version: null,
  cuda_runtime_version: null,
  gpu_name: null,
  gpu_count: 0,
  gpu_memory_total_mb: null,
  torch_version: "2.14.1+cpu",
  ffmpeg_available: true,
  ffmpeg_version: "ffmpeg version 6.1.1",
  ffprobe_available: true,
  jobs_dir_writable: true,
};

export const componentFixtures: ComponentStatus[] = [
  {
    name: "ffmpeg",
    label: "FFmpeg",
    state: "available",
    available: true,
    reason: "Found on PATH.",
    required_for: ["core"],
    detail: {},
  },
  {
    name: "indicf5",
    label: "IndicF5 (Indic voice cloning)",
    state: "not_installed",
    available: false,
    reason: "The f5_tts package is not installed.",
    required_for: ["tts", "cloning"],
    detail: {},
  },
  {
    name: "wav2lip",
    label: "Wav2Lip lip-sync",
    state: "model_not_downloaded",
    available: false,
    reason: "Checkpoint not present in models/.",
    required_for: ["lip_sync"],
    detail: {},
  },
];

export const languagesFixture: LanguagesResponse = {
  auto_detect_code: "auto",
  languages: [
    {
      code: "hi",
      name: "Hindi",
      aliases: ["Hindi", "हिन्दी"],
      script: "Deva",
      flores: "hin_Deva",
      source: true,
      target: true,
      indic: true,
      priority_indic: true,
      dedicated_engine: "indicf5",
      synth_lang: "hi",
      edge_supported: true,
      piper_supported: true,
      chars_per_second: 14.2,
      cloning_supported_by_architecture: true,
      cloning_model_available_on_this_machine: false,
      cloning_engine: "indicf5",
    },
    {
      code: "en",
      name: "English",
      aliases: ["English"],
      script: "Latn",
      flores: "eng_Latn",
      source: true,
      target: true,
      indic: false,
      priority_indic: false,
      dedicated_engine: null,
      synth_lang: "en",
      edge_supported: true,
      piper_supported: true,
      chars_per_second: 15.5,
      cloning_supported_by_architecture: true,
      cloning_model_available_on_this_machine: false,
      cloning_engine: "xtts",
    },
    {
      code: "ta",
      name: "Tamil",
      aliases: ["Tamil", "தமிழ்"],
      script: "Taml",
      flores: "tam_Taml",
      source: true,
      target: true,
      indic: true,
      priority_indic: true,
      dedicated_engine: "indicf5",
      synth_lang: "ta",
      edge_supported: true,
      piper_supported: false,
      chars_per_second: 13.0,
      cloning_supported_by_architecture: true,
      cloning_model_available_on_this_machine: false,
      cloning_engine: "indicf5",
    },
  ],
  sources: [],
  targets: [],
};

// `sources`/`targets` in the fixture are filled by the tests that need them.
languagesFixture.sources = languagesFixture.languages.filter((l) => l.source);
languagesFixture.targets = languagesFixture.languages.filter((l) => l.target);

export const capabilitiesFixture: CapabilitiesResponse = {
  translation_backends: [
    {
      name: "nllb",
      label: "NLLB-200",
      kind: "local",
      available: false,
      state: "not_installed",
      reason: "transformers not installed.",
      detail: {},
    },
  ],
  tts_backends: [],
  cloning_engines: ["indicf5", "xtts"],
  quality_presets: [
    { name: "balanced", settings: {} },
    { name: "high", settings: {} },
  ],
  features: {
    transcription: false,
    translation_local: false,
    translation_online: false,
    tts: false,
    cloning: false,
    lip_sync: false,
    audio_enhancement: false,
    visual_translation: false,
    separation: false,
    diarization: false,
    time_stretch: false,
    chunked_processing: true,
    resume: true,
  },
  sources: ["en", "hi", "ta"],
  targets: ["en", "hi", "ta"],
  priority_indic: ["hi", "ta"],
  routes: [
    {
      source: "en",
      target: "hi",
      backend: "nllb",
      pivots: [],
      is_passthrough: false,
    },
  ],
  notes: ["Translation backends are not installed."],
};

export function jobFixture(
  overrides: Partial<JobStatusResponse> = {},
): JobStatusResponse {
  return {
    job_id: "job-abc123",
    state: "completed",
    status: "complete",
    run_mode: "new",
    source: {},
    source_name: "lecture.mp4",
    source_language: "en",
    target_language: "hi",
    quality_preset: "balanced",
    config: {},
    stages: [],
    segments: { total: 42, complete: 42, pending: 0, failed: 0, partial: 0 },
    progress_fraction: 1,
    live_progress: null,
    live_description: null,
    queue_position: null,
    created_at: 1_700_000_000,
    updated_at: 1_700_000_600,
    started_at: 1_700_000_010,
    finished_at: 1_700_000_600,
    output_available: true,
    output: {
      available: true,
      filename: "lecture_hi.mp4",
      size_bytes: 12_345_678,
      media_type: "video/mp4",
      media_url: "/api/jobs/job-abc123/artifacts/output",
      download_url: "/api/jobs/job-abc123/artifacts/output?download=true",
    },
    resume: {
      can_resume: false,
      reason: "complete",
      message: "This job is already complete.",
    },
    summary: {
      source_language: "en",
      source_language_name: "English",
      target_language: "hi",
      target_language_name: "Hindi",
      translation_backend: "Google Translate (Online, Fast)",
      context_backend: null,
      optimize_translation: false,
      voice_mode: "standard",
      tts_backend: "edge",
      tts_voice: null,
      subtitle_mode: "external",
      subtitle_count: 42,
      lipsync: false,
      lipsync_model: null,
      diarization: false,
      speakers: 0,
      duration_seconds: 600,
      status: "COMPLETE",
      state: "completed",
    },
    error: null,
    events_url: "/api/jobs/job-abc123/events",
    logs_url: "/api/jobs/job-abc123/logs",
    artifacts_url: "/api/jobs/job-abc123/artifacts",
    errors: [],
    ...overrides,
  };
}

export const jobListFixture: JobListResponse = {
  jobs: [
    jobFixture(),
    jobFixture({
      job_id: "job-def456",
      state: "interrupted",
      status: "interrupted",
      source_name: "interview.mov",
      source_language: "hi",
      target_language: "en",
      progress_fraction: 0.45,
      output_available: false,
      output: {
        available: false,
        filename: null,
        size_bytes: null,
        media_type: null,
        media_url: null,
        download_url: null,
      },
      resume: {
        can_resume: true,
        reason: "interrupted",
        message: "This job can be resumed; completed stages will be reused.",
      },
    }),
  ],
  total: 2,
  active: 0,
  pending: 0,
};

// ---------------------------------------------------------------------------
// F5: uploads, artifacts and logs
// ---------------------------------------------------------------------------

export const mediaInfoFixture: MediaInfo = {
  container: "matroska,webm",
  duration: 12.5,
  width: 1280,
  height: 720,
  video_codec: "vp9",
  audio_codec: "opus",
  has_video: true,
  has_audio: true,
};

export function uploadFixture(
  overrides: Partial<UploadResponse> = {},
): UploadResponse {
  return {
    upload_id: "up-1f2c3d4e5f60718293a4b5c6d7e8f901",
    filename: "lecture.webm",
    bytes: 1_048_576,
    media: mediaInfoFixture,
    status: "staged",
    created_at: 1_700_000_000,
    claimed_by: null,
    ...overrides,
  };
}

export const artifactListFixture: ArtifactListResponse = {
  job_id: "job-abc123",
  artifacts: [
    {
      name: "output",
      type: "video",
      label: "Dubbed video",
      filename: "lecture_hi.mp4",
      available: true,
      size_bytes: 12_345_678,
      media_type: "video/mp4",
      download_url: "/api/jobs/job-abc123/artifacts/output?download=true",
      description: "The finished, dubbed video.",
      reason: null,
    },
    {
      name: "subtitles",
      type: "subtitle",
      label: "Subtitles (SRT)",
      filename: "lecture_hi.srt",
      available: true,
      size_bytes: 4096,
      media_type: "application/x-subrip",
      download_url: "/api/jobs/job-abc123/artifacts/subtitles?download=true",
      description: "Target-language subtitles.",
      reason: null,
    },
    {
      name: "lipsync",
      type: "video",
      label: "Lip-synced video",
      filename: null,
      available: false,
      size_bytes: null,
      media_type: null,
      download_url: null,
      description: "Video with re-animated lip movement.",
      reason: "Lip-sync was not enabled for this render.",
    },
  ],
  count: 3,
  available_count: 2,
};

export function logsFixture(overrides: Partial<LogsResponse> = {}): LogsResponse {
  return {
    job_id: "job-abc123",
    lines: [
      "job created from upload up-1f2c",
      "extraction: complete",
      "transcription: 42 segments",
    ],
    returned_lines: 3,
    total_lines: 3,
    has_more: false,
    truncated_lines: false,
    redacted: true,
    ...overrides,
  };
}

export function outputFixture(
  overrides: Partial<OutputResponse> = {},
): OutputResponse {
  return {
    job_id: "job-abc123",
    available: true,
    filename: "lecture_hi.mp4",
    download_url: "/api/jobs/job-abc123/output/download",
    media_url: "/api/jobs/job-abc123/artifacts/output",
    media_type: "video/mp4",
    size_bytes: 12_345_678,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// F6: bundled demo samples
// ---------------------------------------------------------------------------

export function sampleFixture(
  overrides: Partial<SampleInfo> = {},
): SampleInfo {
  return {
    id: "hgf_scene",
    title: "His Girl Friday — Sample Scene",
    description: "A short public-domain scene used for validation.",
    filename: "hgf_scene.webm",
    duration_seconds: 96.7,
    source_language: "en",
    target_language: "hi",
    credit: "His Girl Friday (1940) — public domain.",
    size_bytes: 8_777_418,
    media_type: "video/webm",
    available: true,
    reason: null,
    ...overrides,
  };
}

export const samplesFixture: SampleListResponse = {
  samples: [sampleFixture()],
  total: 1,
};
