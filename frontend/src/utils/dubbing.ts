/**
 * Pure helpers for the dubbing workflow.
 *
 * The rules that decide what the UI offers live here rather than inside a
 * component, so they can be tested without rendering anything. Nothing here
 * decides *capability* — that is always the backend's answer (`features`,
 * `resume.can_resume`). These functions only shape a request, summarize a stage
 * list, or order artifacts for display.
 */

import type {
  ArtifactDescriptor,
  JobCreateRequest,
  JobSummary,
  OptionItem,
  StageStatus,
} from "../api/types";
import { formatDuration } from "./format";

// ---------------------------------------------------------------------------
// Stages
// ---------------------------------------------------------------------------

/**
 * A stage's presentation state, collapsed from the backend's own vocabulary.
 *
 * The backend's `StageStatus.status` is one of `PENDING`, `RUNNING`, `COMPLETE`,
 * `PARTIAL`, `FAILED`, `SKIPPED`. The UI groups them into four tones so a
 * timeline reads at a glance, but never invents a state: an unknown string is
 * treated as pending rather than guessed at.
 */
export type StageTone = "done" | "active" | "failed" | "pending";

export function stageTone(status: string): StageTone {
  switch (status.toUpperCase()) {
    case "COMPLETE":
    case "SKIPPED":
      return "done";
    case "RUNNING":
      return "active";
    case "FAILED":
      return "failed";
    default:
      return "pending";
  }
}

/** A human label for a stage's raw status token. */
export function stageLabel(status: string): string {
  switch (status.toUpperCase()) {
    case "COMPLETE":
      return "Complete";
    case "SKIPPED":
      return "Skipped";
    case "RUNNING":
      return "Running";
    case "FAILED":
      return "Failed";
    case "PARTIAL":
      return "Partial";
    case "PENDING":
      return "Pending";
    default:
      return status || "Unknown";
  }
}

export interface StageProgress {
  done: number;
  total: number;
  /** Index of the first non-done stage, or -1 when all are done. */
  currentIndex: number;
}

/**
 * How far along a stage list is.
 *
 * `SKIPPED` counts as done — the stage will not run again — which is why this
 * cannot simply compare status to `COMPLETE`.
 */
export function stageProgress(stages: StageStatus[]): StageProgress {
  const done = stages.filter((stage) => stageTone(stage.status) === "done").length;
  const currentIndex = stages.findIndex(
    (stage) => stageTone(stage.status) !== "done",
  );
  return { done, total: stages.length, currentIndex };
}

// ---------------------------------------------------------------------------
// Option lists
// ---------------------------------------------------------------------------

/**
 * Options a dropdown may offer for a feature that is currently off.
 *
 * The backend marks every option with `available` and a `state`, and it does not
 * hide the unavailable ones — a user benefits from seeing that "NLLB-200" exists
 * but needs a download. So the list is not filtered; instead `isSelectable`
 * says whether the option may be *chosen right now*, and the component renders
 * the rest as disabled with the reason visible.
 */
export function isSelectable(option: OptionItem): boolean {
  return option.available;
}

/** The default option value for a list, or the first selectable, or "". */
export function defaultOptionValue(options: OptionItem[]): string {
  const selectable = options.find(isSelectable);
  return selectable?.value ?? options[0]?.value ?? "";
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

/**
 * The subset of the form the workflow collects, before it becomes a request.
 *
 * Kept separate from `JobCreateRequest` so the form can hold `null` for "not
 * chosen" and this function decides whether that becomes an omitted field or a
 * sent one. Sending `null` for a backend field that is not nullable is a 422, so
 * the conversion is explicit rather than spread.
 */
export interface DubFormValues {
  sourceLang: string;
  targetLang: string;
  qualityPreset: string | null;
  translationModel: string;
  transcriptionModel: string;
  ttsModel: string;
  ttsVoice: string | null;
  enableDiarization: boolean;
  minSpeakers: number;
  maxSpeakers: number;
  enableLipsync: boolean;
  lipsyncModel: string;
  enableVad: boolean;
  enableAudioEnhancement: boolean;
  enableVisualTranslation: boolean;
  enableTimeStretch: boolean;
  optimizeTranslation: boolean;
  ttsEnableCfg: boolean;
  transcriptionBeamSize: number;
  chunkDuration: number;
  subtitleMode: string;
}

/**
 * Turn the form into the request body, omitting anything left at its default.
 *
 * Only `upload_id`, `source_lang` and `target_lang` are always sent; the rest
 * are included only when the form holds a value, so the request stays small and
 * the backend's own defaults apply to everything untouched. A quality preset,
 * when set, overrides the individual lip-sync fields — the backend documents
 * that, so the two are not sent together to avoid implying the individual
 * switches won.
 */
export function buildJobRequest(
  uploadId: string,
  values: DubFormValues,
): JobCreateRequest {
  const request: JobCreateRequest = {
    upload_id: uploadId,
    source_lang: values.sourceLang,
    target_lang: values.targetLang,
  };

  if (values.qualityPreset) {
    request.quality_preset = values.qualityPreset;
    // Lip-sync switches are owned by the preset; sending both would be
    // contradictory. Everything else still applies.
    request.translation_model_name = values.translationModel || undefined;
    request.transcription_model_name = values.transcriptionModel || undefined;
    request.tts_model_name = values.ttsModel || undefined;
    request.tts_voice = values.ttsVoice ?? undefined;
    request.enable_diarization = values.enableDiarization;
    if (values.enableDiarization) {
      request.min_speakers = values.minSpeakers;
      request.max_speakers = values.maxSpeakers;
    }
    request.enable_vad = values.enableVad;
    request.enable_visual_translation = values.enableVisualTranslation;
    request.enable_time_stretch = values.enableTimeStretch;
    request.optimize_translation = values.optimizeTranslation;
    request.tts_enable_cfg = values.ttsEnableCfg;
    request.transcription_beam_size = values.transcriptionBeamSize;
    request.chunk_duration = values.chunkDuration;
    request.subtitle_mode = values.subtitleMode || undefined;
    return request;
  }

  request.translation_model_name = values.translationModel || undefined;
  request.transcription_model_name = values.transcriptionModel || undefined;
  request.tts_model_name = values.ttsModel || undefined;
  request.tts_voice = values.ttsVoice ?? undefined;
  request.enable_diarization = values.enableDiarization;
  if (values.enableDiarization) {
    request.min_speakers = values.minSpeakers;
    request.max_speakers = values.maxSpeakers;
  }
  request.enable_lipsync = values.enableLipsync;
  if (values.enableLipsync) {
    request.lipsync_model_name = values.lipsyncModel || undefined;
  }
  request.enable_vad = values.enableVad;
  request.enable_audio_enhancement = values.enableAudioEnhancement;
  request.enable_visual_translation = values.enableVisualTranslation;
  request.enable_time_stretch = values.enableTimeStretch;
  request.optimize_translation = values.optimizeTranslation;
  request.tts_enable_cfg = values.ttsEnableCfg;
  request.transcription_beam_size = values.transcriptionBeamSize;
  request.chunk_duration = values.chunkDuration;
  request.subtitle_mode = values.subtitleMode || undefined;
  return request;
}

/**
 * Whether a source/target pair is worth submitting.
 *
 * A pure convenience check so the button can be disabled early; the backend
 * remains the authority and re-validates. An unknown source is only invalid when
 * it is not the auto-detect code, which the caller passes in because it comes
 * from the API rather than being a frontend constant.
 */
export function isLanguagePairValid(
  sourceLang: string,
  targetLang: string,
  autoDetectCode: string,
): boolean {
  if (!sourceLang || !targetLang) return false;
  if (sourceLang !== autoDetectCode && sourceLang === targetLang) return false;
  return true;
}

// ---------------------------------------------------------------------------
// Artifacts
// ---------------------------------------------------------------------------

/**
 * Artifacts in the order a person wants them: the finished video first, then
 * playable media, then text and data.
 */
const ARTIFACT_TYPE_ORDER: Record<string, number> = {
  video: 0,
  subtitle: 1,
  transcript: 2,
  json: 3,
};

export function orderArtifacts(
  artifacts: ArtifactDescriptor[],
): ArtifactDescriptor[] {
  return [...artifacts].sort((a, b) => {
    const rankA = ARTIFACT_TYPE_ORDER[a.type] ?? 9;
    const rankB = ARTIFACT_TYPE_ORDER[b.type] ?? 9;
    if (rankA !== rankB) return rankA - rankB;
    return a.label.localeCompare(b.label);
  });
}

// ---------------------------------------------------------------------------
// Result summary
// ---------------------------------------------------------------------------

/** One labelled line of the result summary, ready to render. */
export interface SummaryRow {
  label: string;
  value: string;
}

const SUBTITLE_MODE_LABELS: Record<string, string> = {
  none: "None",
  external: "Sidecar files (.srt/.vtt)",
  embedded: "Embedded track",
  burned: "Burned into the picture",
};

const VOICE_MODE_LABELS: Record<string, string> = {
  standard: "Standard voice",
  preserve_speaker: "Preserved per speaker",
};

/**
 * The durable summary as rows, for the result page.
 *
 * The backend sends facts, not sentences: `voice_mode` is `standard` or
 * `preserve_speaker`, `subtitle_mode` is one of the four registry values. The
 * mapping to a phrase belongs here, and an unrecognised value is shown verbatim
 * rather than hidden — a new backend mode must not disappear from the page.
 *
 * Rows are omitted when the fact is absent, so a job that predates a field shows
 * a shorter panel rather than a row reading "null".
 */
export function summaryRows(summary: JobSummary): SummaryRow[] {
  const rows: SummaryRow[] = [];

  const source = summary.source_language_name ?? summary.source_language;
  const target = summary.target_language_name ?? summary.target_language;
  if (source || target) {
    rows.push({
      label: "Languages",
      value: `${source ?? "unknown"} → ${target ?? "unknown"}`,
    });
  }
  if (summary.translation_backend) {
    rows.push({ label: "Translation", value: summary.translation_backend });
  }
  if (summary.optimize_translation && summary.context_backend) {
    rows.push({ label: "Context model", value: summary.context_backend });
  }
  if (summary.tts_backend) {
    rows.push({
      label: "Speech synthesis",
      value: summary.tts_voice
        ? `${summary.tts_backend} · ${summary.tts_voice}`
        : summary.tts_backend,
    });
  }
  rows.push({
    label: "Voice",
    value: VOICE_MODE_LABELS[summary.voice_mode] ?? summary.voice_mode,
  });
  if (summary.diarization) {
    rows.push({
      label: "Speakers",
      value:
        summary.speakers > 0 ? `${summary.speakers} detected` : "detection ran",
    });
  }
  rows.push({
    label: "Subtitles",
    value:
      (SUBTITLE_MODE_LABELS[summary.subtitle_mode] ?? summary.subtitle_mode) +
      (summary.subtitle_count > 0 ? ` · ${summary.subtitle_count} cues` : ""),
  });
  if (summary.lipsync) {
    rows.push({ label: "Lip-sync", value: summary.lipsync_model ?? "on" });
  }
  if (
    summary.duration_seconds !== null &&
    summary.duration_seconds !== undefined
  ) {
    rows.push({
      label: "Duration",
      value: formatDuration(summary.duration_seconds),
    });
  }
  return rows;
}
