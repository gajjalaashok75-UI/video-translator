/**
 * Pure helpers for the guided demo.
 *
 * The demo must not advertise a configuration the machine cannot run, and it
 * must not hard-code "green" checks. Everything here reads the backend's own
 * answers — `features`, the backend availability lists, the sample list — and
 * derives a readiness summary plus the demo defaults from them. No capability is
 * decided in this file; it is only *summarized*.
 *
 * The Indian-language focus is expressed by reading `priority_indic` from
 * `/api/capabilities` rather than re-listing the registry here, so the demo can
 * never drift from the languages the backend actually prioritizes.
 */

import type {
  BackendInfo,
  CapabilitiesResponse,
  ComponentStatus,
  LanguageInfo,
  OptionItem,
  OptionsResponse,
  SampleInfo,
} from "../api/types";

// ---------------------------------------------------------------------------
// Readiness
// ---------------------------------------------------------------------------

export type ReadinessState = "ready" | "missing" | "optional";

export interface ReadinessCheck {
  /** Stable key, used for tests and React keys. */
  id: string;
  label: string;
  state: ReadinessState;
  /** True when the demo cannot start without this. */
  blocking: boolean;
  /** The backend's own reason when something is missing. */
  detail: string | null;
}

export interface DemoReadiness {
  checks: ReadinessCheck[];
  /** True when every blocking check is satisfied. */
  ready: boolean;
  /** The first blocking problem, for the "Demo unavailable" message. */
  blocker: ReadinessCheck | null;
}

/** A backend entry with this name is present and available. */
function backendAvailable(
  backends: BackendInfo[] | undefined,
  name: string,
): BackendInfo | null {
  return (backends ?? []).find((b) => b.name === name && b.available) ?? null;
}

function modelAvailable(
  models: ComponentStatus[] | null | undefined,
  name: string,
): boolean {
  return Boolean((models ?? []).find((m) => m.name === name && m.available));
}

/**
 * Build the demo readiness summary from real backend state.
 *
 * The order is the order a user reads: the sample first (it is the thing they
 * are about to run), then the pipeline stages the demo needs, then the optional
 * components that would only enrich a run. Only the first five block a standard
 * demo; cloning and lip-sync are shown as optional and never block it.
 */
export function buildReadiness(input: {
  sample: SampleInfo | null;
  capabilities: CapabilitiesResponse | null;
  models: ComponentStatus[] | null;
}): DemoReadiness {
  const { sample, capabilities, models } = input;
  const features = capabilities?.features ?? {};

  const checks: ReadinessCheck[] = [
    {
      id: "sample",
      label: "Sample video",
      state: sample?.available ? "ready" : "missing",
      blocking: true,
      detail: sample?.available
        ? null
        : (sample?.reason ?? "The bundled sample is not available."),
    },
    {
      id: "ffmpeg",
      label: "Media tools (FFmpeg)",
      state: modelAvailable(models, "ffmpeg") ? "ready" : "missing",
      blocking: true,
      detail: modelAvailable(models, "ffmpeg")
        ? null
        : "FFmpeg is required to read and write video.",
    },
    {
      id: "transcription",
      label: "Speech recognition",
      state: features.transcription ? "ready" : "missing",
      blocking: true,
      detail: features.transcription
        ? null
        : "A speech-to-text model is required to transcribe the source.",
    },
    {
      id: "translation",
      label: "NLLB translation",
      state: backendAvailable(capabilities?.translation_backends, "nllb")
        ? "ready"
        : "missing",
      blocking: true,
      detail: backendAvailable(capabilities?.translation_backends, "nllb")
        ? null
        : "Local NLLB-200 translation is not currently available.",
    },
    {
      id: "tts",
      label: "Speech generation (Edge TTS)",
      state: backendAvailable(capabilities?.tts_backends, "edge")
        ? "ready"
        : "missing",
      blocking: true,
      detail: backendAvailable(capabilities?.tts_backends, "edge")
        ? null
        : "A speech-synthesis backend is required to produce the dub.",
    },
    {
      id: "cloning",
      label: "Voice cloning model",
      state: features.cloning ? "ready" : "optional",
      blocking: false,
      detail: features.cloning
        ? "A cloning model is installed; the standard demo does not use it."
        : "Not installed. The demo uses a standard generated voice.",
    },
    {
      id: "lipsync",
      label: "Lip-sync model",
      state: features.lip_sync ? "ready" : "optional",
      blocking: false,
      detail: features.lip_sync
        ? "Installed. Lip-sync is optional and off in the standard demo."
        : "Not installed. The demo does not use lip-sync.",
    },
  ];

  const blocker = checks.find((c) => c.blocking && c.state !== "ready") ?? null;
  return { checks, ready: blocker === null, blocker };
}

// ---------------------------------------------------------------------------
// Demo defaults
// ---------------------------------------------------------------------------

export interface DemoDefaults {
  sourceLang: string;
  targetLang: string;
  translationModel: string;
  ttsModel: string;
  qualityPreset: string | null;
}

/**
 * The demo's starting configuration, derived from capability, not hard-coded.
 *
 * The preferred route is English → Hindi with local NLLB and Edge TTS, because
 * that is the combination the project has validated end to end. But each part
 * is chosen only if the backend says it is available: if NLLB is missing the
 * router's "Auto" entry is used, if Edge is missing the first available TTS
 * backend is used, and if Hindi is not a target the first priority Indian
 * language is. A missing preset is simply left unset (custom).
 */
export function buildDemoDefaults(input: {
  sample: SampleInfo | null;
  capabilities: CapabilitiesResponse | null;
  languages: LanguageInfo[];
  targets: LanguageInfo[];
  ttsBackends: BackendInfo[];
  options?: OptionsResponse | null;
}): DemoDefaults {
  const { sample, capabilities, languages, targets, ttsBackends, options } = input;

  const targetCodes = new Set(targets.map((t) => t.code));
  const priority = capabilities?.priority_indic ?? [];

  // Target: the sample's own target if it is a real target, else the first
  // priority Indian language, else the first target.
  const preferredTargets = [
    sample?.target_language,
    ...priority,
  ].filter((code): code is string => Boolean(code));
  const targetLang =
    preferredTargets.find((code) => targetCodes.has(code)) ??
    targets[0]?.code ??
    "";

  // Source: the sample's own source when it is a real source, else auto.
  const sourceCodes = new Set(languages.map((l) => l.code));
  const sourceLang =
    sample?.source_language && sourceCodes.has(sample.source_language)
      ? sample.source_language
      : "auto";

  // Translation: prefer local NLLB by its exact option value when available.
  const nllbAvailable = backendAvailable(
    capabilities?.translation_backends,
    "nllb",
  );
  const translationModel = nllbAvailable
    ? NLLB_OPTION_VALUE
    : AUTO_TRANSLATION_OPTION_VALUE;

  // TTS: prefer Edge; otherwise the first available backend.
  const edgeAvailable = backendAvailable(ttsBackends, "edge");
  const firstTts = ttsBackends.find((b) => b.available);
  const ttsModel = edgeAvailable ? "edge" : (firstTts?.name ?? "");

  // Quality: "BALANCED" when this machine can actually run it, else no preset.
  const qualityPreset = demoPreset(capabilities, options);

  return { sourceLang, targetLang, translationModel, ttsModel, qualityPreset };
}

/**
 * The demo's quality preset: "BALANCED" when runnable, otherwise null.
 *
 * The spec's preferred demo route is "Balanced", so that is what is offered —
 * but only when every model the preset names is actually available. It reads the
 * preset's own `settings` and checks the specific models against `/api/options`,
 * which matters on the validated machine: BALANCED turns on lip-sync with the
 * Wav2Lip checkpoint, which is not downloaded here, so it would fail mid-render.
 * Rather than silently switching to a different (lip-sync) preset — which would
 * change what the demo does — it falls back to plain defaults, which is the
 * honest "standard dubbing" run. The readiness panel still shows lip-sync as an
 * optional component the user could enable in the full workspace.
 */
export function demoPreset(
  capabilities: CapabilitiesResponse | null,
  options?: OptionsResponse | null,
): string | null {
  const features = capabilities?.features ?? {};
  const presets = capabilities?.quality_presets ?? [];
  const balanced = presets.find((p) => p.name === "BALANCED");
  if (!balanced) return null;

  const settings = balanced.settings;
  const on = (key: string) => settings[key] === true;
  const optionAvailable = (list: OptionItem[] | undefined, value: unknown) => {
    if (typeof value !== "string") return true;
    const item = (list ?? []).find((o) => o.value === value);
    return item ? item.available : false;
  };

  if (on("enable_lipsync")) {
    if (!features.lip_sync) return null;
    if (!optionAvailable(options?.lipsync_models, settings.lipsync_model_name)) {
      return null;
    }
    if (
      settings.live_portrait_mode &&
      !optionAvailable(options?.live_portrait_modes, settings.live_portrait_mode)
    ) {
      return null;
    }
  }
  if (on("enable_audio_enhancement") && !features.audio_enhancement) return null;
  if (on("enable_visual_translation")) {
    if (!features.visual_translation) return null;
    if (!optionAvailable(options?.ocr_engines, settings.ocr_model_name)) return null;
  }
  return balanced.name;
}

/**
 * The option value the backend uses for local NLLB.
 *
 * These are the exact strings `/api/options` publishes; they are named here so a
 * reader can see the demo is selecting a real backend rather than a display
 * label, and so the drift is caught by the dubbing tests if the backend renames
 * one.
 */
export const NLLB_OPTION_VALUE = "NLLB-200 (Local, Indic + 200 Languages)";
export const AUTO_TRANSLATION_OPTION_VALUE =
  "Auto (NLLB-200 for Indic, else selected)";

/**
 * Popular demo targets, drawn from the backend's own priority list.
 *
 * Returns the priority Indian languages that are actually valid targets, in the
 * backend's order. The frontend never re-lists the IndicF5 registry; it only
 * reorders what the backend already prioritizes.
 */
export function popularDemoTargets(
  capabilities: CapabilitiesResponse | null,
  targets: LanguageInfo[],
  limit = 7,
): LanguageInfo[] {
  const byCode = new Map(targets.map((t) => [t.code, t]));
  const priority = capabilities?.priority_indic ?? [];
  const picked = priority
    .map((code) => byCode.get(code))
    .filter((language): language is LanguageInfo => Boolean(language))
    .slice(0, limit);
  if (picked.length > 0) return picked;
  // No priority list (or none are targets): fall back to the first targets so
  // the demo still offers something real rather than an empty selector.
  return targets.slice(0, limit);
}
