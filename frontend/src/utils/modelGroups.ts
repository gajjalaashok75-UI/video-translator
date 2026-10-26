/**
 * Group the `/api/models` probe list into readable sections.
 *
 * The grouping key is the backend's own `required_for` tag, not a frontend list
 * of component names — so a new probe appears in a sensible group without a
 * frontend change, and no model name is hard-coded here. A component that
 * belongs to several features (IndicF5 is both TTS and cloning) is filed under
 * its most specific tag, so it is listed once rather than duplicated.
 */

import type { ComponentStatus } from "../api/types";

export interface ModelGroup {
  id: string;
  title: string;
  description: string;
  models: ComponentStatus[];
}

/**
 * Group titles and the tag that defines each, in display order. The first tag a
 * model has in this list wins, so the list is ordered most-specific-first.
 */
const GROUP_DEFINITIONS: Array<{
  id: string;
  tag: string;
  title: string;
  description: string;
}> = [
  {
    id: "core",
    tag: "core",
    title: "Core",
    description: "Required for any job.",
  },
  {
    id: "asr",
    tag: "transcription",
    title: "Speech recognition",
    description: "Turns speech into text.",
  },
  {
    id: "translation",
    tag: "translation",
    title: "Translation",
    description: "Translates the transcript.",
  },
  {
    id: "tts",
    tag: "tts",
    title: "Voice & speech synthesis",
    description: "Generates the dubbed speech.",
  },
  {
    id: "cloning",
    tag: "cloning",
    title: "Voice cloning",
    description: "Reproduces a reference voice. Optional.",
  },
  {
    id: "speaker",
    tag: "diarization",
    title: "Speaker & audio processing",
    description: "Separates speakers and stems.",
  },
  {
    id: "separation",
    tag: "separation",
    title: "Audio separation",
    description: "Isolates dialogue from music and effects.",
  },
  {
    id: "lipsync",
    tag: "lip_sync",
    title: "Lip-sync",
    description: "Matches mouth movement to the new audio. Optional.",
  },
  {
    id: "enhancement",
    tag: "audio_enhancement",
    title: "Enhancement & timing",
    description: "Post-processing of the mixed audio. Optional.",
  },
  {
    id: "visual",
    tag: "visual_translation",
    title: "Visual text translation",
    description: "Translates on-screen text. Optional.",
  },
];

/** Fallback group for a probe whose tags match none of the definitions. */
const OTHER = {
  id: "other",
  title: "Other components",
  description: "Not classified by the backend.",
};

export function groupModels(models: ComponentStatus[]): ModelGroup[] {
  const buckets = new Map<string, ModelGroup>();

  for (const model of models) {
    const definition = GROUP_DEFINITIONS.find((candidate) =>
      model.required_for.includes(candidate.tag),
    );
    const id = definition?.id ?? OTHER.id;
    if (!buckets.has(id)) {
      buckets.set(id, {
        id,
        title: definition?.title ?? OTHER.title,
        description: definition?.description ?? OTHER.description,
        models: [],
      });
    }
    buckets.get(id)!.models.push(model);
  }

  // Return in the definition order, then any leftover groups.
  const ordered: ModelGroup[] = [];
  for (const definition of GROUP_DEFINITIONS) {
    const group = buckets.get(definition.id);
    if (group) ordered.push(group);
  }
  const other = buckets.get(OTHER.id);
  if (other) ordered.push(other);
  return ordered;
}
