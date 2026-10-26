/**
 * Presentation mapping for backend states.
 *
 * The *values* come from the API (`ComponentState`, `JobState`); this file only
 * decides how each is worded and colored. It is not a second registry of
 * availability — the state itself is always read from the response, and the
 * exact server `reason` is shown alongside the badge rather than replaced.
 */

import type { ComponentState, JobState } from "../api/types";

export type BadgeTone = "ok" | "warn" | "danger" | "info" | "accent" | "neutral";

export interface StatePresentation {
  label: string;
  tone: BadgeTone;
}

/**
 * How each component state is described to a user.
 *
 * The wording is the point: "Model required" tells a user what to do, where a
 * bare "unavailable" does not. The server's own `reason` string is rendered as
 * secondary text so nothing is lost.
 */
const COMPONENT_STATES: Record<ComponentState, StatePresentation> = {
  available: { label: "Available", tone: "ok" },
  not_installed: { label: "Not installed", tone: "warn" },
  model_not_downloaded: { label: "Model required", tone: "warn" },
  token_required: { label: "Token required", tone: "info" },
  unsupported_platform: { label: "Unsupported", tone: "neutral" },
  unavailable: { label: "Unavailable", tone: "danger" },
  unknown: { label: "Unknown", tone: "neutral" },
};

export function componentStatePresentation(
  state: ComponentState,
): StatePresentation {
  return COMPONENT_STATES[state] ?? { label: "Unknown", tone: "neutral" };
}

const JOB_STATES: Record<JobState, StatePresentation> = {
  queued: { label: "Queued", tone: "neutral" },
  running: { label: "Running", tone: "accent" },
  completed: { label: "Complete", tone: "ok" },
  failed: { label: "Failed", tone: "danger" },
  cancelled: { label: "Cancelled", tone: "warn" },
  interrupted: { label: "Interrupted", tone: "warn" },
  unknown: { label: "Unknown", tone: "neutral" },
};

export function jobStatePresentation(state: JobState): StatePresentation {
  return JOB_STATES[state] ?? { label: "Unknown", tone: "neutral" };
}

/** Tone for a boolean capability, as a labeled badge rather than a dot alone. */
export function availabilityPresentation(available: boolean): StatePresentation {
  return available
    ? { label: "Available", tone: "ok" }
    : { label: "Not available", tone: "neutral" };
}
