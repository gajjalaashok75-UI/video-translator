/**
 * The live backend status indicator.
 *
 * Wording is deliberate: "Checking" is a neutral state on first load, distinct
 * from "Offline". The change is announced through `aria-live` because a user who
 * cannot see the dot still needs to know the backend went away.
 */

import { useBackendHealth } from "../../hooks/useHealth";
import type { BackendStatus } from "../../hooks/backendStatus";

const LABELS: Record<BackendStatus, string> = {
  online: "Backend Online",
  offline: "Backend Offline",
  checking: "Checking backend",
};

export function BackendIndicator({ compact = false }: { compact?: boolean }) {
  const { status, health } = useBackendHealth();

  return (
    <span
      className="backend-indicator"
      data-status={status}
      title={
        health ? `API v${health.api_version}` : "No response from /api/health"
      }
    >
      <span className="backend-dot" aria-hidden="true" />
      <span aria-live="polite" aria-atomic="true">
        {compact ? LABELS[status].replace("Backend ", "") : LABELS[status]}
      </span>
    </span>
  );
}
