/**
 * The preflight verdict for the render the form would start.
 *
 * The backend decides what is ready and why not; this component only decides
 * which of three groups a check belongs to and how it reads. A blocking check is
 * the reason Start is disabled and is always shown, including its detail, so the
 * user is never left guessing. Warnings and satisfied checks are secondary: they
 * are available, but a form with every check listed is noise.
 */

import { Badge } from "../common/Badge";
import { IconCheck, IconInfo } from "../common/Icons";
import type { PreflightCheck } from "../../api/types";

const TONE = {
  blocking: "danger",
  warning: "warn",
  ok: "ok",
} as const;

const SEVERITY_LABEL: Record<string, string> = {
  blocking: "Blocks",
  warning: "Heads up",
  ok: "Ready",
};

function severityTone(severity: string) {
  return TONE[severity as keyof typeof TONE] ?? "info";
}

function severityLabel(severity: string): string {
  return SEVERITY_LABEL[severity] ?? severity;
}

export function PreflightPanel({
  checks,
  loading,
  error,
}: {
  checks: PreflightCheck[];
  loading: boolean;
  error: string | null;
}) {
  if (error) {
    // A failed preflight is not a failed render: say so and let Start proceed,
    // because `createJob` is the authority and it will re-check anyway.
    return (
      <div className="preflight-panel preflight-panel-muted">
        <span className="preflight-note">
          Could not check readiness ({error}). You can still start the render.
        </span>
      </div>
    );
  }

  if (checks.length === 0) {
    return (
      <div className="preflight-panel preflight-panel-muted">
        <span className="preflight-note">
          {loading ? "Checking readiness…" : "No checks to run yet."}
        </span>
      </div>
    );
  }

  const blocking = checks.filter((check) => check.severity === "blocking");
  const warnings = checks.filter((check) => check.severity === "warning");
  const ok = checks.filter((check) => check.severity === "ok");

  return (
    <div className="preflight-panel">
      {blocking.length > 0 ? (
        <ul className="readiness-list">
          {blocking.map((check) => (
            <li className="readiness-item" key={check.id}>
              <span className="readiness-mark readiness-mark-missing" aria-hidden="true">
                <IconInfo size={14} />
              </span>
              <div className="readiness-main">
                <span className="readiness-label">{check.label}</span>
                <span className="readiness-detail">{check.detail}</span>
              </div>
              <Badge tone={severityTone(check.severity)}>
                {severityLabel(check.severity)}
              </Badge>
            </li>
          ))}
        </ul>
      ) : (
        <div className="preflight-ready">
          <span className="readiness-mark readiness-mark-ready" aria-hidden="true">
            <IconCheck size={14} />
          </span>
          <span className="preflight-note">
            {loading ? "Checking readiness…" : "Ready to start."}
          </span>
        </div>
      )}

      {warnings.length > 0 && (
        <ul className="readiness-list preflight-warnings">
          {warnings.map((check) => (
            <li className="readiness-item" key={check.id}>
              <span className="readiness-mark readiness-mark-optional" aria-hidden="true">
                <IconInfo size={14} />
              </span>
              <div className="readiness-main">
                <span className="readiness-label">{check.label}</span>
                <span className="readiness-detail">{check.detail}</span>
              </div>
              <Badge tone={severityTone(check.severity)}>
                {severityLabel(check.severity)}
              </Badge>
            </li>
          ))}
        </ul>
      )}

      {blocking.length === 0 && warnings.length === 0 && ok.length > 0 && (
        <span className="preflight-note preflight-ok-count">
          {ok.length} checks passed.
        </span>
      )}
    </div>
  );
}
