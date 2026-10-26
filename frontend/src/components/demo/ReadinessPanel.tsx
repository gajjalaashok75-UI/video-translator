/**
 * The demo readiness list.
 *
 * Renders exactly the checks `buildReadiness` derived from backend state — this
 * component decides nothing itself. A missing blocking check reads as a problem
 * with a remedy ("System status" is linked from the page); an unavailable
 * optional component reads as neutral, not as an error, because the standard
 * demo runs without it.
 */

import { Badge } from "../common/Badge";
import { IconCheck, IconInfo } from "../common/Icons";
import type { DemoReadiness, ReadinessCheck } from "../../utils/demo";

const TONE = {
  ready: "ok",
  missing: "danger",
  optional: "info",
} as const;

function statusLabel(check: ReadinessCheck): string {
  if (check.state === "ready") return "Ready";
  if (check.state === "optional") return "Optional";
  return "Missing";
}

export function ReadinessPanel({ readiness }: { readiness: DemoReadiness }) {
  return (
    <ul className="readiness-list">
      {readiness.checks.map((check) => (
        <li className="readiness-item" key={check.id}>
          <span
            className={`readiness-mark readiness-mark-${check.state}`}
            aria-hidden="true"
          >
            {check.state === "ready" ? (
              <IconCheck size={14} />
            ) : (
              <IconInfo size={14} />
            )}
          </span>
          <div className="readiness-main">
            <span className="readiness-label">{check.label}</span>
            {check.detail && (
              <span className="readiness-detail">{check.detail}</span>
            )}
          </div>
          <Badge tone={TONE[check.state]}>{statusLabel(check)}</Badge>
        </li>
      ))}
    </ul>
  );
}
