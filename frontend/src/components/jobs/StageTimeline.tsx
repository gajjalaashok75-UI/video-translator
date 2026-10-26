/**
 * The pipeline's stage list as a vertical timeline.
 *
 * The stage *names and order* come from the backend (`JobStatusResponse.stages`,
 * built from `config.JOB_STAGES`); this component never hard-codes them, so a
 * stage added server-side appears without a frontend change. Only the wording
 * and color of each status are decided here.
 */

import { Badge } from "../common/Badge";
import { cx } from "../../utils/format";
import { stageLabel, stageTone } from "../../utils/dubbing";
import type { StageStatus } from "../../api/types";

const TONE_CLASS: Record<string, string> = {
  done: "stage-done",
  active: "stage-active",
  failed: "stage-failed",
  pending: "stage-pending",
};

export function StageTimeline({ stages }: { stages: StageStatus[] }) {
  if (stages.length === 0) {
    return (
      <p className="card-subtitle">
        No stages recorded yet; the manifest is written as the job starts.
      </p>
    );
  }

  return (
    <ol className="stage-list">
      {stages.map((stage) => {
        const tone = stageTone(stage.status);
        return (
          <li key={stage.name} className={cx("stage-item", TONE_CLASS[tone])}>
            <span className="stage-marker" aria-hidden="true" />
            <div className="stage-main">
              <span className="stage-name">{stage.name}</span>
              {stage.attempts > 1 && (
                <span className="stage-attempts">
                  {stage.attempts} attempts
                </span>
              )}
              {stage.error && (
                <span className="stage-error" title={stage.error}>
                  {stage.error}
                </span>
              )}
            </div>
            <Badge tone={tone === "done" ? "ok" : tone === "active" ? "accent" : tone === "failed" ? "danger" : "neutral"}>
              {stageLabel(stage.status)}
            </Badge>
          </li>
        );
      })}
    </ol>
  );
}
