/**
 * One job in the list.
 *
 * Everything shown comes from the `JobStatusResponse`: the state badge uses the
 * backend's `state`, the progress text prefers the backend's own
 * `live_description`, and the resume affordance is driven by
 * `resume.can_resume` — the frontend never re-derives whether a job is
 * resumable. When it cannot be resumed, the backend's own `resume.message` is
 * the explanation, surfaced as the button's title.
 */

import { Link } from "react-router-dom";
import { Badge } from "../common/Badge";
import { IconArrowRight, IconRefresh } from "../common/Icons";
import { formatPercent, formatRelativeTime } from "../../utils/format";
import { jobStatePresentation } from "../../utils/status";
import { displayLanguage } from "../../utils/languages";
import type { JobStatusResponse, LanguageInfo } from "../../api/types";

interface JobRowProps {
  job: JobStatusResponse;
  onResume: (jobId: string) => void;
  resuming: boolean;
  /**
   * The language registry, used only to normalize the route for display.
   * Optional so the row can render before the list has loaded; without it the
   * raw stored value is shown.
   */
  languages?: LanguageInfo[];
}

export function JobRow({ job, onResume, resuming, languages = [] }: JobRowProps) {
  const presentation = jobStatePresentation(job.state);
  const progress = Math.round(job.progress_fraction * 100);
  const activeStage = job.stages.find((stage) => stage.status === "RUNNING");
  const description =
    job.live_description ??
    (activeStage ? activeStage.name : job.status) ??
    job.state;

  return (
    <div className="job-row">
      <div>
        <Link
          className="job-name job-name-link"
          to={`/jobs/${encodeURIComponent(job.job_id)}`}
          title={job.source_name ?? job.job_id}
        >
          {job.source_name ?? "Untitled source"}
        </Link>
        <div className="job-id">{job.job_id}</div>
      </div>

      <div className="job-route">
        <span>{displayLanguage(job.source_language, languages, "AUTO")}</span>
        <IconArrowRight size={14} />
        <span>{displayLanguage(job.target_language, languages)}</span>
      </div>

      <div className="job-progress">
        <div className="job-progress-text">
          <span className="job-progress-label" title={String(description)}>
            {description}
          </span>
          <span>{formatPercent(job.progress_fraction)}</span>
        </div>
        <div className="progress">
          <div
            className="progress-bar"
            style={{ width: `${progress}%` }}
            role="progressbar"
            aria-valuenow={progress}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`Progress ${progress} percent`}
          />
        </div>
        <div className="job-meta">
          {job.quality_preset ? `${job.quality_preset} · ` : ""}
          Updated {formatRelativeTime(job.updated_at ?? job.created_at)}
          {job.output_available ? " · Output ready" : ""}
        </div>
      </div>

      <div className="job-actions">
        <Badge tone={presentation.tone} dot>
          {presentation.label}
        </Badge>
        {job.output_available && (
          <Link
            className="btn btn-secondary"
            to={`/jobs/${encodeURIComponent(job.job_id)}/result`}
          >
            Result
            <IconArrowRight size={14} />
          </Link>
        )}
        {job.resume.can_resume && (
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => onResume(job.job_id)}
            disabled={resuming}
            title={job.resume.message}
          >
            <IconRefresh size={14} />
            {resuming ? "Resuming…" : "Resume"}
          </button>
        )}
      </div>
    </div>
  );
}
