/**
 * A single job: live progress, logs, and the actions the backend allows.
 *
 * This is the page the event stream exists for. It renders from
 * `useJobStream`, which folds SSE events into the durable `JobStatusResponse`
 * rather than reconstructing state from events alone — so a reload mid-render,
 * or a job that finished while the tab was closed, shows the same thing.
 *
 * The two actions are driven entirely by the backend's own answers:
 * `resume.can_resume` decides whether Resume appears and `resume.message` is its
 * explanation; Cancel is offered while the job is in a non-terminal state, and
 * its result is treated as a request (the job may still be running at the next
 * checkpoint), never as an accomplished fact.
 */

import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { cancelJob, resumeJob } from "../api/endpoints";
import { useJobStream, isTerminalState } from "../hooks/useJobStream";
import { useJobLogs } from "../hooks/useJobLogs";
import { useLanguages } from "../hooks/useLanguages";
import { useBackendHealth } from "../hooks/useHealth";
import { Badge } from "../components/common/Badge";
import { ErrorState, SkeletonCard } from "../components/common/States";
import { OfflineBanner } from "../components/status/OfflineBanner";
import { StageTimeline } from "../components/jobs/StageTimeline";
import { LogPanel } from "../components/jobs/LogPanel";
import {
  IconArrowRight,
  IconRefresh,
  IconStop,
} from "../components/common/Icons";
import { formatPercent, formatRelativeTime } from "../utils/format";
import { displayLanguage } from "../utils/languages";
import { jobStatePresentation } from "../utils/status";
import { stageProgress } from "../utils/dubbing";

export function JobDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const { status } = useBackendHealth();
  const languages = useLanguages();
  const stream = useJobStream(jobId);
  // The durable log is the complete record; the live lines fill the gap until it
  // arrives. Refetched when the job reaches a terminal state, when the log is
  // final.
  const logs = useJobLogs(jobId, stream.finished);
  const [actionError, setActionError] = useState<string | null>(null);
  const [acting, setActing] = useState(false);

  const offline = status === "offline";
  const job = stream.job;
  const presentation = job ? jobStatePresentation(job.state) : null;

  // Prefer the live fraction from the stream, fall back to the durable estimate.
  const fraction = stream.liveProgress ?? job?.progress_fraction ?? 0;
  const description =
    stream.liveDescription ??
    job?.live_description ??
    job?.stages.find((stage) => stage.status === "RUNNING")?.name ??
    job?.status ??
    "";

  const terminal = job ? isTerminalState(job.state) : false;
  const canCancel = job !== null && !terminal;

  async function handleResume() {
    if (!jobId) return;
    setActing(true);
    setActionError(null);
    try {
      await resumeJob(jobId);
      stream.refresh();
    } catch (cause) {
      setActionError(
        cause instanceof Error ? cause.message : "Could not resume the job.",
      );
    } finally {
      setActing(false);
    }
  }

  async function handleCancel() {
    if (!jobId) return;
    setActing(true);
    setActionError(null);
    try {
      await cancelJob(jobId);
      // A request, not an outcome: refresh to pick up whatever the server
      // recorded, and the stream will report the terminal event if it comes.
      stream.refresh();
    } catch (cause) {
      setActionError(
        cause instanceof Error ? cause.message : "Could not cancel the job.",
      );
    } finally {
      setActing(false);
    }
  }

  if (!jobId) {
    return (
      <div className="page">
        <ErrorState error={new Error("No job id in the URL.")} />
      </div>
    );
  }

  return (
    <div className="page">
      <header className="page-header">
        <div className="breadcrumb">
          <Link to="/jobs">Jobs</Link>
          <IconArrowRight size={14} />
          <span>{job?.source_name ?? jobId}</span>
        </div>
        <div className="section-heading" style={{ marginBottom: 0 }}>
          <div>
            <h1 className="page-title">{job?.source_name ?? "Job"}</h1>
            <p className="page-subtitle">
              {job
                ? `${displayLanguage(job.source_language, languages.data?.languages ?? [], "Auto")} → ${displayLanguage(job.target_language, languages.data?.languages ?? [])}`
                : jobId}
            </p>
          </div>
          {presentation && (
            <Badge tone={presentation.tone} dot>
              {presentation.label}
            </Badge>
          )}
        </div>
      </header>

      {offline && <OfflineBanner />}

      {actionError && (
        <div className="offline-banner" role="alert">
          <div>
            <div className="offline-banner-title">Action failed</div>
            <p className="offline-banner-body">{actionError}</p>
          </div>
        </div>
      )}

      {stream.error ? (
        <ErrorState error={stream.error} onRetry={stream.refresh} />
      ) : stream.loading && !job ? (
        <SkeletonCard lines={5} />
      ) : job ? (
        <>
          <section className="card">
            <div className="card-head">
              <h2 className="card-title">Progress</h2>
              <span className="stream-status">
                {stream.finished
                  ? "Stream closed"
                  : stream.connected
                    ? "Live"
                    : "Connecting…"}
              </span>
            </div>

            <div className="job-progress-text">
              <span className="job-progress-label" title={description}>
                {description || "Waiting to start"}
              </span>
              <span>{formatPercent(fraction)}</span>
            </div>
            <div className="progress progress-lg">
              <div
                className="progress-bar"
                style={{ width: `${Math.round(fraction * 100)}%` }}
                role="progressbar"
                aria-valuenow={Math.round(fraction * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
              />
            </div>

            {job.queue_position !== null && (
              <p className="field-hint">
                Position {job.queue_position} in the queue.
              </p>
            )}

            {job.segments.total > 0 && (
              <div className="jobs-summary" style={{ marginTop: "var(--space-4)" }}>
                <span className="jobs-summary-item">
                  Segments{" "}
                  <span className="jobs-summary-value">{job.segments.total}</span>
                </span>
                <span className="jobs-summary-item">
                  Complete{" "}
                  <span className="jobs-summary-value">{job.segments.complete}</span>
                </span>
                {job.segments.failed > 0 && (
                  <span className="jobs-summary-item">
                    Failed{" "}
                    <span className="jobs-summary-value">
                      {job.segments.failed}
                    </span>
                  </span>
                )}
              </div>
            )}

            <div className="btn-row" style={{ marginTop: "var(--space-4)" }}>
              {job.output_available && (
                <Link
                  to={`/jobs/${encodeURIComponent(job.job_id)}/result`}
                  className="btn btn-primary"
                >
                  View result
                  <IconArrowRight size={16} />
                </Link>
              )}
              {job.resume.can_resume && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={handleResume}
                  disabled={acting || offline}
                  title={job.resume.message}
                >
                  <IconRefresh size={16} />
                  Resume
                </button>
              )}
              {canCancel && (
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={handleCancel}
                  disabled={acting || offline}
                >
                  <IconStop size={16} />
                  Cancel
                </button>
              )}
            </div>

            {!job.resume.can_resume && job.resume.reason !== "complete" && (
              <p className="field-hint">{job.resume.message}</p>
            )}

            {job.error && (
              <div className="job-error" role="alert">
                <strong>This job failed.</strong> {job.error}
              </div>
            )}
          </section>

          <section className="card">
            <div className="card-head">
              <h2 className="card-title">Stages</h2>
              <span className="card-subtitle">
                {stageProgress(job.stages).done} of {job.stages.length} complete
              </span>
            </div>
            <StageTimeline stages={job.stages} />
          </section>

          <section className="card">
            <LogPanel
              liveLines={stream.logs.map((line) => line.message)}
              durableLines={logs.data?.lines ?? null}
              dropped={stream.droppedLogs}
              truncated={logs.data?.truncated_lines ?? false}
            />
          </section>

          <section className="card">
            <div className="dl">
              <div className="dl-row">
                <span className="dl-label">Job ID</span>
                <span className="dl-value dl-value-mono">{job.job_id}</span>
              </div>
              <div className="dl-row">
                <span className="dl-label">Created</span>
                <span className="dl-value">
                  {formatRelativeTime(job.created_at)}
                </span>
              </div>
              <div className="dl-row">
                <span className="dl-label">Updated</span>
                <span className="dl-value">
                  {formatRelativeTime(job.updated_at)}
                </span>
              </div>
              {job.quality_preset && (
                <div className="dl-row">
                  <span className="dl-label">Quality preset</span>
                  <span className="dl-value">{job.quality_preset}</span>
                </div>
              )}
            </div>
          </section>
        </>
      ) : null}
    </div>
  );
}
