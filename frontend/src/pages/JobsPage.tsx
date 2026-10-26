/**
 * Jobs overview (read-only).
 *
 * Lists the durable jobs the backend reports. This is deliberately not the
 * processing page: no event stream, no detail view, no delete. Resume is offered
 * only where the backend says `resume.can_resume`, because it is already a
 * single well-defined call in F3 and refusing to expose it would be less useful
 * than showing it. Delete is left out entirely — it is destructive and belongs
 * with the detail view in F5 where it can be confirmed.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { useJobs } from "../hooks/useJobs";
import { useLanguages } from "../hooks/useLanguages";
import { useBackendHealth } from "../hooks/useHealth";
import { resumeJob } from "../api/endpoints";
import { JobRow } from "../components/jobs/JobRow";
import {
  EmptyState,
  ErrorState,
  SkeletonCard,
} from "../components/common/States";
import { OfflineBanner } from "../components/status/OfflineBanner";
import {
  IconArrowRight,
  IconInbox,
  IconRefresh,
} from "../components/common/Icons";

export function JobsPage() {
  const { status } = useBackendHealth();
  const jobs = useJobs();
  // Only used to normalize the stored route for display; a failure here must
  // not hide the jobs, so no error state is derived from it.
  const languages = useLanguages();
  const [resuming, setResuming] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const offline = status === "offline";

  async function handleResume(jobId: string) {
    setResuming(jobId);
    setActionError(null);
    try {
      await resumeJob(jobId);
      jobs.reload();
    } catch (error) {
      // The backend's message is the useful one ("already running in this
      // process"), so it is shown verbatim rather than replaced.
      setActionError(
        error instanceof Error ? error.message : "Could not resume the job.",
      );
    } finally {
      setResuming(null);
    }
  }

  return (
    <div className="page">
      <header className="page-header">
        <div className="section-heading" style={{ marginBottom: 0 }}>
          <div>
            <h1 className="page-title">Jobs</h1>
            <p className="page-subtitle">
              Every dubbing job this installation has recorded, read from the
              durable manifest.
            </p>
          </div>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={jobs.reload}
            disabled={jobs.loading}
          >
            <IconRefresh size={16} />
            Refresh
          </button>
        </div>
      </header>

      {offline && <OfflineBanner />}

      {actionError && (
        <div className="offline-banner" role="alert">
          <div>
            <div className="offline-banner-title">Could not resume</div>
            <p className="offline-banner-body">{actionError}</p>
          </div>
        </div>
      )}

      {!offline && jobs.data && (
        <div className="jobs-summary">
          <span className="jobs-summary-item">
            Total <span className="jobs-summary-value">{jobs.data.total}</span>
          </span>
          <span className="jobs-summary-item">
            Active <span className="jobs-summary-value">{jobs.data.active}</span>
          </span>
          <span className="jobs-summary-item">
            Pending{" "}
            <span className="jobs-summary-value">{jobs.data.pending}</span>
          </span>
        </div>
      )}

      {offline ? null : jobs.error ? (
        <ErrorState error={jobs.error} onRetry={jobs.reload} />
      ) : jobs.loading || !jobs.data ? (
        <div className="job-list" aria-hidden="true">
          {Array.from({ length: 4 }).map((_, index) => (
            <SkeletonCard key={index} lines={2} />
          ))}
        </div>
      ) : jobs.data.jobs.length === 0 ? (
        <EmptyState
          title="No dubbing jobs yet"
          body="Start the guided demo on a bundled clip, or upload your own video in the dubbing workspace."
          icon={<IconInbox size={28} />}
          action={
            <div className="btn-row" style={{ marginTop: "var(--space-4)" }}>
              <Link to="/demo" className="btn btn-primary">
                Try the demo
                <IconArrowRight size={16} />
              </Link>
              <Link to="/dub" className="btn btn-secondary">
                Start dubbing
              </Link>
            </div>
          }
        />
      ) : (
        <div className="job-list">
          {jobs.data.jobs.map((job) => (
            <JobRow
              key={job.job_id}
              job={job}
              onResume={handleResume}
              resuming={resuming === job.job_id}
              languages={languages.data?.languages ?? []}
            />
          ))}
        </div>
      )}
    </div>
  );
}
