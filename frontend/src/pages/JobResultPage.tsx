/**
 * A finished job's result: the video, its downloads, and cleanup.
 *
 * Everything is read through the API's own URLs (`media_url` for playback,
 * `download_url` for a named download) — the client is never told a filesystem
 * path. The artifact list is the backend's, including the artifacts this render
 * did not produce, which are shown as unavailable with the backend's reason.
 *
 * Delete is destructive and so is the only action here behind a confirmation.
 * The level chosen is sent verbatim to `DELETE /api/jobs/{id}?level=…`; the
 * backend's `DeleteResponse` says what was actually removed, which is shown
 * rather than assumed.
 */

import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  deleteJob,
  getJob,
  getJobArtifacts,
  getJobOutput,
} from "../api/endpoints";
import { useAsync } from "../hooks/useAsync";
import { useBackendHealth } from "../hooks/useHealth";
import { useLanguages } from "../hooks/useLanguages";
import { Badge } from "../components/common/Badge";
import { EmptyState, ErrorState, SkeletonCard } from "../components/common/States";
import { OfflineBanner } from "../components/status/OfflineBanner";
import { ArtifactList } from "../components/jobs/ArtifactList";
import { IconArrowRight, IconDownload, IconFilm } from "../components/common/Icons";
import { formatBytes, formatRelativeTime } from "../utils/format";
import { displayLanguage } from "../utils/languages";
import { summaryRows } from "../utils/dubbing";
import { jobStatePresentation } from "../utils/status";
import type { CleanupLevel } from "../api/types";

const CLEANUP_LEVELS: Array<{ value: CleanupLevel; label: string; hint: string }> = [
  {
    value: "KEEP_ALL",
    label: "Remove nothing",
    hint: "Unregister the job but leave every file on disk.",
  },
  {
    value: "KEEP_RESUMABLE",
    label: "Keep resumable stages",
    hint: "Remove intermediates that can be regenerated, keep what resume needs.",
  },
  {
    value: "FINAL_ONLY",
    label: "Keep only the final video",
    hint: "Remove everything except the finished output.",
  },
  {
    value: "EVERYTHING",
    label: "Remove everything",
    hint: "Delete the job directory and all its files. Cannot be undone.",
  },
];

export function JobResultPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const { status } = useBackendHealth();
  const languages = useLanguages();
  const job = useAsync(
    (signal) => (jobId ? getJob(jobId, signal) : Promise.resolve(null)),
    [jobId],
  );
  const output = useAsync(
    (signal) => (jobId ? getJobOutput(jobId, signal) : Promise.resolve(null)),
    [jobId],
  );
  const artifacts = useAsync(
    (signal) => (jobId ? getJobArtifacts(jobId, signal) : Promise.resolve(null)),
    [jobId],
  );

  const [deleteOpen, setDeleteOpen] = useState(false);
  const [level, setLevel] = useState<CleanupLevel>("KEEP_RESUMABLE");
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const offline = status === "offline";
  const jobData = job.data;

  async function handleDelete() {
    if (!jobId) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      const result = await deleteJob(jobId, level);
      if (result.deleted) {
        navigate("/jobs");
        return;
      }
      // KEEP_ALL unregisters without removing files; the API reports it, so the
      // page says what happened instead of implying the files are gone.
      setDeleteError(result.message);
    } catch (cause) {
      setDeleteError(
        cause instanceof Error ? cause.message : "Could not delete the job.",
      );
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="page">
      <header className="page-header">
        <div className="breadcrumb">
          <Link to="/jobs">Jobs</Link>
          <IconArrowRight size={14} />
          <Link to={`/jobs/${encodeURIComponent(jobId ?? "")}`}>
            {jobData?.source_name ?? jobId}
          </Link>
          <IconArrowRight size={14} />
          <span>Result</span>
        </div>
        <div className="section-heading" style={{ marginBottom: 0 }}>
          <div>
            <h1 className="page-title">Result</h1>
            {jobData && (
              <p className="page-subtitle">
                {displayLanguage(jobData.source_language, languages.data?.languages ?? [], "Auto")}
                {" → "}
                {displayLanguage(jobData.target_language, languages.data?.languages ?? [])}
              </p>
            )}
          </div>
          {jobData && (
            <Badge tone={jobStatePresentation(jobData.state).tone} dot>
              {jobStatePresentation(jobData.state).label}
            </Badge>
          )}
        </div>
      </header>

      {offline && <OfflineBanner />}

      {job.error ? (
        <ErrorState error={job.error} onRetry={job.reload} />
      ) : job.loading && !jobData ? (
        <SkeletonCard lines={4} />
      ) : jobData ? (
        <>
          <section className="card">
            <div className="card-head">
              <h2 className="card-title">
                <IconFilm size={18} /> Dubbed video
              </h2>
            </div>

            {output.error ? (
              <ErrorState error={output.error} onRetry={output.reload} />
            ) : output.loading ? (
              <SkeletonCard lines={3} />
            ) : output.data?.available && output.data.media_url ? (
              <>
                {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
                <video
                  className="video-player"
                  controls
                  preload="metadata"
                  src={output.data.media_url}
                />
                <div className="btn-row" style={{ marginTop: "var(--space-4)" }}>
                  {output.data.download_url && (
                    <a
                      className="btn btn-primary"
                      href={output.data.download_url}
                      download={output.data.filename ?? undefined}
                    >
                      <IconDownload size={16} />
                      Download
                      {output.data.size_bytes !== null
                        ? ` (${formatBytes(output.data.size_bytes)})`
                        : ""}
                    </a>
                  )}
                  <Link
                    to={`/jobs/${encodeURIComponent(jobData.job_id)}`}
                    className="btn btn-secondary"
                  >
                    Back to job
                  </Link>
                </div>
              </>
            ) : (
              <EmptyState
                title="No video available"
                body="This job did not produce a final video. Open the job to see its stages and log."
                action={
                  <Link
                    to={`/jobs/${encodeURIComponent(jobData.job_id)}`}
                    className="btn btn-secondary"
                  >
                    View job
                  </Link>
                }
              />
            )}
          </section>

          <section className="card">
            <div className="card-head">
              <h2 className="card-title">Downloads</h2>
              {artifacts.data && (
                <span className="card-subtitle">
                  {artifacts.data.available_count} of {artifacts.data.count}{" "}
                  available
                </span>
              )}
            </div>
            {artifacts.error ? (
              <ErrorState error={artifacts.error} onRetry={artifacts.reload} />
            ) : artifacts.loading ? (
              <SkeletonCard lines={3} />
            ) : artifacts.data ? (
              <ArtifactList artifacts={artifacts.data.artifacts} />
            ) : null}
          </section>

          <section className="card">
            <div className="card-head">
              <h2 className="card-title">How this was made</h2>
            </div>
            <div className="dl">
              {summaryRows(jobData.summary).map((row) => (
                <div className="dl-row" key={row.label}>
                  <span className="dl-label">{row.label}</span>
                  <span className="dl-value">{row.value}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="card">
            <div className="dl">
              <div className="dl-row">
                <span className="dl-label">Job ID</span>
                <span className="dl-value dl-value-mono">{jobData.job_id}</span>
              </div>
              <div className="dl-row">
                <span className="dl-label">Finished</span>
                <span className="dl-value">
                  {formatRelativeTime(jobData.finished_at ?? jobData.updated_at)}
                </span>
              </div>
              {jobData.quality_preset && (
                <div className="dl-row">
                  <span className="dl-label">Quality preset</span>
                  <span className="dl-value">{jobData.quality_preset}</span>
                </div>
              )}
            </div>
          </section>

          <section className="card danger-zone">
            <div className="card-head">
              <h2 className="card-title">Delete this job</h2>
            </div>
            <p className="card-subtitle">
              Deleting removes the job from the list. How much of its directory
              is removed is your choice.
            </p>

            {!deleteOpen ? (
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setDeleteOpen(true)}
                disabled={offline}
              >
                Delete job…
              </button>
            ) : (
              <div className="delete-panel">
                <fieldset className="delete-levels">
                  <legend>What to remove</legend>
                  {CLEANUP_LEVELS.map((option) => (
                    <label key={option.value} className="delete-level">
                      <input
                        type="radio"
                        name="cleanup-level"
                        value={option.value}
                        checked={level === option.value}
                        onChange={() => setLevel(option.value)}
                      />
                      <span>
                        <strong>{option.label}</strong>
                        <span className="field-hint">{option.hint}</span>
                      </span>
                    </label>
                  ))}
                </fieldset>

                {deleteError && (
                  <p className="field-error" role="alert">
                    {deleteError}
                  </p>
                )}

                <div className="btn-row">
                  <button
                    type="button"
                    className="btn btn-danger"
                    onClick={handleDelete}
                    disabled={deleting || offline}
                  >
                    {deleting ? "Deleting…" : "Delete job"}
                  </button>
                  <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={() => {
                      setDeleteOpen(false);
                      setDeleteError(null);
                    }}
                    disabled={deleting}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </section>
        </>
      ) : null}
    </div>
  );
}
