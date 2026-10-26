/**
 * A job's durable log (`GET /api/jobs/{id}/logs`).
 *
 * Separate from the live stream on purpose: this is the complete, redacted
 * record on disk, correct after a restart, while the stream only carries what
 * happened during this visit. The detail page shows the durable log once it has
 * loaded and falls back to the live lines before that.
 *
 * `enabled` lets the page skip the call for a job with no log yet, rather than
 * fetching an empty file.
 */

import { getJobLogs } from "../api/endpoints";
import type { LogsResponse } from "../api/types";
import { useAsync, type AsyncState } from "./useAsync";

export function useJobLogs(
  jobId: string | undefined,
  enabled = true,
): AsyncState<LogsResponse | null> {
  return useAsync<LogsResponse | null>(
    (signal) =>
      jobId && enabled ? getJobLogs(jobId, { signal }) : Promise.resolve(null),
    [jobId, enabled],
  );
}
