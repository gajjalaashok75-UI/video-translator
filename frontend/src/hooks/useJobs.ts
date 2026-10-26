/**
 * The durable job list.
 *
 * Polled on an interval because job state advances on the server without any
 * client action, and this page has no event stream (that is F5). The interval
 * is slow enough not to be noise and the poll is skipped while the tab is
 * hidden. `reload` gives the page its Refresh button.
 */

import { getJobs } from "../api/endpoints";
import type { JobListResponse } from "../api/types";
import { useAsync, type AsyncState } from "./useAsync";
import { useEffect } from "react";

const POLL_INTERVAL_MS = 15_000;

export function useJobs(): AsyncState<JobListResponse> {
  const state = useAsync<JobListResponse>((signal) => getJobs(signal), []);
  const { reload } = state;

  useEffect(() => {
    const tick = () => {
      if (document.visibilityState === "visible") reload();
    };
    const interval = window.setInterval(tick, POLL_INTERVAL_MS);
    const onVisible = () => {
      if (document.visibilityState === "visible") reload();
    };
    window.addEventListener("visibilitychange", onVisible);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("visibilitychange", onVisible);
    };
  }, [reload]);

  return state;
}
