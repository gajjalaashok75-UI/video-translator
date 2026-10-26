/**
 * Live readiness for the render the form would start (`POST /api/jobs/preflight`).
 *
 * Unlike the other hooks this one takes a request body that changes on nearly
 * every keystroke, so it debounces and aborts in flight: the answer for an old
 * set of settings must never overwrite the answer for the current one. It is
 * deliberately *not* built on `useAsync`, whose dependency comparison is by
 * reference and would refire on every render that rebuilt the object.
 *
 * The result is advisory. `createJob` re-validates everything server-side; this
 * exists to explain a disabled Start button before the user hits it, so a
 * preflight *failure* is reported as "cannot check right now" and never blocks
 * the render. Only a settled `ready: false` is allowed to disable Start, and
 * even then `createJob` would have refused the same request.
 */

import { useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";
import { preflightJob } from "../api/endpoints";
import type { JobCreateRequest, PreflightResponse } from "../api/types";

const DEBOUNCE_MS = 350;

export interface PreflightState {
  data: PreflightResponse | null;
  loading: boolean;
  error: ApiError | null;
}

export function usePreflight(
  request: JobCreateRequest | null,
  debounceMs: number = DEBOUNCE_MS,
): PreflightState {
  const [data, setData] = useState<PreflightResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const serialized = request ? JSON.stringify(request) : null;
  const previous = useRef<string | null>(null);

  useEffect(() => {
    if (!serialized) {
      // No request yet (nothing staged): clear rather than report a stale
      // verdict for settings that no longer describe anything.
      setData(null);
      setError(null);
      setLoading(false);
      previous.current = null;
      return;
    }
    if (serialized === previous.current) return;
    previous.current = serialized;

    const controller = new AbortController();
    let active = true;
    const timer = window.setTimeout(() => {
      setLoading(true);
      preflightJob(JSON.parse(serialized) as JobCreateRequest, controller.signal)
        .then((result) => {
          if (!active) return;
          setData(result);
          setError(null);
          setLoading(false);
        })
        .catch((cause: unknown) => {
          if (!active) return;
          if (cause instanceof DOMException && cause.name === "AbortError") return;
          setError(cause instanceof ApiError ? cause : ApiError.offline(cause));
          setLoading(false);
        });
    }, debounceMs);

    return () => {
      active = false;
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [serialized, debounceMs]);

  return { data, loading, error };
}
