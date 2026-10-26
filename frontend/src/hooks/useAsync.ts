/**
 * A small data-fetching primitive shared by every hook.
 *
 * Deliberately not a query library: the app has a handful of read-only
 * endpoints, and the states that matter (loading, offline, error, data) are the
 * ones `ApiError` already distinguishes. Centralizing them here means every page
 * gets the same loading and offline behavior for free, and a page cannot forget
 * to handle the offline case.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";

export interface AsyncState<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  /** True when `error` is an unreachable-backend failure. */
  offline: boolean;
  reload: () => void;
}

/**
 * Run `fetcher` on mount and whenever `deps` change, with a manual `reload`.
 *
 * The in-flight request is aborted on unmount or re-run, so a slow response
 * cannot set state on an unmounted component or overwrite a newer one.
 */
export function useAsync<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: readonly unknown[] = [],
): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    setLoading(true);
    setError(null);

    fetcherRef
      .current(controller.signal)
      .then((result) => {
        if (!active) return;
        setData(result);
        setLoading(false);
      })
      .catch((cause: unknown) => {
        if (!active) return;
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        setError(
          cause instanceof ApiError ? cause : ApiError.offline(cause),
        );
        setLoading(false);
      });

    return () => {
      active = false;
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);

  return {
    data,
    error,
    loading,
    offline: error?.isOffline ?? false,
    reload,
  };
}
