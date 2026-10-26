/**
 * Backend liveness, shared across the app through context.
 *
 * `/api/health` is polled on an interval, but deliberately not aggressively: a
 * liveness probe every few seconds from every open tab is load the server does
 * not need. The poll is also refreshed on the events that actually mean "the
 * user came back" — regaining focus and regaining network — so the indicator is
 * fresh when it is being looked at rather than on a timer nobody is watching.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { getHealth } from "../api/endpoints";
import type { HealthResponse } from "../api/types";
import type { BackendStatus } from "./backendStatus";

/** How often to re-probe while the tab is open. */
const POLL_INTERVAL_MS = 30_000;

export interface BackendHealthState {
  status: BackendStatus;
  health: HealthResponse | null;
  /** Re-probe now; used by the Retry button and on focus/online events. */
  refresh: () => void;
  /** Timestamp (ms) of the last completed probe, for "checked Ns ago". */
  lastChecked: number | null;
}

const BackendHealthContext = createContext<BackendHealthState | null>(null);

export function BackendHealthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<BackendStatus>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [lastChecked, setLastChecked] = useState<number | null>(null);
  const [nonce, setNonce] = useState(0);
  const mounted = useRef(true);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    getHealth(controller.signal)
      .then((result) => {
        if (!mounted.current) return;
        setHealth(result);
        setStatus("online");
        setLastChecked(Date.now());
      })
      .catch((cause: unknown) => {
        if (!mounted.current) return;
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        setStatus("offline");
        setLastChecked(Date.now());
      });

    return () => controller.abort();
  }, [nonce]);

  // Interval poll. Skipped while the tab is hidden, and re-run on the next
  // visibility change, so a backgrounded tab is not a background load.
  useEffect(() => {
    const tick = () => {
      if (document.visibilityState === "visible") refresh();
    };
    const interval = window.setInterval(tick, POLL_INTERVAL_MS);
    const onVisible = () => {
      if (document.visibilityState === "visible") refresh();
    };
    window.addEventListener("visibilitychange", onVisible);
    window.addEventListener("online", refresh);
    window.addEventListener("focus", onVisible);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("online", refresh);
      window.removeEventListener("focus", onVisible);
    };
  }, [refresh]);

  return (
    <BackendHealthContext.Provider
      value={{ status, health, refresh, lastChecked }}
    >
      {children}
    </BackendHealthContext.Provider>
  );
}

export function useBackendHealth(): BackendHealthState {
  const context = useContext(BackendHealthContext);
  if (!context) {
    throw new Error("useBackendHealth must be used within BackendHealthProvider");
  }
  return context;
}
