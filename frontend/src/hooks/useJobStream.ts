/**
 * Live job state: the durable snapshot, kept current by the event stream.
 *
 * The split of responsibility mirrors the backend's own. `GET /api/jobs/{id}`
 * is the source of truth for everything durable — stages, segments, whether the
 * output exists, whether the job can be resumed. The SSE stream is an *updater*:
 * it carries a `snapshot` first, then typed progress and log events, then a
 * terminal event. So this hook keeps one `JobStatusResponse` and folds events
 * into it, rather than trying to reconstruct state from events alone.
 *
 * Two things the naive version gets wrong and this one does not:
 *
 *  - **Reconnect storms.** `EventSource` reconnects on its own after the server
 *    closes a finished stream, so a completed job would be re-fetched forever.
 *    The hook closes the stream itself on a terminal event and never reopens.
 *  - **Events after the fact.** A `resync` means the client's position fell out
 *    of the server's ring buffer; the only correct response is to re-read the
 *    durable state, so the hook does that and keeps the stream open.
 *
 * Live-only fields (`live_progress`, `queue_position`) are absent for a job this
 * server process is not running, which is not an error — the durable fields are
 * still complete, and the UI reads both.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";
import { getJob } from "../api/endpoints";
import {
  isTerminalEvent,
  openJobStream,
  type JobEvent,
  type JobEventStream,
} from "../api/sse";
import type { JobStatusResponse } from "../api/types";

/** A log line as the UI shows it: the message plus when it arrived. */
export interface StreamLogLine {
  id: number;
  message: string;
  at: number;
}

export interface JobStreamState {
  job: JobStatusResponse | null;
  /** Live fraction from the stream, preferred over the durable estimate. */
  liveProgress: number | null;
  liveDescription: string | null;
  /** Recent log lines, oldest first, bounded. */
  logs: StreamLogLine[];
  /** True while the initial durable fetch is in flight. */
  loading: boolean;
  /** Set when the durable fetch failed; the stream is not opened until it works. */
  error: ApiError | null;
  /** True when the event stream is connected. */
  connected: boolean;
  /** True once a terminal event arrived and the stream was closed. */
  finished: boolean;
  /** Re-read the durable job state now (used after resume/cancel). */
  refresh: () => void;
  /** Total log lines dropped from the in-memory tail because of the bound. */
  droppedLogs: number;
}

const MAX_LOG_LINES = 500;

export function useJobStream(jobId: string | undefined): JobStreamState {
  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [liveProgress, setLiveProgress] = useState<number | null>(null);
  const [liveDescription, setLiveDescription] = useState<string | null>(null);
  const [logs, setLogs] = useState<StreamLogLine[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);
  const [connected, setConnected] = useState(false);
  const [finished, setFinished] = useState(false);
  // Every log event ever folded in. `droppedLogs` is derived from this and the
  // retained tail, so the updater below stays pure (React may invoke a state
  // updater twice in StrictMode, and a side effect inside one would double).
  const [receivedLogs, setReceivedLogs] = useState(0);
  const [nonce, setNonce] = useState(0);

  // The stream, kept in a ref so `refresh` and cleanup share one instance.
  const streamRef = useRef<JobEventStream | null>(null);
  // The last event id seen, so a reconnect resumes rather than replays.
  const lastEventIdRef = useRef(0);
  // Dedupe: a reconnect can replay the last frame, and the client keys on id.
  const seenEventIdsRef = useRef<Set<number>>(new Set());
  // Whether the stream effect has an open stream, so a durable-only refetch
  // does not open a second one.
  const openedRef = useRef(false);

  /**
   * Re-read the durable state *without* touching the stream.
   *
   * This is what `resync` and a terminal event call: the fix for a gap is a
   * fresh read, and reconnecting on resync would risk the same gap again. A
   * failed refetch leaves the current state alone rather than blanking the page.
   */
  const refetchDurable = useCallback(async () => {
    if (!jobId) return;
    try {
      const status = await getJob(jobId);
      setJob(status);
    } catch {
      // Best-effort: the stream may still deliver, and the page already has a
      // renderable state.
    }
  }, [jobId]);

  /** Full refresh: re-read durable state *and* reconnect (used after resume). */
  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  // Fetch the durable state, then open the stream. Sequential rather than
  // parallel: the stream's first frame is also a snapshot, but the durable fetch
  // is what makes the page renderable if the stream never connects at all (a job
  // from a previous process, for instance).
  useEffect(() => {
    if (!jobId) {
      setLoading(false);
      setError(null);
      return;
    }

    const controller = new AbortController();
    let active = true;

    setLoading(true);
    setError(null);
    openedRef.current = false;

    getJob(jobId, controller.signal)
      .then((status) => {
        if (!active) return;
        setJob(status);
        setLoading(false);
        // A job already in a terminal state needs no stream; the durable state
        // is the whole answer.
        if (isTerminalState(status.state)) {
          setFinished(true);
          return;
        }
        openStream(status);
      })
      .catch((cause: unknown) => {
        if (!active) return;
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        setError(cause instanceof ApiError ? cause : ApiError.offline(cause));
        setLoading(false);
      });

    function openStream(initial: JobStatusResponse) {
      if (openedRef.current) return;
      openedRef.current = true;
      streamRef.current?.close();
      const stream = openJobStream(
        initial.job_id,
        {
          onEvent: (event) => handleEvent(event),
          onResync: () => {
            // Position fell out of the buffer: re-read durable state, but keep
            // the stream — reconnecting could hit the same gap.
            refetchDurable();
          },
          onOpen: () => active && setConnected(true),
          onError: () => active && setConnected(false),
        },
        lastEventIdRef.current || undefined,
      );
      streamRef.current = stream;
    }

    function handleEvent(event: JobEvent) {
      if (!active) return;
      // `heartbeat` is a comment frame and never reaches here; a `snapshot` has
      // id 0 and is handled below.
      if (event.event_id > 0) {
        if (seenEventIdsRef.current.has(event.event_id)) return;
        seenEventIdsRef.current.add(event.event_id);
        lastEventIdRef.current = event.event_id;
      }

      switch (event.type) {
        case "snapshot": {
          // The authoritative durable state, as a full response model.
          setJob(event.payload as unknown as JobStatusResponse);
          break;
        }
        case "progress": {
          const fraction = event.payload.fraction;
          const description = event.payload.description;
          if (typeof fraction === "number") setLiveProgress(fraction);
          if (typeof description === "string") setLiveDescription(description);
          break;
        }
        case "log": {
          const message = event.payload.message;
          if (typeof message === "string") {
            appendLog(event.event_id, message, event.at);
          }
          break;
        }
        case "queued":
        case "started":
        case "result":
        case "failed":
        case "cancelled": {
          const message =
            typeof event.payload.message === "string"
              ? event.payload.message
              : null;
          if (message) appendLog(event.event_id, message, event.at);
          if (event.type === "result" && typeof event.payload.fraction !== "number") {
            setLiveProgress(1);
          }
          if (isTerminalEvent(event.type)) {
            // Close before refetching: the server closes too, and without this
            // the browser would reconnect to a finished job indefinitely.
            streamRef.current?.close();
            streamRef.current = null;
            openedRef.current = false;
            setConnected(false);
            setFinished(true);
            // Durable fields (output, resume, stages) come from a fresh read.
            refetchDurable();
          } else {
            // A non-terminal state change (queued→started) is durable too.
            refetchDurable();
          }
          break;
        }
        default:
          break;
      }
    }

    function appendLog(id: number, message: string, at: number) {
      setReceivedLogs((count) => count + 1);
      setLogs((previous) => {
        const next = [...previous, { id, message, at }];
        return next.length <= MAX_LOG_LINES
          ? next
          : next.slice(next.length - MAX_LOG_LINES);
      });
    }

    return () => {
      active = false;
      controller.abort();
      streamRef.current?.close();
      streamRef.current = null;
      setConnected(false);
    };
    // `nonce` re-runs the whole effect for `refresh`, which is what a resume or
    // cancel needs: a fresh durable read and a fresh stream.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, nonce, refetchDurable]);

  return {
    job,
    liveProgress,
    liveDescription,
    logs,
    loading,
    error,
    connected,
    finished,
    refresh,
    droppedLogs: Math.max(0, receivedLogs - logs.length),
  };
}

/**
 * True for a state after which a job never changes again.
 *
 * `interrupted` is deliberately *not* terminal: an interrupted job can be
 * resumed, so the page keeps its stream (or offers resume) rather than treating
 * it as done. Only the three outcomes the server records as final are final.
 */
export function isTerminalState(state: JobStatusResponse["state"]): boolean {
  return state === "completed" || state === "failed" || state === "cancelled";
}
