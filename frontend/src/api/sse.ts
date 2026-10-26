/**
 * Server-Sent Events client for a job's progress stream.
 *
 * F4 does not render a live processing view — that is F5 — but the transport is
 * small and the shape is fixed by `src/api/events.py`, so it lives here now so
 * F5 has one obvious place to build on rather than inventing a second one.
 *
 * The backend sends `id:`, `event:` and `data:` frames, where `data` is the
 * `Event.to_dict()` JSON. `EventSource` reconnects on its own and replays
 * `Last-Event-ID`; the only event this module must handle specially is `resync`,
 * which means the requested position has fallen out of the server's ring buffer
 * and the caller should re-read the job rather than trust the stream.
 */

import { API_BASE } from "./client";

export type JobEventType =
  | "snapshot"
  | "queued"
  | "started"
  | "progress"
  | "log"
  | "result"
  | "failed"
  | "cancelled"
  | "resync"
  | "heartbeat";

/**
 * The event types after which no further events arrive for a job.
 *
 * The server closes the stream once one of these is sent. `EventSource` treats
 * any close as a drop and reconnects on its own, so a client that does not close
 * the stream itself would reconnect to a finished job forever; the hook below
 * closes on these.
 */
export const TERMINAL_EVENT_TYPES: readonly JobEventType[] = [
  "result",
  "failed",
  "cancelled",
];

export function isTerminalEvent(type: JobEventType): boolean {
  return TERMINAL_EVENT_TYPES.includes(type);
}

export interface JobEvent {
  event_id: number;
  job_id: string;
  type: JobEventType;
  at: number;
  payload: Record<string, unknown>;
}

export interface JobEventStreamHandlers {
  onEvent: (event: JobEvent) => void;
  /** The stream fell behind the buffer; the caller should re-read the job. */
  onResync?: () => void;
  onError?: (error: Event) => void;
  onOpen?: () => void;
}

export interface JobEventStream {
  close: () => void;
}

/**
 * Subscribe to a job's event stream.
 *
 * `lastEventId` is passed when a caller wants to resume a known position; the
 * browser sends it as `Last-Event-ID` on the initial request.
 */
export function openJobStream(
  jobId: string,
  handlers: JobEventStreamHandlers,
  lastEventId?: number,
): JobEventStream {
  const base = `${API_BASE}/jobs/${encodeURIComponent(jobId)}/events`;
  const url = lastEventId ? `${base}?last_event_id=${lastEventId}` : base;
  const source = new EventSource(url);

  const eventTypes: JobEventType[] = [
    "snapshot",
    "queued",
    "started",
    "progress",
    "log",
    "result",
    "failed",
    "cancelled",
    "resync",
  ];

  for (const type of eventTypes) {
    source.addEventListener(type, (raw) => {
      const message = raw as MessageEvent<string>;
      if (type === "resync") {
        handlers.onResync?.();
        return;
      }
      try {
        handlers.onEvent(JSON.parse(message.data) as JobEvent);
      } catch {
        // A malformed frame is dropped rather than thrown: the stream keeps
        // running and the next `snapshot`/`resync` corrects the view.
      }
    });
  }

  source.onopen = () => handlers.onOpen?.();
  source.onerror = (error) => handlers.onError?.(error);

  return { close: () => source.close() };
}
