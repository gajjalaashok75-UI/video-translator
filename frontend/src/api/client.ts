/**
 * The single HTTP client for the backend.
 *
 * Every request goes through `request()`, which is the one place that decides
 * what a failure means. Callers therefore never see a raw `fetch` rejection, a
 * non-JSON body or an HTTP status: they either get parsed data or an `ApiError`
 * carrying the backend's own code. That is what lets a component say "backend
 * offline" instead of "Unexpected token < in JSON".
 */

import type { ApiErrorBody, ApiErrorCode, ApiErrorEnvelope } from "./types";

/**
 * Relative by design. In development Vite proxies `/api` to FastAPI; in
 * production the same path is served by FastAPI itself. Nothing in the app
 * hard-codes a host, so nothing needs to change when the two are co-located.
 */
export const API_BASE = "/api";

/**
 * A normalized failure. `isOffline` is the distinction the UI needs most: a
 * request that never reached the server is a different situation (and a
 * different remedy) from one the server rejected.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly code: ApiErrorCode;
  readonly field: string | null;
  readonly details: Record<string, unknown> | null;
  readonly isOffline: boolean;

  constructor(init: {
    message: string;
    status?: number;
    code?: ApiErrorCode;
    field?: string | null;
    details?: Record<string, unknown> | null;
    isOffline?: boolean;
  }) {
    super(init.message);
    this.name = "ApiError";
    this.status = init.status ?? 0;
    this.code = init.code ?? "NETWORK_ERROR";
    this.field = init.field ?? null;
    this.details = init.details ?? null;
    this.isOffline = init.isOffline ?? false;
  }

  /** The failure was "the server could not be reached", not "it said no". */
  static offline(cause?: unknown): ApiError {
    return new ApiError({
      message:
        cause instanceof Error && cause.message
          ? cause.message
          : "Could not reach the backend.",
      isOffline: true,
      code: "NETWORK_ERROR",
    });
  }
}

/** Narrow an unknown error body to the F3 envelope without trusting it. */
function parseEnvelope(body: unknown): ApiErrorBody | null {
  if (typeof body !== "object" || body === null) return null;
  const error = (body as ApiErrorEnvelope).error;
  if (typeof error !== "object" || error === null) return null;
  if (typeof error.code !== "string" || typeof error.message !== "string") {
    return null;
  }
  return error;
}

/**
 * Turn a non-2xx status and its parsed body into an `ApiError`.
 *
 * Shared with the upload path in `endpoints.ts`, which reads its response from
 * `XMLHttpRequest` rather than `fetch` but must report failures identically —
 * one place decides what the envelope means, so an upload rejection and a job
 * rejection surface the same way.
 */
export function errorFromResponse(status: number, parsed: unknown): ApiError {
  const envelope = parseEnvelope(parsed);
  if (envelope) {
    return new ApiError({
      message: envelope.message,
      code: envelope.code as ApiErrorCode,
      status,
      field: envelope.field ?? null,
      details: envelope.details ?? null,
    });
  }
  // A non-envelope error (a proxy 502, an HTML error page) still becomes an
  // ApiError, so no caller has to special-case it.
  return new ApiError({
    message: `Request failed with status ${status}.`,
    status,
    code: "INTERNAL_ERROR",
  });
}

export interface RequestOptions {
  method?: string;
  /** Serialized as JSON when present. */
  body?: unknown;
  signal?: AbortSignal;
  /** Query parameters; `undefined`/`null` values are dropped. */
  query?: Record<string, string | number | boolean | undefined | null>;
  headers?: Record<string, string>;
}

function buildUrl(path: string, query?: RequestOptions["query"]): string {
  const base = `${API_BASE}${path}`;
  if (!query) return base;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null) continue;
    params.append(key, String(value));
  }
  const qs = params.toString();
  return qs ? `${base}?${qs}` : base;
}

/**
 * Perform one request and normalize every outcome into data or `ApiError`.
 *
 * `accept: application/json` is sent on the discovery endpoints; file routes are
 * consumed through `urlFor()` instead, because a browser plays and downloads
 * those directly rather than through `fetch`.
 */
export async function request<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { method = "GET", body, signal, query, headers } = options;

  let response: Response;
  try {
    response = await fetch(buildUrl(path, query), {
      method,
      signal,
      headers: {
        Accept: "application/json",
        ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
        ...headers,
      },
      ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
    });
  } catch (cause) {
    // An aborted request is the caller's own doing, not a backend outage.
    if (cause instanceof DOMException && cause.name === "AbortError") {
      throw cause;
    }
    throw ApiError.offline(cause);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  const text = await response.text();
  let parsed: unknown = null;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = null;
    }
  }

  if (!response.ok) {
    throw errorFromResponse(response.status, parsed);
  }

  return parsed as T;
}

/**
 * Absolute-or-relative URL for a browser-consumed file route (video, download).
 * Returned as-is because it is already relative to the API base.
 */
export function urlFor(path: string): string {
  return path.startsWith("/") ? `${API_BASE}${path}` : path;
}

/** True for failures caused by the server being unreachable. */
export function isOfflineError(error: unknown): boolean {
  return error instanceof ApiError && error.isOffline;
}
