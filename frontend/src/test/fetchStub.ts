/**
 * A small fetch double.
 *
 * Tests exercise the real `request()` path in `api/client.ts` — the real
 * envelope parsing, the real error normalization — by stubbing only the network
 * boundary (`globalThis.fetch`). Nothing above the transport is mocked, so a
 * change to how errors are parsed is caught by these tests.
 */

import { vi } from "vitest";

export interface StubRoute {
  /** Match a URL path (the part after the query string). */
  path: string;
  method?: string;
  /** HTTP status, default 200. */
  status?: number;
  /** JSON body. */
  json?: unknown;
  /** Raw text body, for non-JSON responses. */
  text?: string;
  /** Reject instead of responding, to simulate an unreachable server. */
  networkError?: boolean;
}

export interface FetchStub {
  /** Every request the code under test made, in order. */
  calls: Array<{ url: string; method: string; body: unknown }>;
  /** Find the first recorded call matching a path fragment. */
  find: (fragment: string) => { url: string; method: string; body: unknown } | undefined;
}

function pathOf(input: RequestInfo | URL): string {
  const raw =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url;
  const withoutOrigin = raw.replace(/^https?:\/\/[^/]+/, "");
  return withoutOrigin.split("?")[0];
}

function queryOf(input: RequestInfo | URL): string {
  const raw =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url;
  const index = raw.indexOf("?");
  return index === -1 ? "" : raw.slice(index);
}

/**
 * Install a fetch stub that answers the given routes.
 *
 * An unmatched path is a test failure, not a silent 404: a typo in a route or a
 * component calling an endpoint the test did not declare should be loud.
 */
export function installFetchStub(routes: StubRoute[]): FetchStub {
  const calls: FetchStub["calls"] = [];

  const stub = vi.fn(
    async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const method = (init?.method ?? "GET").toUpperCase();
      const path = pathOf(input);
      let parsedBody: unknown = undefined;
      if (typeof init?.body === "string") {
        try {
          parsedBody = JSON.parse(init.body);
        } catch {
          parsedBody = init.body;
        }
      }
      calls.push({ url: `${path}${queryOf(input)}`, method, body: parsedBody });

      const route = routes.find(
        (candidate) =>
          candidate.path === path &&
          (candidate.method ?? "GET").toUpperCase() === method,
      );

      if (!route) {
        throw new Error(
          `Unexpected request: ${method} ${path}. Declared routes: ${routes
            .map((r) => `${r.method ?? "GET"} ${r.path}`)
            .join(", ")}`,
        );
      }

      if (route.networkError) {
        throw new TypeError("Failed to fetch");
      }

      const status = route.status ?? 200;
      const body =
        route.text !== undefined
          ? route.text
          : route.json !== undefined
            ? JSON.stringify(route.json)
            : "";

      return new Response(status === 204 ? null : body, {
        status,
        headers: { "Content-Type": "application/json" },
      });
    },
  );

  vi.stubGlobal("fetch", stub);

  return {
    calls,
    find: (fragment: string) =>
      calls.find((call) => call.url.includes(fragment)),
  };
}

/** Build a JSON error envelope in the backend's shape. */
export function errorEnvelope(
  code: string,
  message: string,
  extra: { field?: string; details?: Record<string, unknown> } = {},
): { error: Record<string, unknown> } {
  return { error: { code, message, ...extra } };
}
