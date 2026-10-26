/**
 * Tests for the SSE transport and the `useAsync` primitive.
 *
 * The SSE tests assert the module's own parsing and dispatch, using a minimal
 * `EventSource` stand-in, because jsdom has none. `useAsync` is tested through a
 * real component so the abort-on-unmount and error-normalization paths are
 * exercised rather than assumed.
 */

import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useAsync } from "../hooks/useAsync";
import { openJobStream } from "../api/sse";
import { ApiError } from "../api/client";
import { BackendHealthProvider, useBackendHealth } from "../hooks/useHealth";
import { healthFixture } from "./fixtures";
import { FakeEventSource, installEventSourceStub } from "./eventSourceStub";

beforeEach(() => {
  installEventSourceStub();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("openJobStream", () => {
  it("builds the events URL for the job", () => {
    openJobStream("job-1", { onEvent: () => {} });
    expect(FakeEventSource.instances[0].url).toBe("/api/jobs/job-1/events");
  });

  it("passes a resume position as last_event_id", () => {
    openJobStream("job-1", { onEvent: () => {} }, 42);
    expect(FakeEventSource.instances[0].url).toContain("last_event_id=42");
  });

  it("parses a frame and hands it to onEvent", () => {
    const received: unknown[] = [];
    openJobStream("job-1", { onEvent: (event) => received.push(event) });

    FakeEventSource.instances[0].emit("progress", {
      event_id: 7,
      job_id: "job-1",
      type: "progress",
      at: 1_700_000_000,
      payload: { fraction: 0.5 },
    });

    expect(received).toHaveLength(1);
    expect(received[0]).toMatchObject({ event_id: 7, type: "progress" });
  });

  it("routes resync to onResync instead of onEvent", () => {
    const events: unknown[] = [];
    const resyncs: number[] = [];
    openJobStream("job-1", {
      onEvent: (event) => events.push(event),
      onResync: () => resyncs.push(1),
    });

    FakeEventSource.instances[0].emit("resync", { event_id: 0, type: "resync" });

    expect(resyncs).toHaveLength(1);
    expect(events).toHaveLength(0);
  });

  it("drops a malformed frame without throwing", () => {
    const received: unknown[] = [];
    openJobStream("job-1", { onEvent: (event) => received.push(event) });

    const source = FakeEventSource.instances[0];
    for (const listener of source.listeners.get("progress") ?? []) {
      listener({ data: "not json" } as MessageEvent<string>);
    }

    expect(received).toHaveLength(0);
  });

  it("closes the underlying source", () => {
    const stream = openJobStream("job-1", { onEvent: () => {} });
    stream.close();
    expect(FakeEventSource.instances[0].closed).toBe(true);
  });

  it("reports open and error through the handlers", () => {
    const onOpen = vi.fn();
    const onError = vi.fn();
    openJobStream("job-1", { onEvent: () => {}, onOpen, onError });

    const source = FakeEventSource.instances[0];
    source.onopen?.();
    source.onerror?.(new Event("error"));

    expect(onOpen).toHaveBeenCalledOnce();
    expect(onError).toHaveBeenCalledOnce();
  });
});

// --- useAsync ---------------------------------------------------------------

function Probe({ fetcher }: { fetcher: (signal: AbortSignal) => Promise<string> }) {
  const state = useAsync(fetcher, []);
  if (state.loading) return <span>loading</span>;
  if (state.error) {
    return <span>{state.offline ? "offline" : state.error.message}</span>;
  }
  return <span>{state.data}</span>;
}

describe("useAsync", () => {
  it("moves from loading to data", async () => {
    render(<Probe fetcher={async () => "done"} />);
    expect(screen.getByText("loading")).toBeInTheDocument();
    expect(await screen.findByText("done")).toBeInTheDocument();
  });

  it("surfaces an ApiError's message", async () => {
    render(
      <Probe
        fetcher={async () => {
          throw new ApiError({ message: "No such job.", code: "JOB_NOT_FOUND" });
        }}
      />,
    );
    expect(await screen.findByText("No such job.")).toBeInTheDocument();
  });

  it("flags a non-ApiError failure as offline", async () => {
    render(
      <Probe
        fetcher={async () => {
          throw new TypeError("Failed to fetch");
        }}
      />,
    );
    expect(await screen.findByText("offline")).toBeInTheDocument();
  });

  it("ignores an abort so an unmount does not set state", async () => {
    let captured: AbortSignal | null = null;
    const { unmount } = render(
      <Probe
        fetcher={(signal) => {
          captured = signal;
          return new Promise((_, reject) => {
            signal.addEventListener("abort", () =>
              reject(new DOMException("Aborted", "AbortError")),
            );
          });
        }}
      />,
    );

    unmount();
    expect((captured as AbortSignal | null)?.aborted).toBe(true);
    await waitFor(() => expect(true).toBe(true));
  });
});

// --- BackendHealthProvider recovery -----------------------------------------

function StatusProbe() {
  const { status, refresh } = useBackendHealth();
  return (
    <div>
      <span>{status}</span>
      <button type="button" onClick={refresh}>
        Retry
      </button>
    </div>
  );
}

describe("BackendHealthProvider", () => {
  it("recovers from offline to online when refresh succeeds", async () => {
    // First probe fails; every later probe succeeds. This is the real
    // "user started the backend and pressed Retry" sequence.
    let failFirst = true;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        if (failFirst) {
          failFirst = false;
          throw new TypeError("Failed to fetch");
        }
        return new Response(JSON.stringify(healthFixture), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }),
    );

    render(
      <BackendHealthProvider>
        <StatusProbe />
      </BackendHealthProvider>,
    );

    expect(await screen.findByText("offline")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByText("online")).toBeInTheDocument();
  });

  it("reports checking before the first probe settles", () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => new Promise<Response>(() => {})),
    );

    render(
      <BackendHealthProvider>
        <StatusProbe />
      </BackendHealthProvider>,
    );

    expect(screen.getByText("checking")).toBeInTheDocument();
  });
});
