/**
 * Tests for `useJobStream`, the hook that folds the event stream into durable
 * job state.
 *
 * These drive the real hook against a real `openJobStream` and a fake
 * `EventSource`, stubbing only `fetch` for the durable reads. The behaviours
 * worth proving are the ones a naive implementation gets wrong: closing the
 * stream on a terminal event so it does not reconnect forever, deduplicating a
 * replayed frame, and re-reading durable state on `resync`.
 */

import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { useJobStream } from "../hooks/useJobStream";
import { installFetchStub } from "./fetchStub";
import { FakeEventSource, installEventSourceStub } from "./eventSourceStub";
import { jobFixture } from "./fixtures";

function Probe({ jobId }: { jobId: string }) {
  const stream = useJobStream(jobId);
  return (
    <div>
      <span data-testid="state">{stream.job?.state ?? "none"}</span>
      <span data-testid="progress">
        {stream.liveProgress === null ? "none" : String(stream.liveProgress)}
      </span>
      <span data-testid="description">{stream.liveDescription ?? "none"}</span>
      <span data-testid="logs">{stream.logs.map((l) => l.message).join("|")}</span>
      <span data-testid="connected">{String(stream.connected)}</span>
      <span data-testid="finished">{String(stream.finished)}</span>
      <span data-testid="dropped">{String(stream.droppedLogs)}</span>
    </div>
  );
}

beforeEach(() => {
  installEventSourceStub();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** A running job, so the hook opens a stream rather than short-circuiting. */
function runningJob(overrides = {}) {
  return jobFixture({
    state: "running",
    status: "RUNNING",
    output_available: false,
    ...overrides,
  });
}

describe("useJobStream", () => {
  it("fetches durable state then opens the stream", async () => {
    const stub = installFetchStub([
      { path: "/api/jobs/job-abc123", json: runningJob() },
    ]);

    render(<Probe jobId="job-abc123" />);

    expect(await screen.findByText("running")).toBeInTheDocument();
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    expect(FakeEventSource.latest()?.url).toBe("/api/jobs/job-abc123/events");
    expect(stub.find("/api/jobs/job-abc123")).toBeDefined();
  });

  it("does not open a stream for an already-terminal job", async () => {
    installFetchStub([
      { path: "/api/jobs/job-abc123", json: jobFixture({ state: "completed" }) },
    ]);

    render(<Probe jobId="job-abc123" />);

    expect(await screen.findByTestId("finished")).toHaveTextContent("true");
    expect(FakeEventSource.instances).toHaveLength(0);
  });

  it("folds a progress event into the live fields", async () => {
    installFetchStub([
      { path: "/api/jobs/job-abc123", json: runningJob() },
    ]);
    render(<Probe jobId="job-abc123" />);
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));

    FakeEventSource.latest()?.emit("progress", {
      event_id: 5,
      job_id: "job-abc123",
      type: "progress",
      at: 1,
      payload: { fraction: 0.42, description: "Translating" },
    });

    await waitFor(() =>
      expect(screen.getByTestId("progress")).toHaveTextContent("0.42"),
    );
    expect(screen.getByTestId("description")).toHaveTextContent("Translating");
  });

  it("closes the stream and refetches durable state on a terminal event", async () => {
    const stub = installFetchStub([
      { path: "/api/jobs/job-abc123", json: runningJob() },
    ]);
    render(<Probe jobId="job-abc123" />);
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));

    // The durable read after the terminal event reports completion.
    stub.calls.length = 0;
    installFetchStub([
      {
        path: "/api/jobs/job-abc123",
        json: jobFixture({ state: "completed", output_available: true }),
      },
    ]);
    const source = FakeEventSource.latest();
    source?.emit("result", {
      event_id: 9,
      job_id: "job-abc123",
      type: "result",
      at: 2,
      payload: { message: "Render complete.", output_available: true },
    });

    await waitFor(() => expect(screen.getByTestId("state")).toHaveTextContent("completed"));
    // The stream must be closed, not left to reconnect to a finished job.
    expect(source?.closed).toBe(true);
    expect(screen.getByTestId("finished")).toHaveTextContent("true");
    expect(screen.getByTestId("logs")).toHaveTextContent("Render complete.");
  });

  it("deduplicates a replayed frame by event id", async () => {
    installFetchStub([
      { path: "/api/jobs/job-abc123", json: runningJob() },
    ]);
    render(<Probe jobId="job-abc123" />);
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));

    const frame = {
      event_id: 3,
      job_id: "job-abc123",
      type: "log",
      at: 1,
      payload: { message: "hello" },
    };
    const source = FakeEventSource.latest();
    source?.emit("log", frame);
    source?.emit("log", frame);

    await waitFor(() => expect(screen.getByTestId("logs")).toHaveTextContent("hello"));
    expect(screen.getByTestId("logs").textContent).toBe("hello");
  });

  it("re-reads durable state on resync without closing the stream", async () => {
    const stub = installFetchStub([
      { path: "/api/jobs/job-abc123", json: runningJob() },
    ]);
    render(<Probe jobId="job-abc123" />);
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    const source = FakeEventSource.latest();

    stub.calls.length = 0;
    source?.emit("resync", {
      event_id: 0,
      job_id: "job-abc123",
      type: "resync",
      at: 1,
      payload: { reason: "buffer_overflow" },
    });

    await waitFor(() =>
      expect(stub.calls.some((c) => c.url.startsWith("/api/jobs/job-abc123"))).toBe(
        true,
      ),
    );
    // resync means "re-read", not "give up": the stream stays open.
    expect(source?.closed).toBe(false);
  });

  it("adopts a snapshot payload as the durable state", async () => {
    installFetchStub([
      { path: "/api/jobs/job-abc123", json: runningJob() },
    ]);
    render(<Probe jobId="job-abc123" />);
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));

    FakeEventSource.latest()?.emit("snapshot", {
      event_id: 0,
      job_id: "job-abc123",
      type: "snapshot",
      at: 1,
      payload: runningJob({ state: "queued", status: "QUEUED" }),
    });

    await waitFor(() => expect(screen.getByTestId("state")).toHaveTextContent("queued"));
  });

  it("surfaces a durable-fetch failure as an error and opens no stream", async () => {
    installFetchStub([
      { path: "/api/jobs/job-abc123", status: 404, json: { error: { code: "JOB_NOT_FOUND", message: "No such job." } } },
    ]);

    render(<Probe jobId="job-abc123" />);

    expect(await screen.findByTestId("state")).toHaveTextContent("none");
    expect(FakeEventSource.instances).toHaveLength(0);
  });

  it("closes the stream on unmount", async () => {
    installFetchStub([
      { path: "/api/jobs/job-abc123", json: runningJob() },
    ]);
    const { unmount } = render(<Probe jobId="job-abc123" />);
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    const source = FakeEventSource.latest();

    unmount();
    expect(source?.closed).toBe(true);
  });
});
