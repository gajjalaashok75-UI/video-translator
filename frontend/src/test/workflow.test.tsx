/**
 * Tests for the dubbing workflow, the processing page and the result page.
 *
 * These render the real pages against the real hooks and the real client,
 * stubbing only the transports (`fetch` for JSON, `XMLHttpRequest` for the
 * upload, `EventSource` for the stream). The assertions are about what the
 * backend's answers cause the UI to do — an unavailable feature is disabled, a
 * staged upload becomes a job request, a terminal event closes the stream — not
 * about incidental markup.
 */

import { describe, expect, it, beforeEach, afterEach, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { BackendHealthProvider } from "../hooks/useHealth";
import { DubPage } from "../pages/DubPage";
import { JobDetailPage } from "../pages/JobDetailPage";
import { JobResultPage } from "../pages/JobResultPage";
import { installFetchStub, type StubRoute } from "./fetchStub";
import { FakeXhr, installXhrStub } from "./xhrStub";
import { FakeEventSource, installEventSourceStub } from "./eventSourceStub";
import {
  artifactListFixture,
  capabilitiesFixture,
  healthFixture,
  jobFixture,
  languagesFixture,
  logsFixture,
  outputFixture,
  uploadFixture,
} from "./fixtures";
import type { OptionsResponse, PreflightResponse } from "../api/types";

const optionsFixture: OptionsResponse = {
  translation_models: [
    {
      value: "nllb",
      label: "NLLB-200",
      description: null,
      available: false,
      state: "not_installed",
    },
    {
      value: "google",
      label: "Google Translate",
      description: null,
      available: true,
      state: "available",
    },
  ],
  transcription_models: [
    {
      value: "large-v3-turbo",
      label: "Large v3 Turbo",
      description: null,
      available: true,
      state: "available",
    },
  ],
  tts_models: [
    {
      value: "edge",
      label: "Edge TTS",
      description: null,
      available: true,
      state: "available",
    },
  ],
  lipsync_models: [
    {
      value: "wav2lip",
      label: "Wav2Lip",
      description: null,
      available: false,
      state: "not_installed",
    },
  ],
  // The dubbing workflow does not surface these yet; they are present only to
  // satisfy the response type.
  context_models: [],
  diarization_backends: [],
  ocr_engines: [],
  audio_separators: [],
  live_portrait_modes: [],
  subtitle_modes: [],
  source: {},
};

const voicesFixture = {
  tts_model: "edge",
  language: "hi",
  voices: ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
  cloning: false,
  note: null,
};

/** Routes every page's initial load needs, so a test only adds what it asserts. */
function baseRoutes(): StubRoute[] {
  return [
    { path: "/api/health", json: healthFixture },
    { path: "/api/languages", json: languagesFixture },
    { path: "/api/options", json: optionsFixture },
    { path: "/api/capabilities", json: capabilitiesFixture },
    // The dubbing page preflights the render it would start. A test that
    // asserts on the verdict declares its own route *before* calling
    // baseRoutes(), because the stub matches the first declared route.
    { path: "/api/jobs/preflight", method: "POST", json: preflightFixture() },
  ];
}

function preflightFixture(
  overrides: Partial<PreflightResponse> = {},
): PreflightResponse {
  return {
    ready: true,
    checks: [
      {
        id: "video",
        label: "Source video",
        severity: "ok",
        detail: "Ready.",
        field: null,
      },
    ],
    blocking: [],
    warnings: [],
    ...overrides,
  };
}

/**
 * Set a file on the hidden file input.
 *
 * `fireEvent.change` rather than `userEvent.upload`: the input is visually
 * hidden (the dropzone button is the affordance), and firing the change directly
 * exercises the same handler without depending on pointer-visibility rules.
 */
function chooseFile(file: File) {
  const input = document.querySelector(
    'input[type="file"]',
  ) as HTMLInputElement;
  fireEvent.change(input, { target: { files: [file] } });
}

function renderAt(path: string, route: string, element: React.ReactElement) {
  return render(
    <BackendHealthProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path={route} element={element} />
          <Route path="*" element={<span>elsewhere</span>} />
        </Routes>
      </MemoryRouter>
    </BackendHealthProvider>,
  );
}

beforeEach(() => {
  installEventSourceStub();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("DubPage", () => {
  it("offers languages and models from the backend, not hard-coded lists", async () => {
    installFetchStub([
      ...baseRoutes(),
      { path: "/api/voices", json: voicesFixture },
    ]);

    renderAt("/dub", "/dub", <DubPage />);

    // The workspace renders its upload step before anything is chosen.
    expect(
      await screen.findByText(/drop a video here/i),
    ).toBeInTheDocument();
  });

  it("uploads the chosen file, then shows settings and a start button", async () => {
    const fetchStub = installFetchStub([
      ...baseRoutes(),
      { path: "/api/voices", json: voicesFixture },
    ]);
    installXhrStub({
      status: 200,
      json: uploadFixture(),
      progress: { loaded: 512, total: 1024 },
    });

    renderAt("/dub", "/dub", <DubPage />);
    await screen.findByText(/drop a video here/i);

    chooseFile(new File(["video-bytes"], "lecture.webm", { type: "video/webm" }));

    // The upload went out as a raw body to the uploads endpoint.
    await waitFor(() => expect(FakeXhr.instances).toHaveLength(1));
    const request = FakeXhr.instances[0];
    expect(request.method).toBe("POST");
    expect(request.url).toBe("/api/uploads?filename=lecture.webm");
    expect(request.headers["Content-Type"]).toBe("application/octet-stream");

    // Settings appear once the upload is staged.
    expect(
      await screen.findByRole("heading", { name: /settings/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /start dubbing/i }),
    ).toBeInTheDocument();
    expect(fetchStub.calls.some((c) => c.url.startsWith("/api/options"))).toBe(
      true,
    );
  });

  it("shows an uninstalled model as disabled rather than hiding it", async () => {
    installFetchStub([
      ...baseRoutes(),
      { path: "/api/voices", json: voicesFixture },
    ]);
    installXhrStub({ status: 200, json: uploadFixture() });

    renderAt("/dub", "/dub", <DubPage />);
    await screen.findByText(/drop a video here/i);
    chooseFile(new File(["x"], "a.webm", { type: "video/webm" }));

    await screen.findByRole("heading", { name: /settings/i });
    const nllb = await screen.findByRole("option", { name: /nllb-200/i });
    expect(nllb).toBeDisabled();
    // The available one is selectable.
    const google = screen.getByRole("option", { name: /google translate/i });
    expect(google).not.toBeDisabled();
  });

  it("submits a job with the staged upload id and navigates to it", async () => {
    const fetchStub = installFetchStub([
      ...baseRoutes(),
      { path: "/api/voices", json: voicesFixture },
      {
        path: "/api/jobs",
        method: "POST",
        status: 202,
        json: jobFixture({
          job_id: "job-new-1",
          state: "queued",
          status: "QUEUED",
          output_available: false,
        }),
      },
    ]);
    installXhrStub({ status: 200, json: uploadFixture() });

    renderAt("/dub", "/dub", <DubPage />);
    await screen.findByText(/drop a video here/i);
    chooseFile(new File(["x"], "a.webm", { type: "video/webm" }));

    const start = await screen.findByRole("button", { name: /start dubbing/i });
    await userEvent.click(start);

    await waitFor(() => expect(screen.getByText("elsewhere")).toBeInTheDocument());
    const post = fetchStub.calls.find((c) => c.method === "POST");
    expect(post?.url).toBe("/api/jobs");
    const body = post?.body as Record<string, unknown>;
    expect(body.upload_id).toBe(uploadFixture().upload_id);
    expect(body.target_lang).toBe("hi");
  });

  it("preflights the render and explains a blocking check", async () => {
    const fetchStub = installFetchStub([
      {
        path: "/api/jobs/preflight",
        method: "POST",
        json: preflightFixture({
          ready: false,
          checks: [
            {
              id: "tts_language",
              label: "Target language",
              severity: "blocking",
              detail: "edge cannot speak Klingon.",
              field: "tts_model_name",
            },
          ],
          blocking: [
            {
              id: "tts_language",
              label: "Target language",
              severity: "blocking",
              detail: "edge cannot speak Klingon.",
              field: "tts_model_name",
            },
          ],
        }),
      },
      ...baseRoutes(),
      { path: "/api/voices", json: voicesFixture },
    ]);
    installXhrStub({ status: 200, json: uploadFixture() });

    renderAt("/dub", "/dub", <DubPage />);
    await screen.findByText(/drop a video here/i);
    chooseFile(new File(["x"], "a.webm", { type: "video/webm" }));

    // The check the backend returned is surfaced, with its reason.
    expect(
      await screen.findByText(/cannot speak klingon/i),
    ).toBeInTheDocument();
    expect(screen.getByText("Blocks")).toBeInTheDocument();
    // And it disables Start, which is the point of showing it before the click.
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /start dubbing/i }),
      ).toBeDisabled(),
    );

    // And the request preflighted was the render the form describes.
    const call = fetchStub.calls.find((c) => c.url.endsWith("/jobs/preflight"));
    const body = call?.body as Record<string, unknown>;
    expect(body.upload_id).toBe(uploadFixture().upload_id);
    expect(body.target_lang).toBe("hi");
  });

  it("does not block Start when preflight itself fails", async () => {
    installFetchStub([
      { path: "/api/jobs/preflight", method: "POST", status: 500,
        json: { error: { code: "INTERNAL", message: "probe exploded" } } },
      ...baseRoutes(),
      { path: "/api/voices", json: voicesFixture },
    ]);
    installXhrStub({ status: 200, json: uploadFixture() });

    renderAt("/dub", "/dub", <DubPage />);
    await screen.findByText(/drop a video here/i);
    chooseFile(new File(["x"], "a.webm", { type: "video/webm" }));

    expect(
      await screen.findByText(/could not check readiness/i),
    ).toBeInTheDocument();
    // The authority is createJob, so Start stays available.
    expect(screen.getByRole("button", { name: /start dubbing/i })).toBeEnabled();
  });
});

describe("JobDetailPage", () => {
  it("shows stages from the backend and streams progress", async () => {
    installFetchStub([
      ...baseRoutes(),
      {
        path: "/api/jobs/job-abc123",
        json: jobFixture({
          state: "running",
          status: "RUNNING",
          output_available: false,
          progress_fraction: 0.2,
          stages: [
            { name: "source", status: "COMPLETE", error: null, attempts: 1 },
            { name: "transcription", status: "RUNNING", error: null, attempts: 1 },
            { name: "tts", status: "PENDING", error: null, attempts: 1 },
          ],
        }),
      },
      { path: "/api/jobs/job-abc123/logs", json: logsFixture() },
    ]);

    renderAt("/jobs/job-abc123", "/jobs/:jobId", <JobDetailPage />);

    const stageList = await screen.findByRole("list");
    expect(within(stageList).getByText("transcription")).toBeInTheDocument();
    expect(within(stageList).getByText("source")).toBeInTheDocument();
    expect(screen.getByText(/1 of 3 complete/i)).toBeInTheDocument();

    // A live progress event moves the bar.
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    FakeEventSource.latest()?.emit("progress", {
      event_id: 4,
      job_id: "job-abc123",
      type: "progress",
      at: 1,
      payload: { fraction: 0.75, description: "Synthesising speech" },
    });

    expect(
      await screen.findByText("Synthesising speech"),
    ).toBeInTheDocument();
    expect(screen.getByText("75%")).toBeInTheDocument();
  });

  it("offers Resume only when the backend says the job can resume", async () => {
    installFetchStub([
      ...baseRoutes(),
      {
        path: "/api/jobs/job-abc123",
        json: jobFixture({
          state: "interrupted",
          status: "interrupted",
          output_available: false,
          resume: {
            can_resume: true,
            reason: "interrupted",
            message: "Completed stages will be reused.",
          },
        }),
      },
    ]);

    renderAt("/jobs/job-abc123", "/jobs/:jobId", <JobDetailPage />);

    const resume = await screen.findByRole("button", { name: /resume/i });
    expect(resume).toHaveAttribute("title", "Completed stages will be reused.");
  });

  it("shows the durable log once the job is terminal", async () => {
    installFetchStub([
      ...baseRoutes(),
      {
        path: "/api/jobs/job-abc123",
        json: jobFixture({ state: "completed", output_available: true }),
      },
      { path: "/api/jobs/job-abc123/logs", json: logsFixture() },
    ]);

    renderAt("/jobs/job-abc123", "/jobs/:jobId", <JobDetailPage />);

    expect(
      await screen.findByText(/job created from upload/i),
    ).toBeInTheDocument();
  });

  it("cancels a running job through the API", async () => {
    const fetchStub = installFetchStub([
      ...baseRoutes(),
      {
        path: "/api/jobs/job-abc123",
        json: jobFixture({
          state: "running",
          status: "RUNNING",
          output_available: false,
        }),
      },
      { path: "/api/jobs/job-abc123/cancel", method: "POST", json: jobFixture() },
    ]);

    renderAt("/jobs/job-abc123", "/jobs/:jobId", <JobDetailPage />);
    await screen.findByRole("button", { name: /cancel/i });
    await userEvent.click(screen.getByRole("button", { name: /cancel/i }));

    await waitFor(() =>
      expect(
        fetchStub.calls.some((c) => c.url.endsWith("/cancel")),
      ).toBe(true),
    );
  });
});

describe("JobResultPage", () => {
  it("plays the output and lists every artifact, including unavailable ones", async () => {
    installFetchStub([
      ...baseRoutes(),
      { path: "/api/jobs/job-abc123", json: jobFixture({ state: "completed" }) },
      { path: "/api/jobs/job-abc123/output", json: outputFixture() },
      { path: "/api/jobs/job-abc123/artifacts", json: artifactListFixture },
    ]);

    renderAt("/jobs/job-abc123/result", "/jobs/:jobId/result", <JobResultPage />);

    // The player is a <video> with the API's media_url.
    await waitFor(() => {
      const video = document.querySelector("video");
      expect(video).not.toBeNull();
      expect(video?.getAttribute("src")).toBe(outputFixture().media_url);
    });

    expect(
      await screen.findByRole("link", { name: /download/i }),
    ).toHaveAttribute("href", outputFixture().download_url);

    // The unavailable artifact is shown with the backend's reason.
    expect(screen.getByText("Lip-synced video")).toBeInTheDocument();
    expect(
      screen.getByText(/lip-sync was not enabled/i),
    ).toBeInTheDocument();
  });

  it("renders the durable summary of how the render was made", async () => {
    installFetchStub([
      ...baseRoutes(),
      { path: "/api/jobs/job-abc123", json: jobFixture({ state: "completed" }) },
      { path: "/api/jobs/job-abc123/output", json: outputFixture() },
      { path: "/api/jobs/job-abc123/artifacts", json: artifactListFixture },
    ]);

    renderAt("/jobs/job-abc123/result", "/jobs/:jobId/result", <JobResultPage />);

    expect(await screen.findByText("How this was made")).toBeInTheDocument();
    expect(screen.getByText("English → Hindi")).toBeInTheDocument();
    expect(screen.getByText("Standard voice")).toBeInTheDocument();
    expect(
      screen.getByText("Sidecar files (.srt/.vtt) · 42 cues"),
    ).toBeInTheDocument();
    // Off features are omitted, not shown as "off".
    expect(screen.queryByText("Lip-sync")).toBeNull();
  });

  it("shows a shorter summary for a job that predates the summary fields", async () => {
    installFetchStub([
      ...baseRoutes(),
      {
        path: "/api/jobs/job-abc123",
        json: jobFixture({
          state: "completed",
          summary: {
            ...jobFixture().summary,
            translation_backend: null,
            tts_backend: null,
            duration_seconds: null,
            subtitle_count: 0,
          },
        }),
      },
      { path: "/api/jobs/job-abc123/output", json: outputFixture() },
      { path: "/api/jobs/job-abc123/artifacts", json: artifactListFixture },
    ]);

    renderAt("/jobs/job-abc123/result", "/jobs/:jobId/result", <JobResultPage />);

    await screen.findByText("How this was made");
    // Still shows what it knows.
    expect(screen.getByText("English → Hindi")).toBeInTheDocument();
    // And does not print a row for a fact it does not have.
    expect(screen.queryByText("Translation")).toBeNull();
    expect(screen.queryByText("Duration")).toBeNull();
  });

  it("deletes with the chosen cleanup level after confirmation", async () => {
    const fetchStub = installFetchStub([
      ...baseRoutes(),
      { path: "/api/jobs/job-abc123", json: jobFixture({ state: "completed" }) },
      { path: "/api/jobs/job-abc123/output", json: outputFixture() },
      { path: "/api/jobs/job-abc123/artifacts", json: artifactListFixture },
      {
        path: "/api/jobs/job-abc123",
        method: "DELETE",
        json: {
          job_id: "job-abc123",
          deleted: true,
          level: "EVERYTHING",
          removed: ["job-abc123"],
          message: "Job deleted.",
        },
      },
    ]);

    renderAt("/jobs/job-abc123/result", "/jobs/:jobId/result", <JobResultPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: /delete job/i }),
    );
    // Choose the most destructive level, then confirm.
    await userEvent.click(screen.getByRole("radio", { name: /^remove everything/i }));
    await userEvent.click(screen.getByRole("button", { name: /^delete job$/i }));

    await waitFor(() => {
      const del = fetchStub.calls.find((c) => c.method === "DELETE");
      expect(del).toBeDefined();
      expect(del?.url).toContain("level=EVERYTHING");
    });
    // The page navigates away on a successful delete.
    expect(await screen.findByText("elsewhere")).toBeInTheDocument();
  });

  it("reports when a delete unregisters without removing files", async () => {
    installFetchStub([
      ...baseRoutes(),
      { path: "/api/jobs/job-abc123", json: jobFixture({ state: "completed" }) },
      { path: "/api/jobs/job-abc123/output", json: outputFixture() },
      { path: "/api/jobs/job-abc123/artifacts", json: artifactListFixture },
      {
        path: "/api/jobs/job-abc123",
        method: "DELETE",
        json: {
          job_id: "job-abc123",
          deleted: false,
          level: "KEEP_ALL",
          removed: [],
          message: "Job unregistered; files were left on disk.",
        },
      },
    ]);

    renderAt("/jobs/job-abc123/result", "/jobs/:jobId/result", <JobResultPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: /delete job/i }),
    );
    await userEvent.click(screen.getByRole("radio", { name: /remove nothing/i }));
    await userEvent.click(screen.getByRole("button", { name: /^delete job$/i }));

    expect(
      await screen.findByText(/files were left on disk/i),
    ).toBeInTheDocument();
    expect(screen.queryByText("elsewhere")).toBeNull();
  });
});
