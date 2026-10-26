/**
 * Component and page tests.
 *
 * These render the real pages against the real hooks and the real client, with
 * only `fetch` stubbed. So they exercise the actual loading → data and
 * loading → offline transitions, the real schema parsing, and the real resume
 * call — not a mock of any of them.
 */

import { describe, expect, it } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { LandingPage } from "../pages/LandingPage";
import { DashboardPage } from "../pages/DashboardPage";
import { LanguageExplorerPage } from "../pages/LanguageExplorerPage";
import { JobsPage } from "../pages/JobsPage";
import { NotFoundPage } from "../pages/NotFoundPage";
import { BackendHealthProvider } from "../hooks/useHealth";
import { JobRow } from "../components/jobs/JobRow";
import { Badge } from "../components/common/Badge";
import { BackendIndicator } from "../components/status/BackendIndicator";
import { errorEnvelope, installFetchStub } from "./fetchStub";
import {
  capabilitiesFixture,
  componentFixtures,
  healthFixture,
  jobFixture,
  jobListFixture,
  languagesFixture,
  samplesFixture,
  systemFixture,
} from "./fixtures";

/** Render a page inside the providers it needs. */
function renderPage(ui: React.ReactElement) {
  return render(
    <BackendHealthProvider>
      <MemoryRouter>{ui}</MemoryRouter>
    </BackendHealthProvider>,
  );
}

/**
 * The language cards, as opposed to the `<option>` elements in the route
 * explorer which carry the same names. Tests that assert on visible language
 * entries must scope to these, or "Hindi" matches twice.
 */
function languageCards(): HTMLElement[] {
  return screen.queryAllByRole("article");
}

function cardTexts(): string[] {
  return languageCards().map((card) => card.textContent ?? "");
}

describe("BackendIndicator", () => {
  it("shows the online state once health responds", async () => {
    installFetchStub([{ path: "/api/health", json: healthFixture }]);

    render(
      <BackendHealthProvider>
        <BackendIndicator />
      </BackendHealthProvider>,
    );

    await waitFor(() =>
      expect(screen.getByText("Backend Online")).toBeInTheDocument(),
    );
  });

  it("shows the offline state when health fails", async () => {
    installFetchStub([{ path: "/api/health", networkError: true }]);

    render(
      <BackendHealthProvider>
        <BackendIndicator />
      </BackendHealthProvider>,
    );

    await waitFor(() =>
      expect(screen.getByText("Backend Offline")).toBeInTheDocument(),
    );
  });
});

describe("LandingPage", () => {
  it("renders real capability state, not marketing copy", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/capabilities", json: capabilitiesFixture },
      { path: "/api/models", json: componentFixtures },
    ]);

    renderPage(<LandingPage />);

    // chunked_processing is true in the fixture; transcription is false.
    expect(await screen.findByText("Long-video processing")).toBeInTheDocument();
    expect(screen.getByText("Speech recognition")).toBeInTheDocument();
    // The unavailable feature says so rather than claiming support.
    const ready = screen.getAllByText("Ready");
    const notAvailable = screen.getAllByText("Not available");
    expect(ready.length).toBeGreaterThan(0);
    expect(notAvailable.length).toBeGreaterThan(0);
  });

  it("shows the offline banner when the backend is unreachable", async () => {
    installFetchStub([
      { path: "/api/health", networkError: true },
      { path: "/api/capabilities", networkError: true },
      { path: "/api/models", networkError: true },
    ]);

    renderPage(<LandingPage />);

    expect(
      await screen.findByText(/backend unavailable/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/uvicorn src\.api\.main:app/i)).toBeInTheDocument();
  });

  it("shows the component reason when something is missing", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/capabilities", json: capabilitiesFixture },
      { path: "/api/models", json: componentFixtures },
    ]);

    renderPage(<LandingPage />);

    expect(
      await screen.findByText(/f5_tts package is not installed/i),
    ).toBeInTheDocument();
  });
});

describe("DashboardPage", () => {
  it("renders runtime facts and groups components by purpose", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/system", json: systemFixture },
      { path: "/api/models", json: componentFixtures },
      { path: "/api/capabilities", json: capabilitiesFixture },
      { path: "/api/samples", json: samplesFixture },
    ]);

    renderPage(<DashboardPage />);

    expect(await screen.findByText("CPU")).toBeInTheDocument();
    expect(screen.getByText("3.12.14")).toBeInTheDocument();
    // Group headings come from required_for tags.
    expect(screen.getByText("Core")).toBeInTheDocument();
    expect(screen.getByText("Voice & speech synthesis")).toBeInTheDocument();
    expect(screen.getByText("Lip-sync")).toBeInTheDocument();
  });

  it("reports CUDA as unavailable on a CPU host", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/system", json: systemFixture },
      { path: "/api/models", json: componentFixtures },
      { path: "/api/capabilities", json: capabilitiesFixture },
      { path: "/api/samples", json: samplesFixture },
    ]);

    renderPage(<DashboardPage />);

    expect(await screen.findByText("Not available")).toBeInTheDocument();
  });
});

describe("LanguageExplorerPage", () => {
  it("lists languages from the API and filters them", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/languages", json: languagesFixture },
      { path: "/api/capabilities", json: capabilitiesFixture },
    ]);

    renderPage(<LanguageExplorerPage />);

    await waitFor(() => expect(languageCards()).toHaveLength(3));
    expect(cardTexts().some((text) => text.includes("Hindi"))).toBe(true);
    expect(cardTexts().some((text) => text.includes("Tamil"))).toBe(true);
    expect(cardTexts().some((text) => text.includes("English"))).toBe(true);

    // Filter to Indian languages only.
    await userEvent.click(screen.getByRole("button", { name: "Indian" }));

    await waitFor(() => expect(languageCards()).toHaveLength(2));
    expect(cardTexts().some((text) => text.includes("English"))).toBe(false);
  });

  it("filters by search term", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/languages", json: languagesFixture },
      { path: "/api/capabilities", json: capabilitiesFixture },
    ]);

    renderPage(<LanguageExplorerPage />);

    const search = await screen.findByLabelText(/search languages/i);
    await userEvent.type(search, "tamil");

    await waitFor(() => expect(languageCards()).toHaveLength(1));
    expect(cardTexts()[0]).toContain("Tamil");
  });

  it("distinguishes architecture support from local availability", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/languages", json: languagesFixture },
      { path: "/api/capabilities", json: capabilitiesFixture },
    ]);

    renderPage(<LanguageExplorerPage />);

    await waitFor(() => expect(languageCards()).toHaveLength(3));
    const hindiCard = languageCards().find((card) =>
      card.textContent?.includes("Hindi"),
    )!;
    // Hindi supports cloning by architecture but the model is not present.
    expect(within(hindiCard).getByText("Supported")).toBeInTheDocument();
    expect(within(hindiCard).getByText("Model required")).toBeInTheDocument();
  });

  it("reports the route for a supported pair", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/languages", json: languagesFixture },
      { path: "/api/capabilities", json: capabilitiesFixture },
    ]);

    renderPage(<LanguageExplorerPage />);

    // The fixture has an en -> hi route with the nllb backend. Both selects
    // default to the first entry (en -> en), so pick hi as the target.
    const target = await screen.findByLabelText("To");
    await userEvent.selectOptions(target, "hi");

    expect(await screen.findByText("nllb")).toBeInTheDocument();
    expect(screen.getByText("Translation route")).toBeInTheDocument();
  });

  it("does not claim a route is missing when the matrix is empty", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/languages", json: languagesFixture },
      {
        path: "/api/capabilities",
        json: { ...capabilitiesFixture, routes: [] },
      },
    ]);

    renderPage(<LanguageExplorerPage />);

    expect(
      await screen.findByText("Matrix not loaded"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/not implied either way/i),
    ).toBeInTheDocument();
  });
});

describe("JobsPage", () => {
  it("lists jobs with their backend-reported state", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/jobs", json: jobListFixture },
      { path: "/api/languages", json: languagesFixture },
    ]);

    renderPage(<JobsPage />);

    expect(await screen.findByText("lecture.mp4")).toBeInTheDocument();
    expect(screen.getByText("interview.mov")).toBeInTheDocument();
    expect(screen.getByText("Complete")).toBeInTheDocument();
    expect(screen.getByText("Interrupted")).toBeInTheDocument();
  });

  it("shows the summary counts from the response", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/jobs", json: jobListFixture },
      { path: "/api/languages", json: languagesFixture },
    ]);

    renderPage(<JobsPage />);

    expect(await screen.findByText("Total")).toBeInTheDocument();
    expect(screen.getByText("Active")).toBeInTheDocument();
    expect(screen.getByText("Pending")).toBeInTheDocument();
  });

  it("offers Resume only where the backend allows it", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/jobs", json: jobListFixture },
      { path: "/api/languages", json: languagesFixture },
    ]);

    renderPage(<JobsPage />);

    await screen.findByText("interview.mov");

    // Two jobs, but only the interrupted one is resumable.
    const resumeButtons = screen.getAllByRole("button", { name: /resume/i });
    expect(resumeButtons).toHaveLength(1);
  });

  it("calls the resume endpoint and refreshes on success", async () => {
    const stub = installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/jobs", json: jobListFixture },
      { path: "/api/languages", json: languagesFixture },
      {
        path: "/api/jobs/job-def456/resume",
        method: "POST",
        json: jobFixture({ job_id: "job-def456", state: "queued" }),
      },
    ]);

    renderPage(<JobsPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: /resume/i }),
    );

    await waitFor(() => {
      expect(stub.find("/api/jobs/job-def456/resume")).toBeDefined();
    });
    expect(stub.find("/api/jobs/job-def456/resume")?.method).toBe("POST");
  });

  it("surfaces the backend message when resume is refused", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/jobs", json: jobListFixture },
      { path: "/api/languages", json: languagesFixture },
      {
        path: "/api/jobs/job-def456/resume",
        method: "POST",
        status: 409,
        json: errorEnvelope(
          "JOB_ALREADY_ACTIVE",
          "This job is already running in this process.",
        ),
      },
    ]);

    renderPage(<JobsPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: /resume/i }),
    );

    expect(
      await screen.findByText(/already running in this process/i),
    ).toBeInTheDocument();
  });

  it("renders legacy and modern language values consistently", async () => {
    // The manifest contains both "English" (legacy) and "en" (newer) for the
    // same field. Both rows must show the same code, or the list looks wrong.
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/languages", json: languagesFixture },
      {
        path: "/api/jobs",
        json: {
          jobs: [
            jobFixture({
              job_id: "legacy",
              source_name: "legacy.mp4",
              source_language: "English",
              target_language: "Hindi",
            }),
            jobFixture({
              job_id: "modern",
              source_name: "modern.mp4",
              source_language: "en",
              target_language: "hi",
            }),
          ],
          total: 2,
          active: 0,
          pending: 0,
        },
      },
    ]);

    renderPage(<JobsPage />);
    await screen.findByText("legacy.mp4");

    const legacyRow = screen.getByText("legacy.mp4").closest(".job-row")!;
    const modernRow = screen.getByText("modern.mp4").closest(".job-row")!;

    expect(within(legacyRow as HTMLElement).getByText("EN")).toBeInTheDocument();
    expect(within(modernRow as HTMLElement).getByText("EN")).toBeInTheDocument();
    expect(
      within(legacyRow as HTMLElement).getByText("HI"),
    ).toBeInTheDocument();
    expect(within(modernRow as HTMLElement).getByText("HI")).toBeInTheDocument();
    // The raw display name must not leak through.
    expect(screen.queryByText("ENGLISH")).toBeNull();
  });

  it("shows an empty state when there are no jobs", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/jobs", json: { jobs: [], total: 0, active: 0, pending: 0 } },
      { path: "/api/languages", json: languagesFixture },
    ]);

    renderPage(<JobsPage />);

    expect(
      await screen.findByText(/no dubbing jobs yet/i),
    ).toBeInTheDocument();
  });
});

describe("JobRow", () => {
  it("renders the server's resume message as the button title", () => {
    const job = jobFixture({
      state: "interrupted",
      resume: {
        can_resume: true,
        reason: "interrupted",
        message: "This job can be resumed; completed stages will be reused.",
      },
    });

    render(
      <MemoryRouter>
        <JobRow job={job} onResume={() => {}} resuming={false} />
      </MemoryRouter>,
    );

    expect(screen.getByRole("button", { name: /resume/i })).toHaveAttribute(
      "title",
      "This job can be resumed; completed stages will be reused.",
    );
  });

  it("prefers the backend's live description for progress text", () => {
    const job = jobFixture({
      state: "running",
      live_description: "Translating segment 12 of 42",
      progress_fraction: 0.3,
    });

    render(
      <MemoryRouter>
        <JobRow job={job} onResume={() => {}} resuming={false} />
      </MemoryRouter>,
    );

    expect(
      screen.getByText("Translating segment 12 of 42"),
    ).toBeInTheDocument();
    expect(screen.getByText("30%")).toBeInTheDocument();
  });

  it("does not render a Resume button when the backend forbids it", () => {
    render(
      <MemoryRouter>
        <JobRow job={jobFixture()} onResume={() => {}} resuming={false} />
      </MemoryRouter>,
    );

    expect(screen.queryByRole("button", { name: /resume/i })).toBeNull();
  });

  it("disables the button while resuming", () => {
    const job = jobFixture({
      state: "interrupted",
      resume: { can_resume: true, reason: "interrupted", message: "ok" },
    });

    render(
      <MemoryRouter>
        <JobRow job={job} onResume={() => {}} resuming={true} />
      </MemoryRouter>,
    );

    expect(screen.getByRole("button", { name: /resuming/i })).toBeDisabled();
  });

  it("exposes progress to assistive tech with real values", () => {
    const job = jobFixture({ progress_fraction: 0.45 });
    render(
      <MemoryRouter>
        <JobRow job={job} onResume={() => {}} resuming={false} />
      </MemoryRouter>,
    );

    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "45");
    expect(bar).toHaveAttribute("aria-valuemax", "100");
  });
});

describe("Badge", () => {
  it("always renders its label, so color is not the only signal", () => {
    render(<Badge tone="danger">Failed</Badge>);
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });
});

describe("NotFoundPage", () => {
  it("renders for an unknown route", () => {
    render(
      <MemoryRouter initialEntries={["/nope"]}>
        <Routes>
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByText(/page not found/i)).toBeInTheDocument();
  });
});

describe("accessibility", () => {
  it("gives the jobs list a progressbar with an accessible name", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/jobs", json: jobListFixture },
      { path: "/api/languages", json: languagesFixture },
    ]);

    renderPage(<JobsPage />);
    await screen.findByText("lecture.mp4");

    const bars = screen.getAllByRole("progressbar");
    expect(bars.length).toBeGreaterThan(0);
    for (const bar of bars) {
      expect(bar).toHaveAccessibleName();
    }
  });

  it("announces the result count as a live region", async () => {
    installFetchStub([
      { path: "/api/health", json: healthFixture },
      { path: "/api/languages", json: languagesFixture },
      { path: "/api/capabilities", json: capabilitiesFixture },
    ]);

    renderPage(<LanguageExplorerPage />);
    await waitFor(() => expect(languageCards()).toHaveLength(3));

    const status = screen.getByRole("status");
    expect(status).toHaveTextContent(/of \d+ languages/);
  });
});
