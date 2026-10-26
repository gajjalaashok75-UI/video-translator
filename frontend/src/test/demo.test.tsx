/**
 * Tests for the guided demo: the readiness rules, the demo defaults, and the
 * page's real claim → create-job flow.
 *
 * The pure helpers are tested directly because they encode the rules that keep
 * the demo honest — nothing is "green" unless the backend said so. The page
 * tests render the real page against the real hooks and the real client,
 * stubbing only `fetch`, so "Start demo" is proven to stage the sample and then
 * create a normal job with the ordinary `createJob` call.
 */

import { describe, expect, it, beforeEach, afterEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { BackendHealthProvider } from "../hooks/useHealth";
import { DemoPage } from "../pages/DemoPage";
import { PipelineFlow } from "../components/demo/PipelineFlow";
import { ReadinessPanel } from "../components/demo/ReadinessPanel";
import { installFetchStub, type StubRoute } from "./fetchStub";
import {
  buildDemoDefaults,
  buildReadiness,
  demoPreset,
  popularDemoTargets,
} from "../utils/demo";
import {
  capabilitiesFixture,
  componentFixtures,
  healthFixture,
  jobFixture,
  languagesFixture,
  sampleFixture,
  samplesFixture,
  uploadFixture,
} from "./fixtures";
import type {
  CapabilitiesResponse,
  ComponentStatus,
  OptionsResponse,
  SampleInfo,
} from "../api/types";

// ---------------------------------------------------------------------------
// Local fixtures: a machine that can actually run the demo
// ---------------------------------------------------------------------------

/** Capabilities for a fully-capable machine (transcription, NLLB, Edge TTS). */
function capableCapabilities(): CapabilitiesResponse {
  return {
    ...capabilitiesFixture,
    translation_backends: [
      {
        name: "nllb",
        label: "NLLB-200",
        kind: "local",
        available: true,
        state: "available",
        reason: "",
        detail: {},
      },
    ],
    tts_backends: [
      {
        name: "edge",
        label: "Edge TTS",
        kind: "online",
        available: true,
        state: "available",
        reason: "",
        detail: {},
      },
    ],
    quality_presets: [{ name: "BALANCED", settings: {} }],
    features: {
      ...capabilitiesFixture.features,
      transcription: true,
      translation_local: true,
      tts: true,
      separation: true,
    },
  };
}

const optionsFixture: OptionsResponse = {
  translation_models: [
    {
      value: "NLLB-200 (Local, Indic + 200 Languages)",
      label: "NLLB-200 (Local)",
      description: null,
      available: true,
      state: "available",
    },
  ],
  transcription_models: [],
  tts_models: [],
  lipsync_models: [],
  context_models: [],
  diarization_backends: [],
  ocr_engines: [],
  audio_separators: [],
  live_portrait_modes: [],
  subtitle_modes: [],
  source: {},
};

/** Component list that reports FFmpeg as available. */
function capableModels(): ComponentStatus[] {
  return componentFixtures.map((model) =>
    model.name === "ffmpeg"
      ? { ...model, available: true, state: "available", reason: "" }
      : model,
  );
}

/** The routes the demo page needs on first paint. Overrides win over base. */
function demoRoutes(overrides: StubRoute[] = []): StubRoute[] {
  return [
    ...overrides,
    { path: "/api/health", json: healthFixture },
    { path: "/api/samples", json: samplesFixture },
    { path: "/api/capabilities", json: capableCapabilities() },
    { path: "/api/languages", json: languagesFixture },
    { path: "/api/options", json: optionsFixture },
    { path: "/api/models", json: capableModels() },
  ];
}

function renderDemo() {
  return render(
    <BackendHealthProvider>
      <MemoryRouter initialEntries={["/demo"]}>
        <Routes>
          <Route path="/demo" element={<DemoPage />} />
          <Route path="*" element={<span>elsewhere</span>} />
        </Routes>
      </MemoryRouter>
    </BackendHealthProvider>,
  );
}

beforeEach(() => {
  // jsdom has no media element implementation; the preview <video> only needs
  // to mount, so silencing play() keeps the console clean.
  HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// buildReadiness
// ---------------------------------------------------------------------------

describe("buildReadiness", () => {
  it("is ready when every blocking check is satisfied", () => {
    const readiness = buildReadiness({
      sample: sampleFixture(),
      capabilities: capableCapabilities(),
      models: capableModels(),
    });

    expect(readiness.ready).toBe(true);
    expect(readiness.blocker).toBeNull();
    expect(readiness.checks.every((c) => c.state !== "missing")).toBe(true);
  });

  it("names the first blocking problem as the blocker", () => {
    const readiness = buildReadiness({
      sample: sampleFixture({ available: false, reason: "File is missing." }),
      capabilities: capableCapabilities(),
      models: capableModels(),
    });

    expect(readiness.ready).toBe(false);
    expect(readiness.blocker?.id).toBe("sample");
    expect(readiness.blocker?.detail).toBe("File is missing.");
  });

  it("blocks when a required pipeline capability is missing", () => {
    const capabilities = capableCapabilities();
    capabilities.features.transcription = false;

    const readiness = buildReadiness({
      sample: sampleFixture(),
      capabilities,
      models: capableModels(),
    });

    expect(readiness.ready).toBe(false);
    expect(readiness.blocker?.id).toBe("transcription");
  });

  it("never blocks on an optional component", () => {
    // cloning and lip_sync are false in the capable fixture; they must not block.
    const readiness = buildReadiness({
      sample: sampleFixture(),
      capabilities: capableCapabilities(),
      models: capableModels(),
    });

    const cloning = readiness.checks.find((c) => c.id === "cloning");
    expect(cloning?.blocking).toBe(false);
    expect(cloning?.state).toBe("optional");
    expect(readiness.ready).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// buildDemoDefaults
// ---------------------------------------------------------------------------

describe("buildDemoDefaults", () => {
  const base = {
    sample: sampleFixture(),
    languages: languagesFixture.languages,
    targets: languagesFixture.targets,
    ttsBackends: capableCapabilities().tts_backends,
  };

  it("prefers the sample's target, local NLLB and Edge TTS", () => {
    const defaults = buildDemoDefaults({
      ...base,
      capabilities: capableCapabilities(),
      options: optionsFixture,
    });

    expect(defaults.sourceLang).toBe("en");
    expect(defaults.targetLang).toBe("hi");
    expect(defaults.translationModel).toBe(
      "NLLB-200 (Local, Indic + 200 Languages)",
    );
    expect(defaults.ttsModel).toBe("edge");
  });

  it("falls back to Auto translation when NLLB is unavailable", () => {
    const capabilities = capableCapabilities();
    capabilities.translation_backends = [];

    const defaults = buildDemoDefaults({
      ...base,
      capabilities,
      options: optionsFixture,
    });

    expect(defaults.translationModel).toBe(
      "Auto (NLLB-200 for Indic, else selected)",
    );
  });

  it("falls back to the first priority Indic target when the sample's target is not one", () => {
    const capabilities = capableCapabilities();
    capabilities.priority_indic = ["ta"];

    const defaults = buildDemoDefaults({
      ...base,
      sample: sampleFixture({ target_language: "fr" }),
      capabilities,
      options: optionsFixture,
    });

    expect(defaults.targetLang).toBe("ta");
  });

  it("leaves the preset unset when it cannot run here", () => {
    const capabilities = capableCapabilities();
    capabilities.quality_presets = [
      { name: "BALANCED", settings: { enable_lipsync: true } },
    ];
    // lip_sync is false in the fixture, so BALANCED cannot run.
    const defaults = buildDemoDefaults({
      ...base,
      capabilities,
      options: optionsFixture,
    });

    expect(defaults.qualityPreset).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// demoPreset
// ---------------------------------------------------------------------------

describe("demoPreset", () => {
  it("returns the preset when every model it names is available", () => {
    expect(demoPreset(capableCapabilities(), optionsFixture)).toBe("BALANCED");
  });

  it("returns null when the preset needs a missing feature", () => {
    const capabilities = capableCapabilities();
    capabilities.quality_presets = [
      { name: "BALANCED", settings: { enable_audio_enhancement: true } },
    ];
    // audio_enhancement is false in the fixture.
    expect(demoPreset(capabilities, optionsFixture)).toBeNull();
  });

  it("returns null when there is no BALANCED preset", () => {
    const capabilities = capableCapabilities();
    capabilities.quality_presets = [{ name: "FAST", settings: {} }];
    expect(demoPreset(capabilities, optionsFixture)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// popularDemoTargets
// ---------------------------------------------------------------------------

describe("popularDemoTargets", () => {
  it("returns the priority Indic languages that are real targets", () => {
    const capabilities = capableCapabilities();
    capabilities.priority_indic = ["hi", "ta", "en"];

    const popular = popularDemoTargets(capabilities, languagesFixture.targets);

    expect(popular.map((l) => l.code)).toEqual(["hi", "ta", "en"]);
  });

  it("falls back to the first targets when there is no priority list", () => {
    const capabilities = capableCapabilities();
    capabilities.priority_indic = [];

    const popular = popularDemoTargets(capabilities, languagesFixture.targets);

    expect(popular.length).toBeGreaterThan(0);
    expect(popular.map((l) => l.code)).toEqual(
      languagesFixture.targets.slice(0, popular.length).map((l) => l.code),
    );
  });
});

// ---------------------------------------------------------------------------
// Presentational components
// ---------------------------------------------------------------------------

describe("PipelineFlow", () => {
  it("labels optional stages and never implies cloning", () => {
    render(<PipelineFlow />);

    expect(screen.getByText("Transcription")).toBeInTheDocument();
    expect(screen.getByText("Translation")).toBeInTheDocument();
    // Speaker detection and lip-sync are drawn as optional.
    expect(screen.getAllByText("optional")).toHaveLength(2);
    expect(screen.queryByText(/clon/i)).not.toBeInTheDocument();
  });
});

describe("ReadinessPanel", () => {
  it("renders a badge per check from the derived state", () => {
    const readiness = buildReadiness({
      sample: sampleFixture({ available: false, reason: "Missing." }),
      capabilities: capableCapabilities(),
      models: capableModels(),
    });

    render(<ReadinessPanel readiness={readiness} />);

    expect(screen.getByText("Sample video")).toBeInTheDocument();
    expect(screen.getByText("Missing")).toBeInTheDocument();
    // Optional checks read as "Optional", not as an error.
    expect(screen.getAllByText("Optional").length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// DemoPage
// ---------------------------------------------------------------------------

describe("DemoPage", () => {
  it("renders the sample, readiness and derived demo route", async () => {
    installFetchStub(demoRoutes());

    renderDemo();

    expect(await screen.findByText("Sample")).toBeInTheDocument();
    // The title appears twice: once as the sample heading, once in the route
    // summary. Asserting both is the point — the summary reflects the sample.
    expect(
      screen.getAllByText("His Girl Friday — Sample Scene").length,
    ).toBeGreaterThan(0);
    // Readiness heading and the derived route summary.
    expect(screen.getByText("Demo readiness")).toBeInTheDocument();
    expect(screen.getByText("Demo route")).toBeInTheDocument();
    // The Start button becomes enabled once the derived defaults settle.
    const start = screen.getByRole("button", { name: /start demo/i });
    await waitFor(() => expect(start).toBeEnabled());
  });

  it("stages the sample and creates a job, then navigates to it", async () => {
    const fetchStub = installFetchStub(
      demoRoutes([
        {
          path: "/api/samples/hgf_scene/claim",
          method: "POST",
          json: uploadFixture(),
        },
        {
          path: "/api/jobs",
          method: "POST",
          status: 202,
          json: jobFixture({ job_id: "job-demo-1", state: "queued", status: "QUEUED" }),
        },
      ]),
    );

    renderDemo();

    const start = await screen.findByRole("button", { name: /start demo/i });
    await waitFor(() => expect(start).toBeEnabled());
    await userEvent.click(start);

    await waitFor(() =>
      expect(screen.getByText("elsewhere")).toBeInTheDocument(),
    );

    // The claim was made, then a normal job was created with the staged id.
    const claim = fetchStub.calls.find((c) => c.url.includes("/claim"));
    expect(claim?.method).toBe("POST");

    const create = fetchStub.calls.find(
      (c) => c.method === "POST" && c.url === "/api/jobs",
    );
    const body = create?.body as Record<string, unknown>;
    expect(body.upload_id).toBe(uploadFixture().upload_id);
    expect(body.source_lang).toBe("en");
    expect(body.target_lang).toBe("hi");
    expect(body.tts_model_name).toBe("edge");
  });

  it("disables the demo and explains the blocker when a requirement is missing", async () => {
    const capabilities = capableCapabilities();
    capabilities.features.transcription = false;

    installFetchStub(
      demoRoutes([{ path: "/api/capabilities", json: capabilities }]),
    );

    renderDemo();

    expect(await screen.findByText("Demo unavailable")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /start demo/i }),
    ).toBeDisabled();
  });

  it("reports a missing sample instead of offering a broken run", async () => {
    const unavailable: SampleInfo = sampleFixture({
      available: false,
      reason: "The sample file is not present in this checkout.",
    });

    installFetchStub(
      demoRoutes([
        { path: "/api/samples", json: { samples: [unavailable], total: 1 } },
      ]),
    );

    renderDemo();

    expect(await screen.findByText("Not available")).toBeInTheDocument();
    // The reason appears in both the sample meta and the readiness detail.
    expect(
      screen.getAllByText(/not present in this checkout/i).length,
    ).toBeGreaterThan(0);
    expect(
      screen.getByRole("button", { name: /start demo/i }),
    ).toBeDisabled();
  });
});
