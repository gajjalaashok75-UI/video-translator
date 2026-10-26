/**
 * Tests for the pure presentation and filtering helpers.
 *
 * These carry the rules that would otherwise be buried in components: how a
 * backend state is worded, how languages are filtered and sorted, and how
 * components are grouped. Testing them here keeps the components thin.
 */

import { describe, expect, it } from "vitest";
import {
  formatBytes,
  formatDuration,
  formatPercent,
  formatRelativeTime,
  cx,
} from "../utils/format";
import {
  componentStatePresentation,
  jobStatePresentation,
} from "../utils/status";
import {
  displayLanguage,
  filterLanguages,
  findRoute,
  LANGUAGE_FILTERS,
} from "../utils/languages";
import { groupModels } from "../utils/modelGroups";
import type { ComponentState, JobState } from "../api/types";
import {
  componentFixtures,
  languagesFixture,
} from "./fixtures";

describe("formatBytes", () => {
  it("formats each unit boundary", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
    expect(formatBytes(3 * 1024 * 1024 * 1024)).toBe("3.0 GB");
  });

  it("shows a dash for a missing value rather than zero", () => {
    expect(formatBytes(null)).toBe("—");
    expect(formatBytes(undefined)).toBe("—");
  });
});

describe("formatDuration", () => {
  it("scales from seconds to days", () => {
    expect(formatDuration(45)).toBe("45s");
    expect(formatDuration(90)).toBe("1m 30s");
    expect(formatDuration(3661)).toBe("1h 1m");
    expect(formatDuration(90000)).toBe("1d 1h");
  });

  it("handles a missing value", () => {
    expect(formatDuration(null)).toBe("—");
  });
});

describe("formatPercent", () => {
  it("rounds and clamps to 0..100", () => {
    expect(formatPercent(0.456)).toBe("46%");
    expect(formatPercent(1)).toBe("100%");
    expect(formatPercent(1.5)).toBe("100%");
    expect(formatPercent(-1)).toBe("0%");
  });
});

describe("formatRelativeTime", () => {
  const now = 1_700_000_000_000; // ms

  it("describes recent times as 'just now'", () => {
    expect(formatRelativeTime(1_700_000_000, now)).toBe("just now");
  });

  it("counts minutes, hours and days", () => {
    expect(formatRelativeTime(1_700_000_000 - 300, now)).toBe("5m ago");
    expect(formatRelativeTime(1_700_000_000 - 7200, now)).toBe("2h ago");
    expect(formatRelativeTime(1_700_000_000 - 172800, now)).toBe("2d ago");
  });

  it("falls back to an absolute date beyond a week", () => {
    const result = formatRelativeTime(1_700_000_000 - 86400 * 30, now);
    expect(result).not.toContain("ago");
  });

  it("handles a missing timestamp", () => {
    expect(formatRelativeTime(null, now)).toBe("—");
  });
});

describe("cx", () => {
  it("joins truthy class names", () => {
    expect(cx("a", false, "b", null, undefined, "c")).toBe("a b c");
  });
});

describe("state presentation", () => {
  it("covers every component state with a label and tone", () => {
    const states: ComponentState[] = [
      "available",
      "not_installed",
      "model_not_downloaded",
      "token_required",
      "unsupported_platform",
      "unavailable",
      "unknown",
    ];
    for (const state of states) {
      const presentation = componentStatePresentation(state);
      expect(presentation.label).toBeTruthy();
      expect(presentation.tone).toBeTruthy();
    }
  });

  it("words a missing model as an action, not a failure", () => {
    expect(componentStatePresentation("model_not_downloaded").label).toBe(
      "Model required",
    );
  });

  it("covers every job state", () => {
    const states: JobState[] = [
      "queued",
      "running",
      "completed",
      "failed",
      "cancelled",
      "interrupted",
      "unknown",
    ];
    for (const state of states) {
      expect(jobStatePresentation(state).label).toBeTruthy();
    }
  });

  it("does not color running and failed the same", () => {
    expect(jobStatePresentation("running").tone).not.toBe(
      jobStatePresentation("failed").tone,
    );
  });
});

describe("filterLanguages", () => {
  const languages = languagesFixture.languages;

  it("returns everything by default, priority Indic first", () => {
    const result = filterLanguages(languages);
    expect(result).toHaveLength(3);
    expect(result[0].priority_indic).toBe(true);
    expect(result[1].priority_indic).toBe(true);
    // English is last because it is not a priority Indic language.
    expect(result[2].code).toBe("en");
  });

  it("matches on code, name and alias, case-insensitively", () => {
    expect(filterLanguages(languages, { search: "HINDI" })[0].code).toBe("hi");
    expect(filterLanguages(languages, { search: "ta" })[0].code).toBe("ta");
    expect(filterLanguages(languages, { search: "தமிழ்" })[0].code).toBe("ta");
  });

  it("applies the Indian filter", () => {
    const result = filterLanguages(languages, { filter: "indic" });
    expect(result.map((l) => l.code).sort()).toEqual(["hi", "ta"]);
  });

  it("separates architecture cloning support from local availability", () => {
    const architecture = filterLanguages(languages, {
      filter: "cloning_arch",
    });
    const local = filterLanguages(languages, { filter: "cloning_local" });

    // All three support cloning by architecture...
    expect(architecture).toHaveLength(3);
    // ...but none has the model on this machine.
    expect(local).toHaveLength(0);
  });

  it("returns nothing for a term that matches nothing", () => {
    expect(filterLanguages(languages, { search: "zzzz" })).toEqual([]);
  });

  it("offers a filter definition for every id used in tests", () => {
    const ids = LANGUAGE_FILTERS.map((f) => f.id);
    expect(ids).toContain("all");
    expect(ids).toContain("indic");
    expect(ids).toContain("cloning_local");
  });
});

describe("findRoute", () => {
  const routes = [
    {
      source: "en",
      target: "hi",
      backend: "nllb",
      pivots: [],
      is_passthrough: false,
    },
  ];

  it("finds an exact pair when the matrix is loaded", () => {
    const result = findRoute(routes, "en", "hi", true);
    expect(result.route?.backend).toBe("nllb");
    expect(result.matrixLoaded).toBe(true);
  });

  it("reports a missing pair distinctly from an unloaded matrix", () => {
    expect(findRoute(routes, "en", "ta", true)).toEqual({
      route: null,
      matrixLoaded: true,
    });
    expect(findRoute(routes, "en", "ta", false)).toEqual({
      route: null,
      matrixLoaded: false,
    });
  });
});

describe("displayLanguage", () => {
  const languages = languagesFixture.languages;

  it("maps a stored display name to its code", () => {
    // Legacy jobs stored "English"; newer ones store "en". Both must agree.
    expect(displayLanguage("English", languages)).toBe("EN");
    expect(displayLanguage("en", languages)).toBe("EN");
  });

  it("is case- and whitespace-insensitive", () => {
    expect(displayLanguage("  english ", languages)).toBe("EN");
    expect(displayLanguage("ENGLISH", languages)).toBe("EN");
  });

  it("matches an alias, including a non-Latin one", () => {
    expect(displayLanguage("\u0939\u093f\u0928\u094d\u0926\u0940", languages)).toBe("HI");
  });

  it("passes an unknown language through rather than hiding it", () => {
    expect(displayLanguage("klingon", languages)).toBe("KLINGON");
  });

  it("uses the fallback for a missing value", () => {
    expect(displayLanguage(null, languages)).toBe("—");
    expect(displayLanguage("", languages)).toBe("—");
    expect(displayLanguage(undefined, languages, "AUTO")).toBe("AUTO");
  });

  it("falls back to the raw value when the list has not loaded", () => {
    expect(displayLanguage("English", [])).toBe("ENGLISH");
  });
});

describe("groupModels", () => {
  it("groups by the backend's own required_for tags", () => {
    const groups = groupModels(componentFixtures);
    const ids = groups.map((g) => g.id);
    expect(ids).toContain("core");
    expect(ids).toContain("tts");
    expect(ids).toContain("lipsync");
  });

  it("files a multi-tag component under its most specific group once", () => {
    const groups = groupModels(componentFixtures);
    const cloningOrTts = groups.filter((g) =>
      g.models.some((m) => m.name === "indicf5"),
    );
    // IndicF5 is tagged tts and cloning; it must appear exactly once.
    expect(cloningOrTts).toHaveLength(1);
    expect(cloningOrTts[0].id).toBe("tts");
  });

  it("returns groups in the defined order", () => {
    const ids = groupModels(componentFixtures).map((g) => g.id);
    expect(ids.indexOf("core")).toBeLessThan(ids.indexOf("tts"));
    expect(ids.indexOf("tts")).toBeLessThan(ids.indexOf("lipsync"));
  });

  it("puts an unclassified component in the fallback group", () => {
    const groups = groupModels([
      {
        name: "mystery",
        label: "Mystery",
        state: "unknown",
        available: false,
        reason: "",
        required_for: [],
        detail: {},
      },
    ]);
    expect(groups).toHaveLength(1);
    expect(groups[0].id).toBe("other");
  });
});
