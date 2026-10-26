/**
 * Tests for the dubbing workflow's pure rules.
 *
 * These cover the logic that decides what the form offers and what the request
 * contains — the parts a component test would otherwise only reach indirectly.
 */

import { describe, expect, it } from "vitest";
import {
  buildJobRequest,
  defaultOptionValue,
  isLanguagePairValid,
  isSelectable,
  orderArtifacts,
  stageLabel,
  stageProgress,
  stageTone,
  summaryRows,
  type DubFormValues,
} from "../utils/dubbing";
import type {
  ArtifactDescriptor,
  JobSummary,
  OptionItem,
  StageStatus,
} from "../api/types";
import { artifactListFixture } from "./fixtures";

function option(overrides: Partial<OptionItem> = {}): OptionItem {
  return {
    value: "edge",
    label: "Edge TTS (online)",
    description: null,
    available: true,
    state: "available",
    ...overrides,
  };
}

function stage(name: string, status: string): StageStatus {
  return { name, status, error: null, attempts: 1 };
}

function formValues(overrides: Partial<DubFormValues> = {}): DubFormValues {
  return {
    sourceLang: "auto",
    targetLang: "hi",
    qualityPreset: null,
    translationModel: "Google Translate (Online, Fast)",
    transcriptionModel: "Large v3 Turbo (Fast)",
    ttsModel: "edge",
    ttsVoice: null,
    enableDiarization: false,
    minSpeakers: 1,
    maxSpeakers: 10,
    enableLipsync: false,
    lipsyncModel: "Wav2Lip-GAN (Low Quality - Fast)",
    enableVad: false,
    enableAudioEnhancement: false,
    enableVisualTranslation: false,
    enableTimeStretch: false,
    optimizeTranslation: false,
    ttsEnableCfg: false,
    transcriptionBeamSize: 7,
    chunkDuration: 300,
    subtitleMode: "external",
    ...overrides,
  };
}

describe("stageTone", () => {
  it("treats SKIPPED as done, not pending", () => {
    expect(stageTone("SKIPPED")).toBe("done");
    expect(stageTone("COMPLETE")).toBe("done");
  });

  it("maps the running and failed states", () => {
    expect(stageTone("RUNNING")).toBe("active");
    expect(stageTone("FAILED")).toBe("failed");
    expect(stageTone("PARTIAL")).toBe("pending");
  });

  it("falls back to pending for a state it does not know", () => {
    expect(stageTone("SOMETHING_NEW")).toBe("pending");
  });
});

describe("stageProgress", () => {
  it("counts SKIPPED as done and finds the current stage", () => {
    const stages = [
      stage("source", "COMPLETE"),
      stage("extraction", "COMPLETE"),
      stage("separation", "SKIPPED"),
      stage("transcription", "RUNNING"),
      stage("tts", "PENDING"),
    ];
    const progress = stageProgress(stages);
    expect(progress.done).toBe(3);
    expect(progress.total).toBe(5);
    expect(progress.currentIndex).toBe(3);
  });

  it("reports currentIndex -1 when every stage is done", () => {
    const progress = stageProgress([
      stage("source", "COMPLETE"),
      stage("final", "SKIPPED"),
    ]);
    expect(progress.currentIndex).toBe(-1);
    expect(progress.done).toBe(2);
  });

  it("handles an empty stage list", () => {
    expect(stageProgress([])).toEqual({ done: 0, total: 0, currentIndex: -1 });
  });
});

describe("stageLabel", () => {
  it("words each known status", () => {
    expect(stageLabel("COMPLETE")).toBe("Complete");
    expect(stageLabel("SKIPPED")).toBe("Skipped");
    expect(stageLabel("PARTIAL")).toBe("Partial");
  });

  it("passes an unknown status through rather than hiding it", () => {
    expect(stageLabel("WEIRD")).toBe("WEIRD");
    expect(stageLabel("")).toBe("Unknown");
  });
});

describe("option selection", () => {
  it("reports availability from the backend's flag", () => {
    expect(isSelectable(option({ available: true }))).toBe(true);
    expect(
      isSelectable(option({ available: false, state: "not_installed" })),
    ).toBe(false);
  });

  it("prefers the first selectable option as the default", () => {
    const options = [
      option({ value: "nllb", available: false, state: "not_installed" }),
      option({ value: "google", available: true }),
    ];
    expect(defaultOptionValue(options)).toBe("google");
  });

  it("falls back to the first option when none are selectable", () => {
    const options = [
      option({ value: "nllb", available: false, state: "not_installed" }),
      option({ value: "google", available: false, state: "not_installed" }),
    ];
    expect(defaultOptionValue(options)).toBe("nllb");
  });

  it("returns an empty string for an empty list", () => {
    expect(defaultOptionValue([])).toBe("");
  });
});

describe("isLanguagePairValid", () => {
  it("accepts auto-detect into any target", () => {
    expect(isLanguagePairValid("auto", "hi", "auto")).toBe(true);
  });

  it("rejects the same concrete language on both sides", () => {
    expect(isLanguagePairValid("en", "en", "auto")).toBe(false);
  });

  it("accepts a genuine translation pair", () => {
    expect(isLanguagePairValid("en", "hi", "auto")).toBe(true);
  });

  it("rejects an empty side", () => {
    expect(isLanguagePairValid("", "hi", "auto")).toBe(false);
    expect(isLanguagePairValid("en", "", "auto")).toBe(false);
  });
});

describe("buildJobRequest", () => {
  it("always sends the upload and the language pair", () => {
    const request = buildJobRequest("up-1", formValues());
    expect(request.upload_id).toBe("up-1");
    expect(request.source_lang).toBe("auto");
    expect(request.target_lang).toBe("hi");
  });

  it("omits a preset so the individual switches apply", () => {
    const request = buildJobRequest("up-1", formValues({ enableLipsync: true }));
    expect(request.quality_preset).toBeUndefined();
    expect(request.enable_lipsync).toBe(true);
  });

  it("lets a preset own lip-sync and does not also send the switches", () => {
    const request = buildJobRequest(
      "up-1",
      formValues({ qualityPreset: "HIGH", enableLipsync: true }),
    );
    expect(request.quality_preset).toBe("HIGH");
    expect(request.enable_lipsync).toBeUndefined();
    expect(request.lipsync_model_name).toBeUndefined();
  });

  it("only sends a speaker range when diarization is on", () => {
    const off = buildJobRequest("up-1", formValues({ enableDiarization: false }));
    expect(off.min_speakers).toBeUndefined();
    expect(off.max_speakers).toBeUndefined();

    const on = buildJobRequest(
      "up-1",
      formValues({ enableDiarization: true, minSpeakers: 2, maxSpeakers: 4 }),
    );
    expect(on.min_speakers).toBe(2);
    expect(on.max_speakers).toBe(4);
  });

  it("does not send a null voice, which the backend rejects", () => {
    const request = buildJobRequest("up-1", formValues({ ttsVoice: null }));
    expect(request.tts_voice).toBeUndefined();
  });

  it("sends a chosen voice", () => {
    const request = buildJobRequest(
      "up-1",
      formValues({ ttsVoice: "en-US-AriaNeural" }),
    );
    expect(request.tts_voice).toBe("en-US-AriaNeural");
  });

  it("carries the numeric and boolean settings through", () => {
    const request = buildJobRequest(
      "up-1",
      formValues({
        transcriptionBeamSize: 5,
        chunkDuration: 600,
        enableVad: true,
        optimizeTranslation: true,
      }),
    );
    expect(request.transcription_beam_size).toBe(5);
    expect(request.chunk_duration).toBe(600);
    expect(request.enable_vad).toBe(true);
    expect(request.optimize_translation).toBe(true);
  });

  it("sends the chosen subtitle mode", () => {
    const request = buildJobRequest(
      "up-1",
      formValues({ subtitleMode: "burned" }),
    );
    expect(request.subtitle_mode).toBe("burned");
  });

  it("still sends the subtitle mode when a preset owns lip-sync", () => {
    // A preset controls quality, not deliverables, so it must not drop the
    // subtitle choice.
    const request = buildJobRequest(
      "up-1",
      formValues({ qualityPreset: "FAST", subtitleMode: "embedded" }),
    );
    expect(request.quality_preset).toBe("FAST");
    expect(request.subtitle_mode).toBe("embedded");
  });

  it("omits an empty subtitle mode so the backend default applies", () => {
    const request = buildJobRequest("up-1", formValues({ subtitleMode: "" }));
    expect(request.subtitle_mode).toBeUndefined();
  });
});

describe("orderArtifacts", () => {
  it("puts video before subtitles and keeps the rest stable", () => {
    const ordered = orderArtifacts(artifactListFixture.artifacts);
    expect(ordered[0].type).toBe("video");
    expect(ordered.map((a) => a.name)).toContain("subtitles");
  });

  it("sorts by label within a type", () => {
    const artifacts: ArtifactDescriptor[] = [
      { ...artifactListFixture.artifacts[0], name: "b", label: "Beta" },
      { ...artifactListFixture.artifacts[0], name: "a", label: "Alpha" },
    ];
    expect(orderArtifacts(artifacts).map((a) => a.name)).toEqual(["a", "b"]);
  });

  it("does not mutate the input", () => {
    const input = [...artifactListFixture.artifacts];
    const before = input.map((a) => a.name);
    orderArtifacts(input);
    expect(input.map((a) => a.name)).toEqual(before);
  });
});

// ---------------------------------------------------------------------------
// Result summary
// ---------------------------------------------------------------------------
describe("summaryRows", () => {
  const base: JobSummary = {
    source_language: "en",
    source_language_name: "English",
    target_language: "hi",
    target_language_name: "Hindi",
    translation_backend: "Google Translate (Online, Fast)",
    context_backend: null,
    optimize_translation: false,
    voice_mode: "standard",
    tts_backend: "edge",
    tts_voice: null,
    subtitle_mode: "external",
    subtitle_count: 12,
    lipsync: false,
    lipsync_model: null,
    diarization: false,
    speakers: 0,
    duration_seconds: 90,
    status: "COMPLETE",
    state: "completed",
  };

  function value(summary: JobSummary, label: string): string | undefined {
    return summaryRows(summary).find((row) => row.label === label)?.value;
  }

  it("reads languages as a pair", () => {
    expect(value(base, "Languages")).toBe("English → Hindi");
  });

  it("falls back to the code when the name is unknown", () => {
    expect(
      value({ ...base, source_language_name: null, source_language: "zz" }, "Languages"),
    ).toBe("zz → Hindi");
  });

  it("omits the context model when translation optimization is off", () => {
    expect(summaryRows(base).some((r) => r.label === "Context model")).toBe(false);
    expect(
      value({ ...base, optimize_translation: true, context_backend: "HY-MT" }, "Context model"),
    ).toBe("HY-MT");
  });

  it("appends the chosen voice when there is one", () => {
    expect(value({ ...base, tts_voice: "hi-IN-SwaraNeural" }, "Speech synthesis")).toBe(
      "edge · hi-IN-SwaraNeural",
    );
  });

  it("shows the speaker count only when diarization ran", () => {
    expect(summaryRows(base).some((r) => r.label === "Speakers")).toBe(false);
    expect(
      value({ ...base, diarization: true, speakers: 3 }, "Speakers"),
    ).toBe("3 detected");
    expect(
      value({ ...base, diarization: true, speakers: 0 }, "Speakers"),
    ).toBe("detection ran");
  });

  it("spells out the subtitle mode and cue count", () => {
    expect(value(base, "Subtitles")).toBe("Sidecar files (.srt/.vtt) · 12 cues");
  });

  it("passes an unknown subtitle mode through rather than hiding it", () => {
    expect(value({ ...base, subtitle_mode: "burned-in-new" }, "Subtitles")).toBe(
      "burned-in-new · 12 cues",
    );
  });

  it("names an unknown voice mode verbatim", () => {
    expect(value({ ...base, voice_mode: "clone_all" }, "Voice")).toBe("clone_all");
  });

  it("omits lip-sync when it is off and names the model when it is on", () => {
    expect(summaryRows(base).some((r) => r.label === "Lip-sync")).toBe(false);
    expect(
      value({ ...base, lipsync: true, lipsync_model: "Wav2Lip-GAN" }, "Lip-sync"),
    ).toBe("Wav2Lip-GAN");
  });

  it("omits duration when the pipeline never recorded one", () => {
    expect(summaryRows(base).some((r) => r.label === "Duration")).toBe(true);
    expect(
      summaryRows({ ...base, duration_seconds: null }).some((r) => r.label === "Duration"),
    ).toBe(false);
  });
});
