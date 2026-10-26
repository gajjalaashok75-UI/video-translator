/**
 * The dubbing workflow: choose a video, set the options, start the render.
 *
 * Three phases on one page rather than a modal wizard, because the settings a
 * user changes depend on what the backend can do and it is clearer to see the
 * upload they chose while editing them. The phase is derived, not stored: with no
 * file it is "choose", with a staged upload it is "configure", and the Start
 * button submits.
 *
 * Everything selectable comes from the backend. The language lists are
 * `/api/languages` (`sources`/`targets`), the model lists are `/api/options`, the
 * quality presets and feature flags are `/api/capabilities`, and the voices are
 * `/api/voices` for the chosen pair. The frontend decides only layout and
 * wording — never which languages, models or features exist.
 *
 * Upload and job creation are separate calls on purpose, matching the backend:
 * a large upload is staged first, and the job claims it exactly once. That is
 * why a failure after upload (a bad setting) does not require re-uploading.
 */

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError } from "../api/client";
import { createJob, discardUpload, uploadVideo } from "../api/endpoints";
import type { UploadResponse } from "../api/types";
import { useBackendHealth } from "../hooks/useHealth";
import { useCapabilities } from "../hooks/useCapabilities";
import { useLanguages } from "../hooks/useLanguages";
import { useOptions } from "../hooks/useOptions";
import { usePreflight } from "../hooks/usePreflight";
import { useVoices } from "../hooks/useVoices";
import { OfflineBanner } from "../components/status/OfflineBanner";
import { UploadDropzone } from "../components/jobs/UploadDropzone";
import { PreflightPanel } from "../components/jobs/PreflightPanel";
import { Badge } from "../components/common/Badge";
import { ErrorState, SkeletonCard } from "../components/common/States";
import { IconArrowRight, IconCheck, IconUpload, IconWand } from "../components/common/Icons";
import { cx, formatBytes } from "../utils/format";
import {
  buildJobRequest,
  defaultOptionValue,
  isSelectable,
  type DubFormValues,
} from "../utils/dubbing";
import { displayLanguage } from "../utils/languages";
import { componentStatePresentation } from "../utils/status";
import type { OptionItem } from "../api/types";

type Phase = "choose" | "uploading" | "configure";

export function DubPage() {
  const navigate = useNavigate();
  const { status } = useBackendHealth();
  const languages = useLanguages();
  const options = useOptions();
  const capabilities = useCapabilities();

  const [file, setFile] = useState<File | null>(null);
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [uploadFraction, setUploadFraction] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | Error | null>(null);

  const offline = status === "offline";
  const phase: Phase = upload ? "configure" : busy ? "uploading" : "choose";

  // --- form state ---------------------------------------------------------
  const [sourceLang, setSourceLang] = useState("auto");
  const [targetLang, setTargetLang] = useState("");
  const [qualityPreset, setQualityPreset] = useState<string | null>(null);
  const [translationModel, setTranslationModel] = useState("");
  const [transcriptionModel, setTranscriptionModel] = useState("");
  const [ttsModel, setTtsModel] = useState("");
  const [ttsVoice, setTtsVoice] = useState<string | null>(null);
  const [enableDiarization, setEnableDiarization] = useState(false);
  const [minSpeakers, setMinSpeakers] = useState(1);
  const [maxSpeakers, setMaxSpeakers] = useState(10);
  const [enableLipsync, setEnableLipsync] = useState(false);
  const [lipsyncModel, setLipsyncModel] = useState("");
  const [enableVad, setEnableVad] = useState(false);
  const [enableAudioEnhancement, setEnableAudioEnhancement] = useState(false);
  const [enableVisualTranslation, setEnableVisualTranslation] = useState(false);
  const [enableTimeStretch, setEnableTimeStretch] = useState(false);
  const [optimizeTranslation, setOptimizeTranslation] = useState(false);
  const [ttsEnableCfg, setTtsEnableCfg] = useState(false);
  const [beamSize, setBeamSize] = useState(7);
  const [chunkDuration, setChunkDuration] = useState(300);
  const [subtitleMode, setSubtitleMode] = useState("external");
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const sources = languages.data?.sources ?? [];
  const targets = languages.data?.targets ?? [];
  const autoCode = languages.data?.auto_detect_code ?? "auto";

  const translationOptions = options.data?.translation_models ?? [];
  const transcriptionOptions = options.data?.transcription_models ?? [];
  const ttsOptions = options.data?.tts_models ?? [];
  const lipsyncOptions = options.data?.lipsync_models ?? [];
  const subtitleOptions = options.data?.subtitle_modes ?? [];

  // Seed defaults from the backend's own lists once they arrive, so a fresh form
  // offers the same defaults the Gradio UI does rather than empty selects. Each
  // guard means a value the user already chose is never overwritten.
  useEffect(() => {
    if (targetLang === "" && targets.length > 0) {
      // Hindi is the priority Indic target when present, else the first.
      const preferred = targets.find((t) => t.code === "hi") ?? targets[0];
      setTargetLang(preferred.code);
    }
  }, [targets, targetLang]);

  useEffect(() => {
    if (translationModel === "" && translationOptions.length > 0) {
      setTranslationModel(defaultOptionValue(translationOptions));
    }
  }, [options.data, translationModel]);

  useEffect(() => {
    if (transcriptionModel === "" && transcriptionOptions.length > 0) {
      setTranscriptionModel(defaultOptionValue(transcriptionOptions));
    }
  }, [options.data, transcriptionModel]);

  useEffect(() => {
    if (ttsModel === "" && ttsOptions.length > 0) {
      setTtsModel(defaultOptionValue(ttsOptions));
    }
  }, [options.data, ttsModel]);

  useEffect(() => {
    if (lipsyncModel === "" && lipsyncOptions.length > 0) {
      setLipsyncModel(defaultOptionValue(lipsyncOptions));
    }
  }, [options.data, lipsyncModel]);

  const voices = useVoices(ttsModel || null, targetLang || null);
  const cloningBackend = voices.data?.cloning ?? false;

  const features = capabilities.data?.features ?? {};
  const presets = capabilities.data?.quality_presets ?? [];

  const canStart = useMemo(() => {
    if (!upload) return false;
    if (!targetLang) return false;
    if (sourceLang !== autoCode && sourceLang === targetLang) return false;
    return true;
  }, [upload, targetLang, sourceLang, autoCode]);

  /**
   * The request the form currently describes.
   *
   * Built once and shared by Start and preflight so the readiness verdict is
   * always about the render that would actually be submitted — a second
   * hand-rolled copy would drift from this one.
   */
  const formValues: DubFormValues = {
    sourceLang,
    targetLang,
    qualityPreset,
    translationModel,
    transcriptionModel,
    ttsModel,
    ttsVoice: cloningBackend ? null : ttsVoice,
    enableDiarization,
    minSpeakers,
    maxSpeakers,
    enableLipsync,
    lipsyncModel,
    enableVad,
    enableAudioEnhancement,
    enableVisualTranslation,
    enableTimeStretch,
    optimizeTranslation,
    ttsEnableCfg,
    transcriptionBeamSize: beamSize,
    chunkDuration,
    subtitleMode,
  };

  const preflightRequest = useMemo(
    () => (upload ? buildJobRequest(upload.upload_id, formValues) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [upload, JSON.stringify(formValues)],
  );
  const preflight = usePreflight(preflightRequest);

  /**
   * Whether preflight says the render as configured cannot be correct.
   *
   * Only a *settled* verdict counts: while a newer check is in flight the old
   * one is stale, and a failed preflight is advisory (see `PreflightPanel`), so
   * neither may disable Start. `createJob` re-validates regardless, so the worst
   * case of trusting a stale "ready" is the 422 the user would have got anyway.
   */
  const preflightBlocked =
    !preflight.loading && preflight.data !== null && !preflight.data.ready;

  // --- actions ------------------------------------------------------------

  async function handleUpload(chosen: File) {
    setFile(chosen);
    setError(null);
    setBusy(true);
    setUploadFraction(0);
    try {
      const staged = await uploadVideo(chosen, {
        onProgress: (loaded, total) => {
          setUploadFraction(total > 0 ? loaded / total : null);
        },
      });
      setUpload(staged);
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      setError(cause instanceof Error ? cause : new Error("Upload failed."));
      setFile(null);
    } finally {
      setBusy(false);
      setUploadFraction(null);
    }
  }

  async function handleDiscard() {
    if (upload) {
      // Best-effort: the upload is harmless if this fails, so a failure is not
      // surfaced. The server also drops unclaimed uploads on its own schedule.
      discardUpload(upload.upload_id).catch(() => undefined);
    }
    setUpload(null);
    setFile(null);
    setError(null);
  }

  async function handleStart() {
    if (!upload) return;
    setBusy(true);
    setError(null);
    try {
      const job = await createJob(buildJobRequest(upload.upload_id, formValues));
      navigate(`/jobs/${encodeURIComponent(job.job_id)}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause : new Error("Could not start the job."));
      setBusy(false);
    }
  }

  // --- render -------------------------------------------------------------

  return (
    <div className="page">
      <header className="page-header">
        <h1 className="page-title">Dubbing workspace</h1>
        <p className="page-subtitle">
          Upload a video, choose languages and settings, then watch the render
          progress.
        </p>
      </header>

      {offline && <OfflineBanner />}

      <div className="dub-steps" aria-hidden="true">
        <Step index={1} label="Source video" state={upload ? "done" : "active"} />
        <Step index={2} label="Settings" state={upload ? "active" : "todo"} />
        <Step index={3} label="Render" state="todo" />
      </div>

      {/* 1. Upload */}
      <section className="card">
        <div className="card-head">
          <h2 className="card-title">
            <IconUpload size={18} /> Source video
          </h2>
          {upload && <Badge tone="ok">Staged</Badge>}
        </div>

        <UploadDropzone
          file={file}
          onFile={handleUpload}
          disabled={offline || busy || Boolean(upload)}
        />

        {phase === "uploading" && (
          <div className="upload-progress">
            <div className="job-progress-text">
              <span>Uploading {file?.name}</span>
              <span>
                {uploadFraction !== null
                  ? `${Math.round(uploadFraction * 100)}%`
                  : "…"}
              </span>
            </div>
            <div className="progress">
              <div
                className="progress-bar"
                style={{ width: `${(uploadFraction ?? 0) * 100}%` }}
              />
            </div>
          </div>
        )}

        {upload && (
          <div className="upload-meta">
            <span>{formatBytes(upload.bytes)}</span>
            {upload.media.duration !== null && (
              <span>{Math.round(upload.media.duration)}s</span>
            )}
            {upload.media.width !== null && upload.media.height !== null && (
              <span>
                {upload.media.width}×{upload.media.height}
              </span>
            )}
            {upload.media.has_audio === false && (
              <Badge tone="warn">No audio track</Badge>
            )}
            <button type="button" className="btn btn-ghost" onClick={handleDiscard}>
              Choose a different video
            </button>
          </div>
        )}

        {error && !upload && (
          <ErrorState error={error} onRetry={() => setError(null)} />
        )}
      </section>

      {/* 2. Settings */}
      {upload && (
        <section className="card">
          <div className="card-head">
            <h2 className="card-title">
              <IconWand size={18} /> Settings
            </h2>
          </div>

          {languages.loading || options.loading || capabilities.loading ? (
            <SkeletonCard lines={4} />
          ) : languages.error ? (
            <ErrorState error={languages.error} onRetry={languages.reload} />
          ) : (
            <>
              <div className="route-picker">
                <div className="select-field">
                  <label htmlFor="source-lang">Source language</label>
                  <select
                    id="source-lang"
                    className="select-input"
                    value={sourceLang}
                    onChange={(event) => setSourceLang(event.target.value)}
                  >
                    <option value={autoCode}>
                      Auto-detect
                      {languages.data?.languages.find((l) => l.code === autoCode)
                        ? ""
                        : ""}
                    </option>
                    {sources.map((language) => (
                      <option key={language.code} value={language.code}>
                        {language.name}
                      </option>
                    ))}
                  </select>
                </div>

                <span className="route-arrow">
                  <IconArrowRight size={18} />
                </span>

                <div className="select-field">
                  <label htmlFor="target-lang">Target language</label>
                  <select
                    id="target-lang"
                    className="select-input"
                    value={targetLang}
                    onChange={(event) => {
                      setTargetLang(event.target.value);
                      setTtsVoice(null);
                    }}
                  >
                    {targets.map((language) => (
                      <option key={language.code} value={language.code}>
                        {language.name}
                        {language.priority_indic ? " ★" : ""}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {sourceLang !== autoCode && sourceLang === targetLang && (
                <p className="field-error" role="alert">
                  Source and target are the same language; pick a different
                  target.
                </p>
              )}

              <div className="form-grid">
                <div className="select-field">
                  <label htmlFor="quality">Quality preset</label>
                  <select
                    id="quality"
                    className="select-input"
                    value={qualityPreset ?? ""}
                    onChange={(event) =>
                      setQualityPreset(event.target.value || null)
                    }
                  >
                    <option value="">Custom (choose options below)</option>
                    {presets.map((preset) => (
                      <option key={preset.name} value={preset.name}>
                        {preset.name}
                      </option>
                    ))}
                  </select>
                  <span className="field-hint">
                    A preset sets lip-sync and enhancement together and
                    overrides those switches.
                  </span>
                </div>

                <OptionSelect
                  id="translation"
                  label="Translation model"
                  options={translationOptions}
                  value={translationModel}
                  onChange={setTranslationModel}
                />

                <OptionSelect
                  id="transcription"
                  label="Transcription model"
                  options={transcriptionOptions}
                  value={transcriptionModel}
                  onChange={setTranscriptionModel}
                  feature={features.transcription}
                  featureName="Transcription"
                />

                <OptionSelect
                  id="tts"
                  label="Speech synthesis"
                  options={ttsOptions}
                  value={ttsModel}
                  onChange={(value) => {
                    setTtsModel(value);
                    setTtsVoice(null);
                  }}
                  feature={features.tts}
                  featureName="Speech synthesis"
                />

                {!cloningBackend && (
                  <div className="select-field">
                    <label htmlFor="voice">Voice</label>
                    <select
                      id="voice"
                      className="select-input"
                      value={ttsVoice ?? ""}
                      disabled={voices.loading || (voices.data?.voices.length ?? 0) === 0}
                      onChange={(event) =>
                        setTtsVoice(event.target.value || null)
                      }
                    >
                      <option value="">
                        {voices.loading
                          ? "Loading voices…"
                          : (voices.data?.voices.length ?? 0) === 0
                            ? "Default voice"
                            : "Default voice"}
                      </option>
                      {(voices.data?.voices ?? []).map((voice) => (
                        <option key={voice} value={voice}>
                          {voice}
                        </option>
                      ))}
                    </select>
                    {voices.data?.note && (
                      <span className="field-hint">{voices.data.note}</span>
                    )}
                  </div>
                )}

                {cloningBackend && (
                  <div className="select-field">
                    <label>Voice</label>
                    <span className="field-static">
                      This backend clones the speaker&rsquo;s voice from the
                      source audio; there is no voice to pick.
                    </span>
                  </div>
                )}
              </div>

              <FeatureToggle
                id="diarization"
                label="Detect speakers (diarization)"
                checked={enableDiarization}
                onChange={setEnableDiarization}
                available={features.diarization}
                featureName="Diarization"
              >
                {enableDiarization && (
                  <div className="speaker-range">
                    <label htmlFor="min-speakers">
                      Min speakers
                      <input
                        id="min-speakers"
                        type="number"
                        className="select-input"
                        min={1}
                        max={50}
                        value={minSpeakers}
                        onChange={(event) =>
                          setMinSpeakers(clamp(Number(event.target.value), 1, 50))
                        }
                      />
                    </label>
                    <label htmlFor="max-speakers">
                      Max speakers
                      <input
                        id="max-speakers"
                        type="number"
                        className="select-input"
                        min={1}
                        max={50}
                        value={maxSpeakers}
                        onChange={(event) =>
                          setMaxSpeakers(clamp(Number(event.target.value), 1, 50))
                        }
                      />
                    </label>
                  </div>
                )}
              </FeatureToggle>

              {qualityPreset === null && (
                <FeatureToggle
                  id="lipsync"
                  label="Lip sync"
                  checked={enableLipsync}
                  onChange={setEnableLipsync}
                  available={features.lip_sync}
                  featureName="Lip sync"
                >
                  {enableLipsync && (
                    <OptionSelect
                      id="lipsync-model"
                      label="Lip-sync model"
                      options={lipsyncOptions}
                      value={lipsyncModel}
                      onChange={setLipsyncModel}
                    />
                  )}
                </FeatureToggle>
              )}

              <button
                type="button"
                className="btn btn-ghost advanced-toggle"
                onClick={() => setAdvancedOpen((open) => !open)}
                aria-expanded={advancedOpen}
              >
                {advancedOpen ? "Hide advanced options" : "Show advanced options"}
              </button>

              {advancedOpen && (
                <div className="form-grid advanced-grid">
                  <FeatureToggle
                    id="vad"
                    label="Voice activity detection"
                    checked={enableVad}
                    onChange={setEnableVad}
                    available={features.transcription}
                    featureName="Transcription"
                  />
                  <FeatureToggle
                    id="enhance"
                    label="Enhance audio"
                    checked={enableAudioEnhancement}
                    onChange={setEnableAudioEnhancement}
                    available={features.audio_enhancement}
                    featureName="Audio enhancement"
                  />
                  <FeatureToggle
                    id="visual"
                    label="Translate on-screen text"
                    checked={enableVisualTranslation}
                    onChange={setEnableVisualTranslation}
                    available={features.visual_translation}
                    featureName="Visual translation"
                  />
                  <FeatureToggle
                    id="stretch"
                    label="Time-stretch audio to fit"
                    checked={enableTimeStretch}
                    onChange={setEnableTimeStretch}
                    available={features.time_stretch}
                    featureName="Time stretching"
                  />
                  <FeatureToggle
                    id="optimize"
                    label="Optimize translation context"
                    checked={optimizeTranslation}
                    onChange={setOptimizeTranslation}
                  />
                  <FeatureToggle
                    id="cfg"
                    label="Enable TTS guidance (CFG)"
                    checked={ttsEnableCfg}
                    onChange={setTtsEnableCfg}
                  />
                  <div className="select-field">
                    <label htmlFor="beam">Transcription beam size</label>
                    <input
                      id="beam"
                      type="number"
                      className="select-input"
                      min={1}
                      max={20}
                      value={beamSize}
                      onChange={(event) =>
                        setBeamSize(clamp(Number(event.target.value), 1, 20))
                      }
                    />
                  </div>
                  <div className="select-field">
                    <label htmlFor="chunk">Chunk duration (seconds)</label>
                    <input
                      id="chunk"
                      type="number"
                      className="select-input"
                      min={0}
                      max={3600}
                      value={chunkDuration}
                      onChange={(event) =>
                        setChunkDuration(clamp(Number(event.target.value), 0, 3600))
                      }
                    />
                    <span className="field-hint">
                      0 processes the whole video at once.
                    </span>
                  </div>
                  <div className="select-field">
                    <label htmlFor="subtitle-mode">Subtitles</label>
                    <select
                      id="subtitle-mode"
                      className="select-input"
                      value={subtitleMode}
                      onChange={(event) => setSubtitleMode(event.target.value)}
                    >
                      {(subtitleOptions.length > 0
                        ? subtitleOptions
                        : [{ value: "external", label: "Sidecar files (SRT + WebVTT)", available: true }]
                      ).map((option) => (
                        <option
                          key={option.value}
                          value={option.value}
                          disabled={!option.available}
                        >
                          {option.label}
                        </option>
                      ))}
                    </select>
                    <span className="field-hint">
                      {subtitleMode === "none"
                        ? "No subtitle files or tracks are produced."
                        : subtitleMode === "external"
                          ? "Writes .srt and .vtt sidecar files next to the video."
                          : "Writes sidecar files and adds the track to the video itself."}
                    </span>
                  </div>
                </div>
              )}

              {error && upload && (
                <ErrorState error={error} onRetry={handleStart} />
              )}

              <PreflightPanel
                checks={preflight.data?.checks ?? []}
                loading={preflight.loading}
                error={preflight.error ? preflight.error.message : null}
              />

              <div className="btn-row start-row">
                <button
                  type="button"
                  className="btn btn-primary btn-lg"
                  onClick={handleStart}
                  disabled={!canStart || busy || offline || preflightBlocked}
                >
                  {busy ? "Starting…" : "Start dubbing"}
                  {!busy && <IconArrowRight size={18} />}
                </button>
                <span className="field-hint">
                  {displayLanguage(sourceLang, languages.data?.languages ?? [], "Auto-detect")}
                  {" → "}
                  {displayLanguage(targetLang, languages.data?.languages ?? [])}
                </span>
              </div>
            </>
          )}
        </section>
      )}
    </div>
  );
}

function clamp(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function Step({
  index,
  label,
  state,
}: {
  index: number;
  label: string;
  state: "done" | "active" | "todo";
}) {
  return (
    <span className={cx("dub-step", `dub-step-${state}`)}>
      <span className="dub-step-index">
        {state === "done" ? <IconCheck size={14} /> : index}
      </span>
      {label}
    </span>
  );
}

/**
 * A select over a backend option list.
 *
 * Unavailable options are shown but disabled, with the backend's own state
 * wording beside them, so a user can see that a model exists and needs a
 * download instead of it silently missing from the list.
 */
function OptionSelect({
  id,
  label,
  options,
  value,
  onChange,
  feature,
  featureName,
}: {
  id: string;
  label: string;
  options: OptionItem[];
  value: string;
  onChange: (value: string) => void;
  feature?: boolean;
  featureName?: string;
}) {
  const anySelectable = options.some(isSelectable);
  return (
    <div className="select-field">
      <label htmlFor={id}>{label}</label>
      <select
        id={id}
        className="select-input"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map((option) => {
          const selectable = isSelectable(option);
          const presentation = componentStatePresentation(option.state);
          return (
            <option
              key={option.value}
              value={option.value}
              disabled={!selectable}
            >
              {option.label}
              {!selectable ? ` — ${presentation.label}` : ""}
            </option>
          );
        })}
      </select>
      {feature === false && featureName && (
        <span className="field-hint field-hint-warn">
          {featureName} is not available in this installation.
        </span>
      )}
      {feature !== false && !anySelectable && options.length > 0 && (
        <span className="field-hint field-hint-warn">
          None of these are installed yet.
        </span>
      )}
    </div>
  );
}

/**
 * A checkbox for a pipeline feature, gated on the backend's capability flag.
 *
 * When the feature cannot run, the control is disabled and the reason is shown;
 * the checkbox is not hidden, because "this exists but you cannot use it" is
 * more useful than an empty space. A feature with no flag passed is always
 * allowed — used for pure pipeline options like context optimization.
 */
function FeatureToggle({
  id,
  label,
  checked,
  onChange,
  available,
  featureName,
  children,
}: {
  id: string;
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  available?: boolean;
  featureName?: string;
  children?: ReactNode;
}) {
  const disabled = available === false;
  return (
    <div className={cx("feature-toggle", disabled && "feature-toggle-off")}>
      <label className="toggle-row" htmlFor={id}>
        <input
          id={id}
          type="checkbox"
          checked={checked && !disabled}
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span>{label}</span>
        {disabled && featureName && (
          <Badge tone="neutral">Unavailable</Badge>
        )}
      </label>
      {disabled && featureName && (
        <span className="field-hint field-hint-warn">
          {featureName} cannot run in this installation.
        </span>
      )}
      {children}
    </div>
  );
}
