/**
 * Guided demo.
 *
 * The point of this page is to let someone see the real pipeline work without
 * first finding a video to upload. It is *not* a simulation: "Start Demo" stages
 * the bundled sample as an ordinary upload (`POST /api/samples/{id}/claim`) and
 * then creates a normal durable job through the same `createJob` call the
 * dubbing workspace uses, and the browser navigates to the ordinary
 * `/jobs/{id}` page for progress, SSE, cancel and result. There is no demo-only
 * runner and no second progress screen.
 *
 * Everything selectable is capability-checked against the backend. The defaults
 * come from `buildDemoDefaults`, the readiness panel from `buildReadiness`, and
 * the target list from the backend's own `priority_indic` — nothing here decides
 * what the machine can do, it only presents what the backend reports.
 */

import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ApiError } from "../api/client";
import { claimSample, createJob } from "../api/endpoints";
import { useBackendHealth } from "../hooks/useHealth";
import { useCapabilities } from "../hooks/useCapabilities";
import { useLanguages } from "../hooks/useLanguages";
import { useModels } from "../hooks/useModels";
import { useOptions } from "../hooks/useOptions";
import { useSamples } from "../hooks/useSamples";
import { Badge } from "../components/common/Badge";
import { ErrorState, SkeletonCard } from "../components/common/States";
import { OfflineBanner } from "../components/status/OfflineBanner";
import { PipelineFlow } from "../components/demo/PipelineFlow";
import { ReadinessPanel } from "../components/demo/ReadinessPanel";
import {
  IconArrowRight,
  IconCheck,
  IconPlay,
  IconShield,
} from "../components/common/Icons";
import { buildJobRequest, type DubFormValues } from "../utils/dubbing";
import {
  buildDemoDefaults,
  buildReadiness,
  popularDemoTargets,
} from "../utils/demo";
import { displayLanguage } from "../utils/languages";
import { formatBytes } from "../utils/format";
import type { SampleInfo } from "../api/types";

export function DemoPage() {
  const navigate = useNavigate();
  const { status } = useBackendHealth();
  const samples = useSamples();
  const capabilities = useCapabilities();
  const languages = useLanguages();
  const options = useOptions();
  const models = useModels();

  const offline = status === "offline";

  const sample: SampleInfo | null = useMemo(
    () => samples.data?.samples[0] ?? null,
    [samples.data],
  );

  const targets = languages.data?.targets ?? [];
  const allLanguages = languages.data?.languages ?? [];
  const ttsBackends = capabilities.data?.tts_backends ?? [];

  // --- form state ---------------------------------------------------------
  const [targetLang, setTargetLang] = useState("");
  const [translationModel, setTranslationModel] = useState("");
  const [ttsModel, setTtsModel] = useState("");
  const [qualityPreset, setQualityPreset] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | Error | null>(null);

  // Seed the demo configuration from capability once the backend data is in.
  // Each guard means a value the user has already chosen is never overwritten.
  useEffect(() => {
    if (!samples.data || !capabilities.data || !languages.data) return;
    const defaults = buildDemoDefaults({
      sample,
      capabilities: capabilities.data,
      languages: allLanguages,
      targets,
      ttsBackends,
      options: options.data,
    });
    setTargetLang((current) => current || defaults.targetLang);
    setTranslationModel((current) => current || defaults.translationModel);
    setTtsModel((current) => current || defaults.ttsModel);
    setQualityPreset((current) => (current === null ? defaults.qualityPreset : current));
    // Deliberately keyed on the data that shapes defaults, not on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [samples.data, capabilities.data, languages.data, options.data]);

  const readiness = useMemo(
    () =>
      buildReadiness({
        sample,
        capabilities: capabilities.data,
        models: models.data,
      }),
    [sample, capabilities.data, models.data],
  );

  const popular = useMemo(
    () => popularDemoTargets(capabilities.data, targets),
    [capabilities.data, targets],
  );

  const loading =
    samples.loading || capabilities.loading || languages.loading || options.loading;

  const canStart =
    readiness.ready &&
    Boolean(sample?.available) &&
    Boolean(targetLang) &&
    Boolean(ttsModel) &&
    !busy &&
    !offline;

  const ttsBackend = ttsBackends.find((b) => b.name === ttsModel) ?? null;

  // --- actions ------------------------------------------------------------

  async function handleStart() {
    if (!sample || !canStart) return;
    setBusy(true);
    setError(null);
    try {
      // Stage the bundled sample as a normal upload, then create a normal job.
      const staged = await claimSample(sample.id);
      const values: DubFormValues = {
        sourceLang: sample.source_language,
        targetLang,
        qualityPreset,
        translationModel,
        transcriptionModel: "",
        ttsModel,
        ttsVoice: null,
        enableDiarization: false,
        minSpeakers: 1,
        maxSpeakers: 10,
        enableLipsync: false,
        lipsyncModel: "",
        enableVad: false,
        enableAudioEnhancement: false,
        enableVisualTranslation: false,
        enableTimeStretch: false,
        optimizeTranslation: false,
        ttsEnableCfg: false,
        transcriptionBeamSize: 7,
        chunkDuration: 300,
        subtitleMode: "external",
      };
      const job = await createJob(buildJobRequest(staged.upload_id, values));
      navigate(`/jobs/${encodeURIComponent(job.job_id)}`);
    } catch (cause) {
      setError(
        cause instanceof Error ? cause : new Error("Could not start the demo."),
      );
      setBusy(false);
    }
  }

  // --- render -------------------------------------------------------------

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <span className="hero-eyebrow">
            <IconShield size={14} />
            Runs on this machine
          </span>
          <h1 className="page-title" style={{ marginTop: "var(--space-3)" }}>
            Try the demo
          </h1>
          <p className="page-subtitle">
            Run a real dub on a bundled public-domain clip — no upload needed.
            The demo stages the sample and starts the same durable job the
            dubbing workspace does.
          </p>
        </div>
      </header>

      {offline && <OfflineBanner />}

      {offline ? null : loading ? (
        <SkeletonCard lines={6} />
      ) : samples.error ? (
        <ErrorState error={samples.error} onRetry={samples.reload} />
      ) : capabilities.error ? (
        <ErrorState error={capabilities.error} onRetry={capabilities.reload} />
      ) : !sample ? (
        <ErrorState
          error={new Error("No demo sample is available on this installation.")}
          onRetry={samples.reload}
        />
      ) : (
        <>
          {/* The sample */}
          <section className="card">
            <div className="card-head">
              <h2 className="card-title">Sample</h2>
              {sample.available ? (
                <Badge tone="ok" dot>
                  Available
                </Badge>
              ) : (
                <Badge tone="warn" dot>
                  Not available
                </Badge>
              )}
            </div>
            <div className="demo-sample">
              <div className="demo-sample-info">
                <div className="demo-sample-title">{sample.title}</div>
                <p className="demo-sample-desc">{sample.description}</p>
                <div className="demo-sample-meta">
                  {sample.duration_seconds !== null && (
                    <span>{Math.round(sample.duration_seconds)}s</span>
                  )}
                  {sample.size_bytes !== null && (
                    <span>{formatBytes(sample.size_bytes)}</span>
                  )}
                  <span>
                    {displayLanguage(sample.source_language, allLanguages, "Auto")}
                    {" → "}
                    {displayLanguage(sample.target_language, allLanguages)}
                  </span>
                  <span className="demo-credit">{sample.credit}</span>
                </div>
                {!sample.available && (
                  <p className="field-hint field-hint-warn">
                    {sample.reason ??
                      "The sample media is not present in this checkout."}
                  </p>
                )}
              </div>
              {sample.available && (
                // eslint-disable-next-line jsx-a11y/media-has-caption
                <video
                  className="video-player demo-sample-player"
                  controls
                  preload="metadata"
                  src={`/api/samples/${encodeURIComponent(sample.id)}/media`}
                />
              )}
            </div>
          </section>

          {/* Readiness */}
          <section className="card">
            <div className="card-head">
              <h2 className="card-title">Demo readiness</h2>
              {readiness.ready ? (
                <Badge tone="ok" dot>
                  Ready
                </Badge>
              ) : (
                <Badge tone="danger" dot>
                  Blocked
                </Badge>
              )}
            </div>
            <ReadinessPanel readiness={readiness} />
          </section>

          {/* Configuration */}
          <section className="card">
            <div className="card-head">
              <h2 className="card-title">Demo settings</h2>
            </div>

            {!readiness.ready && readiness.blocker && (
              <div className="demo-unavailable" role="alert">
                <strong>Demo unavailable</strong>
                <p>
                  {readiness.blocker.detail ??
                    `${readiness.blocker.label} is not available.`}
                </p>
                <div className="btn-row">
                  <Link to="/dashboard" className="btn btn-secondary">
                    System status
                    <IconArrowRight size={14} />
                  </Link>
                </div>
              </div>
            )}

            <div className="form-grid">
              <div className="select-field">
                <label htmlFor="demo-source">Source language</label>
                <span className="field-static">
                  {displayLanguage(sample.source_language, allLanguages, "Auto-detect")}
                  <span className="field-hint"> (from the sample)</span>
                </span>
              </div>

              <div className="select-field">
                <label htmlFor="demo-target">Target language</label>
                {popular.length > 0 && (
                  <div
                    className="demo-target-chips"
                    role="group"
                    aria-label="Popular demo targets"
                  >
                    {popular.map((language) => (
                      <button
                        key={language.code}
                        type="button"
                        className="filter-chip"
                        aria-pressed={targetLang === language.code}
                        onClick={() => setTargetLang(language.code)}
                      >
                        {language.name}
                      </button>
                    ))}
                  </div>
                )}
                <select
                  id="demo-target"
                  className="select-input"
                  value={targetLang}
                  onChange={(event) => setTargetLang(event.target.value)}
                >
                  {targets.map((language) => (
                    <option key={language.code} value={language.code}>
                      {language.name}
                      {language.priority_indic ? " ★" : ""}
                    </option>
                  ))}
                </select>
                <span className="field-hint">
                  Popular Indian-language targets first; the full list is in the
                  selector.
                </span>
              </div>

              <div className="select-field">
                <label htmlFor="demo-translation">Translation</label>
                <select
                  id="demo-translation"
                  className="select-input"
                  value={translationModel}
                  onChange={(event) => setTranslationModel(event.target.value)}
                >
                  {(options.data?.translation_models ?? []).map((option) => (
                    <option
                      key={option.value}
                      value={option.value}
                      disabled={!option.available}
                    >
                      {option.label}
                      {!option.available ? " — unavailable" : ""}
                    </option>
                  ))}
                </select>
              </div>

              <div className="select-field">
                <label htmlFor="demo-tts">Voice</label>
                <select
                  id="demo-tts"
                  className="select-input"
                  value={ttsModel}
                  onChange={(event) => setTtsModel(event.target.value)}
                >
                  {ttsBackends.map((backend) => (
                    <option
                      key={backend.name}
                      value={backend.name}
                      disabled={!backend.available}
                    >
                      {backend.label}
                      {!backend.available ? " — unavailable" : ""}
                    </option>
                  ))}
                </select>
                <span className="field-hint">
                  {ttsBackend?.name === "edge"
                    ? "A standard generated voice. The demo does not clone a voice."
                    : ttsBackend?.label}
                </span>
              </div>
            </div>

            {qualityPreset && (
              <p className="field-hint">
                Quality: {qualityPreset}. A preset sets lip-sync and enhancement
                together.
              </p>
            )}

            {error && <ErrorState error={error} onRetry={() => setError(null)} />}

            <div className="btn-row start-row">
              <button
                type="button"
                className="btn btn-primary btn-lg"
                onClick={handleStart}
                disabled={!canStart}
              >
                <IconPlay size={16} />
                {busy ? "Starting…" : "Start demo"}
              </button>
              <span className="field-hint">
                {displayLanguage(sample.source_language, allLanguages, "Auto-detect")}
                {" → "}
                {displayLanguage(targetLang, allLanguages)}
                {ttsModel ? ` · ${ttsModel}` : ""}
              </span>
            </div>
          </section>

          {/* How it works */}
          <section className="card">
            <div className="card-head">
              <h2 className="card-title">What happens</h2>
            </div>
            <p className="card-subtitle">
              The same pipeline the full workspace runs. Optional stages depend on
              the configuration and the models installed here.
            </p>
            <PipelineFlow />
          </section>

          {/* The preferred route, as a check list of what the demo does. */}
          <section className="card">
            <div className="card-head">
              <h2 className="card-title">Demo route</h2>
            </div>
            <ul className="demo-route-list">
              <RouteItem label="Bundled sample" value={sample.title} />
              <RouteItem
                label="Languages"
                value={`${displayLanguage(sample.source_language, allLanguages, "Auto")} → ${displayLanguage(targetLang, allLanguages)}`}
              />
              <RouteItem
                label="Translation"
                value={
                  (options.data?.translation_models ?? []).find(
                    (o) => o.value === translationModel,
                  )?.label ?? translationModel
                }
              />
              <RouteItem
                label="Voice"
                value={ttsBackend?.label ?? ttsModel}
              />
              <RouteItem
                label="Quality"
                value={qualityPreset ?? "Custom (defaults)"}
              />
            </ul>
            <p className="field-hint">
              <IconCheck size={14} /> Starting the demo creates a normal durable
              job you can follow, cancel or resume from the Jobs page.
            </p>
          </section>
        </>
      )}
    </div>
  );
}

function RouteItem({ label, value }: { label: string; value: string }) {
  return (
    <li className="demo-route-item">
      <span className="demo-route-label">{label}</span>
      <span className="demo-route-value">{value}</span>
    </li>
  );
}
