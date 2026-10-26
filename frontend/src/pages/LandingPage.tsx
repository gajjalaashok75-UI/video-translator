/**
 * Landing page.
 *
 * The product claims are written to match what the backend can actually do. The
 * capability section is driven by `/api/capabilities` and `/api/models` rather
 * than static text, so a feature the machine cannot run is shown as
 * "Model required" instead of a check mark. That is the whole reason the section
 * is data-driven: marketing copy that promises cloning on a machine without the
 * weights would be wrong, and the backend already knows.
 */

import { Link } from "react-router-dom";
import { useCapabilities } from "../hooks/useCapabilities";
import { useModels } from "../hooks/useModels";
import { useBackendHealth } from "../hooks/useHealth";
import { Badge } from "../components/common/Badge";
import { ErrorState, SkeletonGrid } from "../components/common/States";
import { OfflineBanner } from "../components/status/OfflineBanner";
import { PipelineFlow } from "../components/demo/PipelineFlow";
import { componentStatePresentation } from "../utils/status";
import type { ComponentStatus } from "../api/types";
import {
  IconArrowRight,
  IconGlobe,
  IconLayers,
  IconMic,
  IconShield,
  IconSparkle,
  IconWand,
} from "../components/common/Icons";

/**
 * Human wording for the capability keys the backend returns in `features`.
 * Only the label lives here; whether the feature is available is read from the
 * response, never assumed.
 */
const FEATURE_LABELS: Record<string, string> = {
  transcription: "Speech recognition",
  separation: "Music & effects separation",
  diarization: "Multi-speaker detection",
  translation_local: "Local translation",
  translation_online: "Online translation",
  tts: "Voice generation",
  cloning: "Voice cloning",
  lip_sync: "Lip-sync",
  audio_enhancement: "Audio enhancement",
  visual_translation: "On-screen text translation",
  time_stretch: "Timing adjustment",
  chunked_processing: "Long-video processing",
  resume: "Resume & recovery",
};

const FEATURE_ORDER = [
  "transcription",
  "translation_online",
  "translation_local",
  "tts",
  "diarization",
  "separation",
  "lip_sync",
  "cloning",
  "audio_enhancement",
  "visual_translation",
  "time_stretch",
  "chunked_processing",
  "resume",
];

export function LandingPage() {
  const capabilities = useCapabilities();
  const models = useModels();
  const { status } = useBackendHealth();

  const offline = status === "offline";

  return (
    <div className="page">
      <section className="hero">
        <div className="hero-inner">
          <span className="hero-eyebrow">
            <IconShield size={14} />
            Runs on your own hardware
          </span>
          <h1 className="hero-title">
            AI Video Dubbing{" "}
            <span className="hero-title-accent">Across Languages</span>
          </h1>
          <p className="hero-subtitle">
            Translate a video into another language with speaker-aware voices,
            preserved background music and optional lip-sync. Built
            Indian-language-first, and able to resume a long render after an
            interruption.
          </p>
          <div className="btn-row">
            <Link to="/demo" className="btn btn-primary btn-lg">
              <IconSparkle size={16} />
              Try the Demo
            </Link>
            <Link to="/dub" className="btn btn-secondary btn-lg">
              Start Dubbing
              <IconArrowRight size={16} />
            </Link>
            <Link to="/languages" className="btn btn-ghost btn-lg">
              <IconGlobe size={16} />
              Explore Languages
            </Link>
          </div>
          <p className="hero-note">
            No clip handy? The demo dubs a bundled public-domain scene on this
            machine — no upload needed.
          </p>
        </div>
      </section>

      {offline && <OfflineBanner />}

      <section className="section">
        <div className="section-heading">
          <div>
            <h2>How it works</h2>
            <p>
              Speech in the source video is transcribed, translated and
              re-voiced, then mixed back with the original music and effects.
            </p>
          </div>
        </div>
        <div className="card">
          <PipelineFlow />
        </div>
      </section>

      <section className="section">
        <div className="section-heading">
          <div>
            <h2>What it does</h2>
            <p>Each capability below is read from this installation.</p>
          </div>
        </div>

        {offline ? null : capabilities.error ? (
          <ErrorState
            error={capabilities.error}
            onRetry={capabilities.reload}
          />
        ) : capabilities.loading || !capabilities.data ? (
          <SkeletonGrid count={6} lines={2} />
        ) : (
          <FeatureGrid features={capabilities.data.features} />
        )}
      </section>

      <section className="section">
        <div className="section-heading">
          <div>
            <h2>Optional components</h2>
            <p>
              Availability on this machine, with the reason when something is
              missing.
            </p>
          </div>
        </div>

        {offline ? null : models.error ? (
          <ErrorState error={models.error} onRetry={models.reload} />
        ) : models.loading || !models.data ? (
          <SkeletonGrid count={4} lines={2} />
        ) : models.data.length === 0 ? (
          <p className="page-subtitle">No optional components are reported.</p>
        ) : (
          <div className="card">
            {models.data.map((model) => (
              <ModelRow key={model.name} model={model} />
            ))}
          </div>
        )}
      </section>

      <section className="section">
        <div className="section-heading">
          <div>
            <h2>Privacy &amp; consent</h2>
            <p>Where your video goes, stated plainly.</p>
          </div>
        </div>
        <div className="card">
          <ul className="consent-list">
            <li className="consent-item">
              <span className="consent-icon" aria-hidden="true">
                <IconShield size={18} />
              </span>
              <div>
                <div className="consent-title">Processing stays local</div>
                <p className="consent-body">
                  Transcription, translation and speech generation run on this
                  machine by default. Your video is written to this server's own
                  storage while a job runs, and the finished file stays here until
                  you delete the job.
                </p>
              </div>
            </li>
            <li className="consent-item">
              <span className="consent-icon" aria-hidden="true">
                <IconGlobe size={18} />
              </span>
              <div>
                <div className="consent-title">Online services, when selected</div>
                <p className="consent-body">
                  Choosing an online backend — Edge TTS for voices or Google
                  Translate for text — sends the relevant audio or text to that
                  provider. That choice is yours and is shown on the dubbing form;
                  local backends keep everything on this machine.
                </p>
              </div>
            </li>
            <li className="consent-item">
              <span className="consent-icon" aria-hidden="true">
                <IconMic size={18} />
              </span>
              <div>
                <div className="consent-title">Only dub what you may</div>
                <p className="consent-body">
                  Dubbing reproduces a person's voice in another language. Use
                  only material you have the right to process, and do not use it
                  to impersonate or mislead.
                </p>
              </div>
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}

function FeatureGrid({ features }: { features: Record<string, boolean> }) {
  return (
    <div className="card-grid">
      {FEATURE_ORDER.filter((key) => key in features).map((key) => {
        const available = features[key];
        const label = FEATURE_LABELS[key] ?? key;
        return (
          <div className="feature-item" key={key}>
            <span className="feature-icon" aria-hidden="true">
              {featureIcon(key)}
            </span>
            <div>
              <div className="feature-title">
                {label}
                <Badge tone={available ? "ok" : "neutral"} dot>
                  {available ? "Ready" : "Not available"}
                </Badge>
              </div>
              <p className="feature-body">{featureBlurb(key)}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ModelRow({ model }: { model: ComponentStatus }) {
  const presentation = componentStatePresentation(model.state);
  return (
    <div className="capability-row">
      <span className="capability-name">
        <span>
          {model.label}
          <span className="capability-reason">{model.reason}</span>
        </span>
      </span>
      <Badge tone={presentation.tone} dot>
        {presentation.label}
      </Badge>
    </div>
  );
}

function featureIcon(key: string) {
  switch (key) {
    case "tts":
    case "cloning":
      return <IconMic size={18} />;
    case "lip_sync":
    case "audio_enhancement":
      return <IconWand size={18} />;
    case "translation_local":
    case "translation_online":
    case "transcription":
      return <IconGlobe size={18} />;
    default:
      return <IconLayers size={18} />;
  }
}

function featureBlurb(key: string): string {
  switch (key) {
    case "transcription":
      return "Speech is transcribed from the source video before translation.";
    case "translation_local":
      return "Translation runs on this machine with a local model.";
    case "translation_online":
      return "Translation uses an online service, which needs network access.";
    case "tts":
      return "Speech is synthesized in the target language.";
    case "cloning":
      return "A reference voice can be reproduced. Requires the model weights.";
    case "diarization":
      return "Different speakers are told apart so each keeps a consistent voice.";
    case "separation":
      return "Background music and effects are separated from dialogue and kept.";
    case "lip_sync":
      return "Mouth movements are adjusted to match the new audio. Optional.";
    case "audio_enhancement":
      return "The mixed audio is cleaned up after dubbing.";
    case "visual_translation":
      return "Text burned into the video is detected and translated. Optional.";
    case "time_stretch":
      return "Speech is stretched to fit the original timing.";
    case "chunked_processing":
      return "Long videos are processed in chunks rather than all at once.";
    case "resume":
      return "An interrupted job can continue from its last completed stage.";
    default:
      return "";
  }
}
