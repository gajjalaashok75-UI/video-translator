/**
 * System dashboard.
 *
 * Shows the runtime facts (`/api/system`), the backend version (`/api/health`)
 * and the availability of every optional component (`/api/models`), grouped by
 * what each component is for. The exact reason a component is unavailable is
 * always shown next to its badge rather than hidden in a tooltip, because on a
 * touch device a tooltip is unreachable.
 */

import { useMemo } from "react";
import { Link } from "react-router-dom";
import { useBackendHealth } from "../hooks/useHealth";
import { useSystem } from "../hooks/useSystem";
import { useModels } from "../hooks/useModels";
import { useCapabilities } from "../hooks/useCapabilities";
import { useSamples } from "../hooks/useSamples";
import { Badge } from "../components/common/Badge";
import {
  EmptyState,
  ErrorState,
  SkeletonCard,
  SkeletonGrid,
} from "../components/common/States";
import { OfflineBanner } from "../components/status/OfflineBanner";
import { ReadinessPanel } from "../components/demo/ReadinessPanel";
import { IconArrowRight } from "../components/common/Icons";
import { buildReadiness } from "../utils/demo";
import { componentStatePresentation } from "../utils/status";
import { groupModels } from "../utils/modelGroups";
import { formatBytes, formatDuration } from "../utils/format";

export function DashboardPage() {
  const { status, health } = useBackendHealth();
  const system = useSystem();
  const models = useModels();
  const capabilities = useCapabilities();
  const samples = useSamples();

  const offline = status === "offline";

  const readinessLoading =
    samples.loading || capabilities.loading || models.loading;

  const readiness = useMemo(
    () =>
      buildReadiness({
        sample: samples.data?.samples[0] ?? null,
        capabilities: capabilities.data,
        models: models.data,
      }),
    [samples.data, capabilities.data, models.data],
  );

  return (
    <div className="page">
      <header className="page-header">
        <h1 className="page-title">System</h1>
        <p className="page-subtitle">
          Runtime environment and the availability of optional components on
          this machine.
        </p>
      </header>

      {offline && <OfflineBanner />}

      <section className="section">
        <div className="section-heading">
          <h2>Backend</h2>
        </div>
        <div className="stat-grid">
          <div className="card">
            <div className="stat-value stat-value-sm">
              {status === "online" ? "Online" : status === "offline" ? "Offline" : "Checking"}
            </div>
            <div className="stat-label">Status</div>
          </div>
          <div className="card">
            <div className="stat-value stat-value-sm">
              {health?.api_version ?? "—"}
            </div>
            <div className="stat-label">API version</div>
          </div>
          <div className="card">
            <div className="stat-value stat-value-sm">
              {health ? formatDuration(health.uptime_seconds) : "—"}
            </div>
            <div className="stat-label">Uptime</div>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="section-heading">
          <div>
            <h2>Compute</h2>
            <p>The device the pipeline will use for inference.</p>
          </div>
        </div>
        {offline ? null : system.error ? (
          <ErrorState error={system.error} onRetry={system.reload} />
        ) : system.loading || !system.data ? (
          <SkeletonGrid count={4} lines={2} />
        ) : (
          <div className="stat-grid">
            <StatCard label="Device" value={system.data.device.toUpperCase()} />
            <StatCard
              label="CUDA"
              value={system.data.cuda_available ? "Available" : "Not available"}
              note={system.data.cuda_version ?? undefined}
            />
            <StatCard
              label="GPU"
              value={system.data.gpu_name ?? "None detected"}
              note={
                system.data.gpu_count > 0
                  ? `${system.data.gpu_count} device(s)`
                  : undefined
              }
            />
            <StatCard
              label="GPU memory"
              value={
                system.data.gpu_memory_total_mb
                  ? formatBytes(system.data.gpu_memory_total_mb * 1024 * 1024)
                  : "—"
              }
            />
            <StatCard
              label="PyTorch"
              value={system.data.torch_version ?? "—"}
            />
            <StatCard
              label="Python"
              value={system.data.python_version}
              note={system.data.python_implementation}
            />
          </div>
        )}
      </section>

      <section className="section">
        <div className="section-heading">
          <div>
            <h2>Media tooling</h2>
            <p>External tools the pipeline shells out to.</p>
          </div>
        </div>
        {offline ? null : system.error ? null : system.loading || !system.data ? (
          <SkeletonGrid count={2} lines={2} />
        ) : (
          <div className="card-grid card-grid-wide">
            <div className="card">
              <div className="card-title">FFmpeg</div>
              <div className="card-subtitle">
                Decoding, encoding and audio extraction.
              </div>
              <div style={{ marginTop: "var(--space-3)" }}>
                <Badge tone={system.data.ffmpeg_available ? "ok" : "danger"} dot>
                  {system.data.ffmpeg_available ? "Available" : "Not installed"}
                </Badge>
                {system.data.ffmpeg_version && (
                  <div className="stat-note">{system.data.ffmpeg_version}</div>
                )}
              </div>
            </div>
            <div className="card">
              <div className="card-title">Jobs directory</div>
              <div className="card-subtitle">
                Where durable job manifests and artifacts are written.
              </div>
              <div style={{ marginTop: "var(--space-3)" }}>
                <Badge tone={system.data.jobs_dir_writable ? "ok" : "danger"} dot>
                  {system.data.jobs_dir_writable ? "Writable" : "Not writable"}
                </Badge>
              </div>
            </div>
          </div>
        )}
      </section>

      <section className="section">
        <div className="section-heading">
          <div>
            <h2>AI components</h2>
            <p>
              Grouped by purpose. The reason is shown for anything not available
              here.
            </p>
          </div>
        </div>
        {offline ? null : models.error ? (
          <ErrorState error={models.error} onRetry={models.reload} />
        ) : models.loading || !models.data ? (
          <SkeletonCard lines={6} />
        ) : models.data.length === 0 ? (
          <EmptyState
            title="No components reported"
            body="The backend did not report any optional components."
          />
        ) : (
          <div className="card">
            {groupModels(models.data).map((group) => (
              <div className="model-group" key={group.id}>
                <div className="model-group-title">
                  <span>{group.title}</span>
                  <span className="card-subtitle">
                    {group.models.filter((m) => m.available).length}/
                    {group.models.length} available
                  </span>
                </div>
                {group.models.map((model) => (
                  <ModelRow key={model.name} model={model} />
                ))}
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="section">
        <div className="section-heading">
          <div>
            <h2>Demo readiness</h2>
            <p>
              Whether the guided demo can run here, and what is missing if not.
            </p>
          </div>
        </div>
        {offline ? null : readinessLoading ? (
          <SkeletonCard lines={4} />
        ) : (
          <>
            <ReadinessPanel readiness={readiness} />
            <div className="btn-row" style={{ marginTop: "var(--space-4)" }}>
              <Link to="/demo" className="btn btn-primary">
                {readiness.ready ? "Open the demo" : "See demo status"}
                <IconArrowRight size={16} />
              </Link>
            </div>
          </>
        )}
      </section>

      <section className="section">
        <div className="section-heading">
          <h2>Explore</h2>
        </div>
        <div className="btn-row">
          <Link to="/demo" className="btn btn-secondary">
            Demo
          </Link>
          <Link to="/languages" className="btn btn-secondary">
            Language explorer
          </Link>
          <Link to="/jobs" className="btn btn-secondary">
            Jobs
          </Link>
        </div>
      </section>
    </div>
  );
}

function StatCard({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note?: string;
}) {
  return (
    <div className="card">
      <div className="stat-value stat-value-sm">{value}</div>
      <div className="stat-label">{label}</div>
      {note && <div className="stat-note">{note}</div>}
    </div>
  );
}

function ModelRow({ model }: { model: import("../api/types").ComponentStatus }) {
  const presentation = componentStatePresentation(model.state);
  return (
    <div className="model-row">
      <div className="model-row-main">
        <div className="model-row-label">{model.label}</div>
        <div className="model-row-reason">{model.reason}</div>
      </div>
      <Badge tone={presentation.tone} dot>
        {presentation.label}
      </Badge>
    </div>
  );
}
