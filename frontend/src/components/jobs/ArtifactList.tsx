/**
 * The artifacts a job produced, as a list of downloads.
 *
 * Every artifact the API offers is listed, including the ones this render did
 * not produce — an unavailable artifact shows the backend's own `reason` rather
 * than being hidden, so a user learns that lip-sync exists but was not enabled
 * instead of wondering where it went. `name` is the only handle used; it is the
 * server-side allow-list key, never a path.
 */

import { Badge } from "../common/Badge";
import { IconDownload } from "../common/Icons";
import { formatBytes } from "../../utils/format";
import { orderArtifacts } from "../../utils/dubbing";
import type { ArtifactDescriptor } from "../../api/types";

export function ArtifactList({ artifacts }: { artifacts: ArtifactDescriptor[] }) {
  if (artifacts.length === 0) {
    return <p className="card-subtitle">This job produced no artifacts.</p>;
  }

  return (
    <ul className="artifact-list">
      {orderArtifacts(artifacts).map((artifact) => (
        <li
          key={artifact.name}
          className={artifact.available ? "artifact" : "artifact artifact-off"}
        >
          <div className="artifact-main">
            <span className="artifact-label">
              {artifact.label}
              {artifact.available ? (
                <Badge tone="ok">Ready</Badge>
              ) : (
                <Badge tone="neutral">Not produced</Badge>
              )}
            </span>
            <span className="artifact-desc">
              {artifact.available ? artifact.description : artifact.reason ?? artifact.description}
            </span>
          </div>

          {artifact.available && artifact.download_url ? (
            <a
              className="btn btn-secondary"
              href={artifact.download_url}
              download={artifact.filename ?? undefined}
            >
              <IconDownload size={16} />
              {artifact.size_bytes !== null
                ? formatBytes(artifact.size_bytes)
                : "Download"}
            </a>
          ) : (
            <span className="artifact-unavailable">Unavailable</span>
          )}
        </li>
      ))}
    </ul>
  );
}
