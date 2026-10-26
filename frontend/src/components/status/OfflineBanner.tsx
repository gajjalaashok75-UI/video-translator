/**
 * The "backend unavailable" banner.
 *
 * Shown in place of page content when the API cannot be reached, so the user
 * gets an instruction ("start the Python API server") rather than a stack of
 * failed requests. Retry re-probes liveness directly.
 */

import { useBackendHealth } from "../../hooks/useHealth";
import { IconAlert, IconRefresh } from "../common/Icons";

export function OfflineBanner() {
  const { refresh } = useBackendHealth();

  return (
    <div className="offline-banner" role="alert">
      <span className="offline-banner-icon">
        <IconAlert size={22} />
      </span>
      <div>
        <div className="offline-banner-title">Backend unavailable</div>
        <p className="offline-banner-body">
          Start the Python API server and try again. From the repository root, run{" "}
          <code>uvicorn src.api.main:app --port 8000</code>.
        </p>
        <button type="button" className="btn btn-secondary" onClick={refresh}>
          <IconRefresh size={16} />
          Retry
        </button>
      </div>
    </div>
  );
}
