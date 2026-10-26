/**
 * The log panel for a running or finished job.
 *
 * Two sources, shown as one: the *live* lines that arrived over the event stream
 * during this visit, and the *durable* log read from `GET /api/jobs/{id}/logs`,
 * which is the complete record and survives a restart. When the durable log is
 * present it is the whole story, so it is what is shown; the live lines fill the
 * gap before it loads, so the panel is never empty while a job is clearly
 * running.
 *
 * Lines are already redacted server-side; nothing here re-filters them.
 */

import { useEffect, useRef } from "react";
import { cx } from "../../utils/format";
import { IconTerminal } from "../common/Icons";

export function LogPanel({
  liveLines,
  durableLines,
  dropped,
  truncated,
}: {
  liveLines: string[];
  durableLines: string[] | null;
  /** Live lines dropped from the bounded tail, reported so nothing is silent. */
  dropped: number;
  /** True when the durable log shortened a line. */
  truncated: boolean;
}) {
  const scrollerRef = useRef<HTMLDivElement>(null);
  const usingDurable = durableLines !== null && durableLines.length > 0;
  const lines = usingDurable ? durableLines : liveLines;

  // Follow the tail, but only when the user is already at the bottom, so
  // scrolling up to read does not fight the incoming lines.
  useEffect(() => {
    const node = scrollerRef.current;
    if (!node) return;
    const nearBottom =
      node.scrollHeight - node.scrollTop - node.clientHeight < 40;
    if (nearBottom) node.scrollTop = node.scrollHeight;
  }, [lines.length]);

  return (
    <div className="log-panel">
      <div className="log-head">
        <span className="log-title">
          <IconTerminal size={16} />
          Log
        </span>
        <span className="log-source">
          {usingDurable ? "Durable log" : "Live stream"}
        </span>
      </div>

      <div className="log-scroller" ref={scrollerRef}>
        {lines.length === 0 ? (
          <p className="log-empty">No log output yet.</p>
        ) : (
          <pre className="log-body">
            {lines.map((line, index) => (
              <div className={cx("log-line")} key={index}>
                {line}
              </div>
            ))}
          </pre>
        )}
      </div>

      {(dropped > 0 || truncated) && (
        <p className="log-note">
          {dropped > 0
            ? `${dropped} earlier line${dropped === 1 ? "" : "s"} not shown; the durable log has the full record.`
            : "Some lines were shortened; the full text is in the job's log file."}
        </p>
      )}
    </div>
  );
}
