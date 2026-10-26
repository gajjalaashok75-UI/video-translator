/**
 * A status badge.
 *
 * Every badge shows a text label as well as a color, so status is never carried
 * by color alone (which fails for color-blind users and in grayscale).
 */

import type { ReactNode } from "react";
import { cx } from "../../utils/format";
import type { BadgeTone } from "../../utils/status";

interface BadgeProps {
  tone?: BadgeTone;
  children: ReactNode;
  /** Show a leading dot; decorative, the label carries the meaning. */
  dot?: boolean;
}

export function Badge({ tone = "neutral", children, dot = false }: BadgeProps) {
  const toneClass = tone === "neutral" ? "" : `badge-${tone}`;
  return (
    <span className={cx("badge", toneClass)}>
      {dot && <span className="badge-dot" aria-hidden="true" />}
      {children}
    </span>
  );
}
