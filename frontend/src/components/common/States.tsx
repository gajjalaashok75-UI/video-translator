/**
 * Reusable loading, empty and error blocks.
 *
 * The four situations the brief calls out — backend offline, request failed, no
 * data, feature unavailable — are four distinct components, because they need
 * four different messages. A single generic "something went wrong" would hide
 * the one piece of information (what to do about it) that the user needs.
 */

import type { ReactNode } from "react";
import { ApiError } from "../../api/client";
import { cx } from "../../utils/format";
import { IconAlert, IconInbox, IconRefresh } from "./Icons";

// --- Loading ----------------------------------------------------------------

/**
 * A skeleton block shaped like the content it replaces. Preferred over a
 * full-page spinner because it preserves layout and shows what is coming.
 */
export function SkeletonCard({ lines = 3 }: { lines?: number }) {
  return (
    <div className="card" aria-hidden="true">
      <div className="skeleton skeleton-title" />
      {Array.from({ length: lines }).map((_, index) => (
        <div key={index} className="skeleton skeleton-line" />
      ))}
    </div>
  );
}

export function SkeletonGrid({ count = 6, lines = 3 }: { count?: number; lines?: number }) {
  return (
    <div className="card-grid" aria-hidden="true">
      {Array.from({ length: count }).map((_, index) => (
        <SkeletonCard key={index} lines={lines} />
      ))}
    </div>
  );
}

/** A compact inline spinner with a text label, for small regions. */
export function InlineSpinner({ label }: { label: string }) {
  return (
    <span className="loading-inline" role="status">
      <span className="spinner" aria-hidden="true" />
      {label}
    </span>
  );
}

// --- Empty ------------------------------------------------------------------

export function EmptyState({
  title,
  body,
  icon,
  action,
}: {
  title: string;
  body: string;
  icon?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="state-block">
      <span className="state-icon">{icon ?? <IconInbox size={28} />}</span>
      <span className="state-title">{title}</span>
      <span className="state-body">{body}</span>
      {action}
    </div>
  );
}

// --- Error ------------------------------------------------------------------

/**
 * Turn any thrown value into a title and body.
 *
 * Offline is called out separately: it is the one failure whose remedy is not
 * "try again" but "start the server".
 */
export function describeError(error: ApiError | Error | null): {
  title: string;
  body: string;
  canRetry: boolean;
} {
  if (error instanceof ApiError) {
    if (error.isOffline) {
      return {
        title: "Backend unavailable",
        body: "Start the Python API server and try again.",
        canRetry: true,
      };
    }
    return { title: "Request failed", body: error.message, canRetry: true };
  }
  if (error) {
    return { title: "Request failed", body: error.message, canRetry: true };
  }
  return { title: "Request failed", body: "An unexpected error occurred.", canRetry: true };
}

export function ErrorState({
  error,
  onRetry,
  className,
}: {
  error: ApiError | Error | null;
  onRetry?: () => void;
  className?: string;
}) {
  const { title, body, canRetry } = describeError(error);
  return (
    <div className={cx("state-block", "state-error", className)} role="alert">
      <span className="state-icon">
        <IconAlert size={28} />
      </span>
      <span className="state-title">{title}</span>
      <span className="state-body">{body}</span>
      {onRetry && canRetry && (
        <button type="button" className="btn btn-secondary" onClick={onRetry}>
          <IconRefresh size={16} />
          Retry
        </button>
      )}
    </div>
  );
}

/**
 * A feature that exists in the product but is not usable in this installation.
 * Distinct from an error: nothing failed, the capability simply is not present.
 */
export function UnavailableState({
  title,
  body,
  reason,
}: {
  title: string;
  body: string;
  reason?: string | null;
}) {
  return (
    <div className="state-block">
      <span className="state-icon">
        <IconAlert size={28} />
      </span>
      <span className="state-title">{title}</span>
      <span className="state-body">{body}</span>
      {reason && <span className="state-body">Reason: {reason}</span>}
    </div>
  );
}

