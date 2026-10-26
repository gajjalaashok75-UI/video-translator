/**
 * Presentation helpers. Pure formatting only — no backend semantics live here.
 */

/** Format a byte count as a short human string. */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unit]}`;
}

/** Format an uptime in seconds as "2h 14m" / "45s". */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  const total = Math.max(0, Math.floor(seconds));
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${total % 60}s`;
  return `${total}s`;
}

/**
 * Format a Unix timestamp (seconds) as a relative time such as "3m ago".
 * Falls back to an absolute date once the gap is over a week.
 */
export function formatRelativeTime(
  epochSeconds: number | null | undefined,
  now: number = Date.now(),
): string {
  if (epochSeconds === null || epochSeconds === undefined) return "—";
  const ms = epochSeconds * 1000;
  const deltaSeconds = Math.round((now - ms) / 1000);

  if (deltaSeconds < 0) return "just now";
  if (deltaSeconds < 45) return "just now";
  if (deltaSeconds < 90) return "1m ago";
  const minutes = Math.round(deltaSeconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(ms).toLocaleDateString();
}

/** Format a fraction 0..1 as a whole-number percentage. */
export function formatPercent(fraction: number | null | undefined): string {
  if (fraction === null || fraction === undefined) return "—";
  return `${Math.round(Math.min(1, Math.max(0, fraction)) * 100)}%`;
}

/** Join class names, dropping falsy values. */
export function cx(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}
