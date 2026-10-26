/**
 * Language filtering and route lookup.
 *
 * Pure functions over the `/api/languages` response, so the filtering rules are
 * unit-testable without rendering anything. Every predicate reads a field the
 * backend already computed; nothing here re-derives support from a hard-coded
 * list.
 */

import type { LanguageInfo, RouteInfo } from "../api/types";

export type LanguageFilterId =
  | "all"
  | "indic"
  | "priority"
  | "source"
  | "target"
  | "cloning_arch"
  | "cloning_local";

export interface LanguageFilter {
  id: LanguageFilterId;
  label: string;
  test: (language: LanguageInfo) => boolean;
}

/**
 * The filters offered, each backed by a field the API provides. `cloning_arch`
 * and `cloning_local` are deliberately separate: the first means the project can
 * clone this language, the second means this machine currently can.
 */
export const LANGUAGE_FILTERS: LanguageFilter[] = [
  { id: "all", label: "All", test: () => true },
  { id: "indic", label: "Indian", test: (l) => l.indic },
  { id: "priority", label: "Priority Indic", test: (l) => l.priority_indic },
  { id: "source", label: "Source supported", test: (l) => l.source },
  { id: "target", label: "Target supported", test: (l) => l.target },
  {
    id: "cloning_arch",
    label: "Voice cloning supported",
    test: (l) => l.cloning_supported_by_architecture,
  },
  {
    id: "cloning_local",
    label: "Cloning available locally",
    test: (l) => l.cloning_model_available_on_this_machine,
  },
];

/**
 * Apply a search term and a filter to the language list.
 *
 * Matching is case-insensitive across the code, the name and every alias, so
 * "hindi", "hi" and "hin" all find Hindi. Priority Indian languages are sorted
 * first, then alphabetical, which makes the Indian set easy to scan.
 */
export function filterLanguages(
  languages: LanguageInfo[],
  options: { search?: string; filter?: LanguageFilterId } = {},
): LanguageInfo[] {
  const search = options.search?.trim().toLowerCase() ?? "";
  const filter =
    LANGUAGE_FILTERS.find((f) => f.id === options.filter) ?? LANGUAGE_FILTERS[0];

  const matched = languages.filter((language) => {
    if (!filter.test(language)) return false;
    if (!search) return true;
    const haystack = [language.code, language.name, ...language.aliases]
      .join(" ")
      .toLowerCase();
    return haystack.includes(search);
  });

  return matched.sort((a, b) => {
    if (a.priority_indic !== b.priority_indic) {
      return a.priority_indic ? -1 : 1;
    }
    return a.name.localeCompare(b.name);
  });
}

export interface RouteLookup {
  /** The route the backend reports, if one exists for this pair. */
  route: RouteInfo | null;
  /**
   * True when the matrix was requested but contains no entry for the pair. The
   * matrix is opt-in, so a missing route is only meaningful when it was asked
   * for.
   */
  matrixLoaded: boolean;
}

/** Find the route for a source/target pair in the (opt-in) route matrix. */
export function findRoute(
  routes: RouteInfo[],
  source: string,
  target: string,
  matrixLoaded: boolean,
): RouteLookup {
  if (!matrixLoaded) return { route: null, matrixLoaded: false };
  const route =
    routes.find((r) => r.source === source && r.target === target) ?? null;
  return { route, matrixLoaded: true };
}

/**
 * Resolve a job's language field to a short uppercase code for display.
 *
 * The manifest is written over time, so old jobs stored a display name
 * ("English") and newer ones a code ("en"). Both are still on disk, and
 * rendering them raw makes the same field look inconsistent from one row to the
 * next. This maps either form through the API's own language list, so a name
 * resolves to its code and a code stays itself.
 *
 * Anything unrecognised is passed through uppercased rather than hidden: a
 * language the list does not know about is still worth showing.
 */
export function displayLanguage(
  value: string | null | undefined,
  languages: LanguageInfo[],
  fallback = "—",
): string {
  if (!value) return fallback;
  const needle = value.trim().toLowerCase();
  if (!needle) return fallback;

  const match = languages.find(
    (language) =>
      language.code.toLowerCase() === needle ||
      language.name.toLowerCase() === needle ||
      language.aliases.some((alias) => alias.toLowerCase() === needle),
  );

  return (match?.code ?? value).trim().toUpperCase();
}
