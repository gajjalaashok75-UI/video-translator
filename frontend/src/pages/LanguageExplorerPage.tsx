/**
 * Language explorer.
 *
 * Two things this page must not do: hard-code which languages are supported,
 * and collapse "the project supports this" into "this machine can do this". The
 * first is why every row comes from `/api/languages`; the second is why the
 * cloning badge has two lines — architecture support and local availability.
 */

import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useLanguages } from "../hooks/useLanguages";
import { useCapabilities } from "../hooks/useCapabilities";
import { Badge } from "../components/common/Badge";
import {
  EmptyState,
  ErrorState,
  SkeletonGrid,
} from "../components/common/States";
import { OfflineBanner } from "../components/status/OfflineBanner";
import { useBackendHealth } from "../hooks/useHealth";
import {
  LANGUAGE_FILTERS,
  filterLanguages,
  findRoute,
  type LanguageFilterId,
} from "../utils/languages";
import { IconGlobe, IconSearch, IconSparkle } from "../components/common/Icons";
import type { LanguageInfo } from "../api/types";

export function LanguageExplorerPage() {
  const { status } = useBackendHealth();
  const languages = useLanguages();
  // The route matrix is opt-in server-side; only this page needs it.
  const capabilities = useCapabilities(true);

  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<LanguageFilterId>("all");

  const offline = status === "offline";

  const results = useMemo(
    () =>
      languages.data
        ? filterLanguages(languages.data.languages, { search, filter })
        : [],
    [languages.data, search, filter],
  );

  return (
    <div className="page">
      <header className="page-header page-header-split">
        <div>
          <h1 className="page-title">Languages</h1>
          <p className="page-subtitle">
            Every language the pipeline can transcribe or dub, with what this
            installation can do for each. Support comes from the backend, not from
            this page.
          </p>
        </div>
        <Link to="/demo" className="btn btn-secondary">
          <IconSparkle size={16} />
          Try a demo
        </Link>
      </header>

      {offline && <OfflineBanner />}

      {!offline && capabilities.data && (
        <RouteExplorer
          routes={capabilities.data.routes}
          matrixLoaded={capabilities.data.routes.length > 0}
          sources={capabilities.data.sources}
          targets={capabilities.data.targets}
          languages={languages.data?.languages ?? []}
        />
      )}

      <section className="section">
        <div className="section-heading">
          <h2>All languages</h2>
        </div>

        <div className="search-field">
          <span className="search-icon">
            <IconSearch size={16} />
          </span>
          <label className="sr-only" htmlFor="language-search">
            Search languages
          </label>
          <input
            id="language-search"
            className="search-input"
            type="search"
            placeholder="Search by name or code…"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>

        <div className="filter-bar" role="group" aria-label="Language filters">
          {LANGUAGE_FILTERS.map((item) => (
            <button
              key={item.id}
              type="button"
              className="filter-chip"
              aria-pressed={filter === item.id}
              onClick={() => setFilter(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>

        {offline ? null : languages.error ? (
          <ErrorState error={languages.error} onRetry={languages.reload} />
        ) : languages.loading || !languages.data ? (
          <SkeletonGrid count={8} lines={3} />
        ) : results.length === 0 ? (
          <EmptyState
            title="No languages match these filters"
            body="Try a different search term or clear the filters to see every language."
          />
        ) : (
          <>
            <p className="result-count" role="status">
              {results.length} of {languages.data.languages.length} languages
            </p>
            <div className="lang-grid">
              {results.map((language) => (
                <LanguageCard key={language.code} language={language} />
              ))}
            </div>
          </>
        )}
      </section>
    </div>
  );
}

/**
 * One language, with the fields that matter for dubbing. Cloning is shown as
 * two separate lines so the architecture/local distinction stays visible.
 */
function LanguageCard({ language }: { language: LanguageInfo }) {
  return (
    <article className="lang-card">
      <div className="lang-card-head">
        <div>
          <div className="lang-name">{language.name}</div>
          <div className="lang-meta">
            {language.code}
            {language.script ? ` · ${language.script}` : ""}
            {language.flores ? ` · ${language.flores}` : ""}
          </div>
        </div>
        {language.priority_indic && (
          <Badge tone="accent" dot>
            Priority
          </Badge>
        )}
      </div>

      <div className="lang-badges">
        {language.source && <Badge tone="info">Source</Badge>}
        {language.target && <Badge tone="info">Target</Badge>}
        {language.indic && <Badge>Indian</Badge>}
        {language.edge_supported && <Badge>Edge TTS</Badge>}
        {language.piper_supported && <Badge>Piper TTS</Badge>}
      </div>

      {language.cloning_supported_by_architecture && (
        <div className="dl" style={{ marginTop: "var(--space-3)" }}>
          <div className="dl-row">
            <span className="dl-label">Cloning (architecture)</span>
            <span className="dl-value">
              <Badge tone="ok">Supported</Badge>
            </span>
          </div>
          <div className="dl-row">
            <span className="dl-label">Cloning (this machine)</span>
            <span className="dl-value">
              {language.cloning_model_available_on_this_machine ? (
                <Badge tone="ok">Available</Badge>
              ) : (
                <Badge tone="warn">Model required</Badge>
              )}
            </span>
          </div>
          {language.cloning_engine && (
            <div className="dl-row">
              <span className="dl-label">Engine</span>
              <span className="dl-value dl-value-mono">
                {language.cloning_engine}
              </span>
            </div>
          )}
        </div>
      )}
    </article>
  );
}

/**
 * Route explorer: pick a source and a target and see what the backend says
 * about the pair. Reads the opt-in route matrix; when the matrix is empty (the
 * backend did not compute it) it says so rather than implying no route exists.
 */
function RouteExplorer({
  routes,
  matrixLoaded,
  sources,
  targets,
  languages,
}: {
  routes: import("../api/types").RouteInfo[];
  matrixLoaded: boolean;
  sources: string[];
  targets: string[];
  languages: LanguageInfo[];
}) {
  const nameFor = (code: string) =>
    languages.find((l) => l.code === code)?.name ?? code;

  const [source, setSource] = useState(sources[0] ?? "");
  const [target, setTarget] = useState(targets[0] ?? "");

  const lookup = findRoute(routes, source, target, matrixLoaded);

  return (
    <section className="section">
      <div className="section-heading">
        <div>
          <h2>Route explorer</h2>
          <p>What the backend reports for a source and target pair.</p>
        </div>
      </div>

      <div className="card">
        <div className="route-picker">
          <div className="select-field">
            <label htmlFor="route-source">From</label>
            <select
              id="route-source"
              className="select-input"
              value={source}
              onChange={(event) => setSource(event.target.value)}
            >
              {sources.map((code) => (
                <option key={code} value={code}>
                  {nameFor(code)}
                </option>
              ))}
            </select>
          </div>
          <span className="route-arrow" aria-hidden="true">
            <IconGlobe size={18} />
          </span>
          <div className="select-field">
            <label htmlFor="route-target">To</label>
            <select
              id="route-target"
              className="select-input"
              value={target}
              onChange={(event) => setTarget(event.target.value)}
            >
              {targets.map((code) => (
                <option key={code} value={code}>
                  {nameFor(code)}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="dl" style={{ marginTop: "var(--space-4)" }}>
          {lookup.route ? (
            <>
              <div className="dl-row">
                <span className="dl-label">Translation route</span>
                <span className="dl-value">
                  <Badge tone="ok">Supported</Badge>
                </span>
              </div>
              {lookup.route.backend && (
                <div className="dl-row">
                  <span className="dl-label">Preferred backend</span>
                  <span className="dl-value dl-value-mono">
                    {lookup.route.backend}
                  </span>
                </div>
              )}
              {lookup.route.pivots.length > 0 && (
                <div className="dl-row">
                  <span className="dl-label">Via</span>
                  <span className="dl-value dl-value-mono">
                    {lookup.route.pivots.map(nameFor).join(" → ")}
                  </span>
                </div>
              )}
              {lookup.route.is_passthrough && (
                <div className="dl-row">
                  <span className="dl-label">Note</span>
                  <span className="dl-value">Pass-through, no translation</span>
                </div>
              )}
            </>
          ) : (
            <div className="dl-row">
              <span className="dl-label">Translation route</span>
              <span className="dl-value">
                {matrixLoaded ? (
                  <Badge tone="warn">Not reported</Badge>
                ) : (
                  <Badge tone="neutral">Matrix not loaded</Badge>
                )}
              </span>
            </div>
          )}
        </div>

        {!matrixLoaded && (
          <p className="stat-note">
            The backend did not return a route matrix for this request. Support
            for a specific pair is not implied either way.
          </p>
        )}
      </div>
    </section>
  );
}
