import type { NavigationDestination, RiskTier, SafetyState } from "../uc07/types";
import {
  BIN_EDGES,
  DonutChart,
  HorizontalBarChart,
  NAV_META,
  RISK_META,
  SAFETY_META,
  type CategoryDatum,
} from "../uc07/components/AnalyticsCharts";
import { NAVIGATION_DESTINATION_COLOR } from "../uc07/navigationDisplay";
import "../uc07/components/AnalyticsCharts.css";

/**
 * Saved-population equivalent of frontend/src/uc07/components/
 * AnalyticsCharts.tsx's RiskDonut/SafetyDonut/NavigationBar/
 * ProbabilityHistogram -- same visual chart shell (DonutChart/
 * HorizontalBarChart, imported directly, never reimplemented), but fed
 * with counts the BACKEND already aggregated via SQL GROUP BY
 * (backend/db/repositories/populations.py's get_population_summary)
 * instead of a client-side tally over every member's decision. This is
 * what lets a saved population with 10,000+ members show the same
 * analytics without ever fetching more than ~20 aggregate numbers.
 */

function totalOf(counts: Record<string, number>): number {
  return Object.values(counts).reduce((sum, n) => sum + n, 0);
}

export function SavedRiskDonut({
  tierCounts,
  activeTier,
  onSelectTier,
}: {
  tierCounts: Record<string, number>;
  activeTier: RiskTier | null;
  onSelectTier: (tier: RiskTier | null) => void;
}) {
  const total = totalOf(tierCounts);
  const categories: CategoryDatum[] = RISK_META.map((r) => ({
    key: r.key,
    label: r.label,
    count: tierCounts[r.key] ?? 0,
    color: `var(--risk-${r.key.toLowerCase()})`,
  }));

  return (
    <DonutChart
      title="Risk Distribution"
      accent="risk"
      categories={categories}
      total={total}
      centerLabel="Click a segment to filter the member table by risk tier."
      activeKey={activeTier}
      onSelect={(key) => onSelectTier(activeTier === key ? null : (key as RiskTier))}
    />
  );
}

export function SavedSafetyDonut({
  safetyCounts,
  activeSafety,
  onSelectSafety,
}: {
  safetyCounts: Record<string, number>;
  activeSafety: SafetyState | null;
  onSelectSafety: (state: SafetyState | null) => void;
}) {
  const total = totalOf(safetyCounts);
  const overrideCount = safetyCounts.OVERRIDE ?? 0;
  const categories: CategoryDatum[] = SAFETY_META.map((s) => ({
    key: s.key,
    label: s.label,
    count: safetyCounts[s.key] ?? 0,
    color: `var(--safety-${s.key.toLowerCase()})`,
  }));

  return (
    <DonutChart
      title="Safety Distribution"
      accent="safety"
      categories={categories}
      total={total}
      centerLabel="Click a segment to filter the member table by safety state."
      activeKey={activeSafety}
      onSelect={(key) => onSelectSafety(activeSafety === key ? null : (key as SafetyState))}
      calloutText={
        overrideCount > 0
          ? `⚠ ${overrideCount} member${overrideCount === 1 ? "" : "s"} in OVERRIDE -- safety has final authority regardless of risk or navigation.`
          : undefined
      }
    />
  );
}

export function SavedNavigationBar({
  navigationCounts,
  activeDestination,
  onSelectDestination,
}: {
  navigationCounts: Record<string, number>;
  activeDestination: NavigationDestination | null;
  onSelectDestination: (destination: NavigationDestination | null) => void;
}) {
  const total = totalOf(navigationCounts);
  // The backend buckets a null destination (only possible when
  // safety.state === "OVERRIDE") under the "NONE" key -- merge it into
  // NO_PROACTIVE_NAVIGATION here so this matches the live/CSV-pathway
  // chart's semantics exactly (AnalyticsCharts.tsx's NavigationBar keys
  // a null destination as "NO_PROACTIVE_NAVIGATION" too).
  const noneCount = navigationCounts.NONE ?? 0;
  const categories: CategoryDatum[] = NAV_META.map((n) => ({
    key: n.key,
    label: n.label,
    count: (navigationCounts[n.key] ?? 0) + (n.key === "NO_PROACTIVE_NAVIGATION" ? noneCount : 0),
    color: NAVIGATION_DESTINATION_COLOR[n.key],
  }));

  return (
    <HorizontalBarChart
      title="Navigation Distribution"
      accent="navigation"
      categories={categories}
      total={total}
      activeKey={activeDestination}
      onSelect={(key) => onSelectDestination(activeDestination === key ? null : (key as NavigationDestination))}
    />
  );
}

export function SavedProbabilityHistogram({
  bins,
  moderateThreshold,
  highThreshold,
  activeBin,
  onSelectBin,
}: {
  bins: number[];
  moderateThreshold: number | null;
  highThreshold: number | null;
  activeBin: string | null;
  onSelectBin: (bin: { min: number; max: number | null } | null) => void;
}) {
  // Presentational-only bucket labels, matching AnalyticsCharts.tsx's
  // ProbabilityHistogram exactly (same BIN_EDGES) -- rendered as the
  // same HorizontalBarChart-adjacent legend list rather than a second
  // recharts implementation, since the KPI-card treatment there is
  // tightly coupled to the live/client-computed histogram's own markup.
  const total = bins.reduce((sum, n) => sum + n, 0);
  const categories: CategoryDatum[] = BIN_EDGES.map((edge, i) => ({
    key: String(edge),
    label: i === BIN_EDGES.length - 1 ? `${edge}%+` : `${edge}–${BIN_EDGES[i + 1] - 1}%`,
    count: bins[i] ?? 0,
    color: "var(--accent)",
  }));

  return (
    <div className="analytics-card analytics-card--probability">
      <div className="analytics-card__title-row">
        <h3 className="analytics-card__title">Probability Distribution</h3>
      </div>
      {total === 0 ? (
        <p className="analytics-card__empty">No members in this population.</p>
      ) : (
        <>
          <ul className="analytics-legend">
            {categories.map((c, i) => {
              const isActive = activeBin === c.key;
              const isDimmed = activeBin !== null && !isActive;
              return (
                <li key={c.key}>
                  <button
                    type="button"
                    className={`analytics-legend__item${isActive ? " analytics-legend__item--active" : ""}${isDimmed ? " analytics-legend__item--dimmed" : ""}`}
                    onClick={() => {
                      const max = i === BIN_EDGES.length - 1 ? null : BIN_EDGES[i + 1];
                      onSelectBin(isActive ? null : { min: BIN_EDGES[i], max });
                    }}
                    aria-pressed={isActive}
                  >
                    <span className="analytics-legend__swatch" style={{ background: c.color }} aria-hidden="true" />
                    <span className="analytics-legend__label" aria-hidden="true">
                      {c.label}
                    </span>
                    <span className="analytics-legend__count tabular" aria-hidden="true">
                      {c.count}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
          {(moderateThreshold !== null || highThreshold !== null) && (
            <p className="analytics-card__hint">
              {moderateThreshold !== null && `Moderate threshold ≈ ${(moderateThreshold * 100).toFixed(1)}%`}
              {moderateThreshold !== null && highThreshold !== null && " · "}
              {highThreshold !== null && `High threshold ≈ ${(highThreshold * 100).toFixed(1)}%`}
            </p>
          )}
        </>
      )}
    </div>
  );
}
