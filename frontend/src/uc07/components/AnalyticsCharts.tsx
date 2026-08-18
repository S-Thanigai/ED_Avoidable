import { useId, useMemo } from "react";
import type { ReactElement } from "react";
import { Bar, BarChart, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { FinalUC07Decision, NavigationDestination, RiskTier, SafetyState } from "../types";
import {
  NAVIGATION_DESTINATION_COLOR,
  NAVIGATION_DESTINATION_LABEL,
  NAVIGATION_DESTINATION_ORDER,
} from "../navigationDisplay";
import "./AnalyticsCharts.css";

/**
 * Part 4-9 -- population-level analytics, modeled on Power BI/Tableau
 * usability (click a segment to filter, hover for a tooltip, always-
 * visible legend) while staying restrained/healthcare-appropriate.
 *
 * Hard invariant carried over from every other view in this app: these
 * charts NEVER compute a risk tier, navigation destination, safety
 * state, or probability -- every number here is a tally over the
 * FinalUC07Decision[] population the backend already returned. Clicking
 * a segment only calls back into the existing filters/updateFilters
 * mechanism (tableState.ts) that MemberFilters' dropdowns already use --
 * it is never a second, independent filtering implementation.
 *
 * Design decision (documented per Part 10's instruction to document
 * cross-filter behavior): every chart on this dashboard always tallies
 * the FULL population handed to <PopulationSummary>, never the
 * currently-filtered subset. Selecting a segment highlights it and
 * drives the member table filter below, but the chart itself does not
 * shrink to "explain itself" -- if it did, clicking HIGH would turn the
 * risk donut into a trivial 100%-HIGH circle, which is confusing rather
 * than informative. This keeps every count on this dashboard always
 * accurate against the true population and never misleading.
 */

export interface CategoryDatum {
  key: string;
  label: string;
  count: number;
  color: string;
}

function useCounts<T extends string>(decisions: FinalUC07Decision[], getKey: (d: FinalUC07Decision) => T) {
  return useMemo(() => {
    const counts = new Map<string, number>();
    for (const d of decisions) {
      const key = getKey(d);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return counts;
  }, [decisions, getKey]);
}

function pct(count: number, total: number): number {
  return total > 0 ? Math.round((count / total) * 100) : 0;
}

/** Recharts' onClick payload shape differs slightly between Pie/Bar and
 * between versions -- the original datum is sometimes the entry itself,
 * sometimes nested under `.payload`. Defensively check both so a click
 * on the chart visual always resolves to the actual category key. */
function chartClickKey(entry: unknown): string | undefined {
  const e = entry as { key?: string; payload?: { key?: string } };
  return e?.payload?.key ?? e?.key;
}

/** Real, keyboard-accessible interaction + always-visible textual counts
 * (Part 23): each category is a real <button>, not just an SVG wedge.
 * The recharts visual next to it is decorative/supplementary (hover
 * tooltips only) -- color is never the only way to tell categories
 * apart, since the label + count + percentage are always plain text. */
function CategoryLegend({
  categories,
  total,
  activeKey,
  onSelect,
}: {
  categories: CategoryDatum[];
  total: number;
  activeKey: string | null;
  onSelect: (key: string) => void;
}) {
  return (
    <ul className="analytics-legend">
      {categories.map((c) => {
        const isActive = activeKey === c.key;
        const isDimmed = activeKey !== null && !isActive;
        return (
          <li key={c.key}>
            <button
              type="button"
              className={`analytics-legend__item${isActive ? " analytics-legend__item--active" : ""}${isDimmed ? " analytics-legend__item--dimmed" : ""}`}
              onClick={() => onSelect(c.key)}
              aria-pressed={isActive}
              aria-label={`${c.label}: ${c.count} members (${pct(c.count, total)}%)${isActive ? ", filter active" : ""}`}
            >
              <span className="analytics-legend__swatch" style={{ background: c.color }} aria-hidden="true" />
              <span className="analytics-legend__label" aria-hidden="true">
                {c.label}
              </span>
              <span className="analytics-legend__count tabular" aria-hidden="true">
                {c.count}
              </span>
              <span className="analytics-legend__percent tabular" aria-hidden="true">
                {pct(c.count, total)}%
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

type ChartAccent = "risk" | "safety" | "navigation" | "probability";

const CHART_ACCENT_ICON: Record<ChartAccent, ReactElement> = {
  risk: (
    <>
      <path d="M12 3 2 20h20L12 3Z" />
      <path d="M12 10v4" />
      <path d="M12 17h.01" />
    </>
  ),
  safety: <path d="M12 3 4.5 6v6.2c0 4.6 3.2 7.7 7.5 8.8 4.3-1.1 7.5-4.2 7.5-8.8V6L12 3Z" />,
  navigation: <circle cx="12" cy="12" r="9" />,
  probability: (
    <>
      <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />
    </>
  ),
};

function ChartTitle({ id, accent, children }: { id: string; accent: ChartAccent; children: string }) {
  return (
    <div className="analytics-card__title-row">
      <span className={`analytics-card__title-icon analytics-card__title-icon--${accent}`} aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          {CHART_ACCENT_ICON[accent]}
        </svg>
      </span>
      <h3 className="analytics-card__title" id={id}>
        {children}
      </h3>
    </div>
  );
}

/** Exported so a saved (server-paginated) population's summary view can
 * render the exact same chart shell fed with SQL-aggregated counts
 * (backend/db/repositories/populations.py's get_population_summary)
 * instead of a client-side tally over a full FinalUC07Decision[] --
 * see frontend/src/populations/SavedAnalyticsCharts.tsx. Same
 * component, same visuals; only where the counts come from differs. */
export function DonutChart({
  title,
  accent,
  categories,
  total,
  centerLabel,
  activeKey,
  onSelect,
  calloutText,
}: {
  title: string;
  accent: ChartAccent;
  categories: CategoryDatum[];
  total: number;
  centerLabel: string;
  activeKey: string | null;
  onSelect: (key: string) => void;
  calloutText?: string;
}) {
  const titleId = useId();
  const summary = categories.map((c) => `${c.label}: ${c.count} (${pct(c.count, total)}%)`).join(", ");

  return (
    <div className={`analytics-card analytics-card--${accent}`} aria-labelledby={titleId}>
      <ChartTitle id={titleId} accent={accent}>
        {title}
      </ChartTitle>
      {total === 0 ? (
        <p className="analytics-card__empty">No members in the current population.</p>
      ) : (
        <>
          <div className="analytics-card__body">
            <div
              className="analytics-donut"
              role="img"
              aria-label={`${title}: ${summary}. Total ${total} members.`}
            >
              <ResponsiveContainer width="100%" height={160}>
                <PieChart>
                  <Pie
                    data={categories}
                    dataKey="count"
                    nameKey="label"
                    innerRadius="62%"
                    outerRadius="100%"
                    paddingAngle={categories.filter((c) => c.count > 0).length > 1 ? 2 : 0}
                    stroke="var(--surface-card)"
                    strokeWidth={2}
                    isAnimationActive={false}
                    onClick={(entry) => {
                      const key = chartClickKey(entry);
                      if (key) onSelect(key);
                    }}
                  >
                    {categories.map((c) => (
                      <Cell
                        key={c.key}
                        fill={c.color}
                        opacity={activeKey === null || activeKey === c.key ? 1 : 0.35}
                        style={{ cursor: "pointer" }}
                      />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value, _name, item) => {
                      const n = Number(value) || 0;
                      return [`${n} (${pct(n, total)}%)`, item.payload.label];
                    }}
                    contentStyle={{
                      background: "var(--surface-elevated)",
                      border: "1px solid var(--border)",
                      borderRadius: "var(--radius-sm)",
                      fontSize: "0.78rem",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="analytics-donut__center" aria-hidden="true">
                <span className="analytics-donut__center-value tabular">{total}</span>
                <span className="analytics-donut__center-label">Members</span>
              </div>
            </div>
            <CategoryLegend categories={categories} total={total} activeKey={activeKey} onSelect={onSelect} />
          </div>
          <p className="analytics-card__hint">{centerLabel}</p>
          {calloutText && <p className="analytics-card__callout">{calloutText}</p>}
        </>
      )}
    </div>
  );
}

/** Exported for the same reason as DonutChart above. */
export function HorizontalBarChart({
  title,
  accent,
  categories,
  total,
  activeKey,
  onSelect,
}: {
  title: string;
  accent: ChartAccent;
  categories: CategoryDatum[];
  total: number;
  activeKey: string | null;
  onSelect: (key: string) => void;
}) {
  const titleId = useId();
  const summary = categories.map((c) => `${c.label}: ${c.count} (${pct(c.count, total)}%)`).join(", ");

  return (
    <div className={`analytics-card analytics-card--${accent}`} aria-labelledby={titleId}>
      <ChartTitle id={titleId} accent={accent}>
        {title}
      </ChartTitle>
      {total === 0 ? (
        <p className="analytics-card__empty">No members in the current population.</p>
      ) : (
        <>
          <div
            className="analytics-bar-chart"
            role="img"
            aria-label={`${title}: ${summary}. Total ${total} members.`}
          >
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={categories} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
                <XAxis type="number" hide domain={[0, "dataMax"]} />
                <YAxis
                  type="category"
                  dataKey="label"
                  width={130}
                  tick={{ fontSize: 11, fill: "var(--text-secondary)" }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  formatter={(value) => {
                    const n = Number(value) || 0;
                    return [`${n} (${pct(n, total)}%)`, "Members"];
                  }}
                  contentStyle={{
                    background: "var(--surface-elevated)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius-sm)",
                    fontSize: "0.78rem",
                  }}
                />
                <Bar
                  dataKey="count"
                  radius={[0, 4, 4, 0]}
                  isAnimationActive={false}
                  onClick={(entry) => {
                    const key = chartClickKey(entry);
                    if (key) onSelect(key);
                  }}
                  cursor="pointer"
                >
                  {categories.map((c) => (
                    <Cell key={c.key} fill={c.color} opacity={activeKey === null || activeKey === c.key ? 1 : 0.35} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <CategoryLegend categories={categories} total={total} activeKey={activeKey} onSelect={onSelect} />
        </>
      )}
    </div>
  );
}

export const RISK_META: { key: RiskTier; label: string }[] = [
  { key: "LOW", label: "Low" },
  { key: "MODERATE", label: "Moderate" },
  { key: "HIGH", label: "High" },
];

export function RiskDonut({
  decisions,
  activeTier,
  onSelectTier,
}: {
  decisions: FinalUC07Decision[];
  activeTier: RiskTier | null;
  onSelectTier: (tier: RiskTier | null) => void;
}) {
  const counts = useCounts(decisions, (d) => d.risk.tier);
  const categories: CategoryDatum[] = RISK_META.map((r) => ({
    key: r.key,
    label: r.label,
    count: counts.get(r.key) ?? 0,
    color: `var(--risk-${r.key.toLowerCase()})`,
  }));

  return (
    <DonutChart
      title="Risk Distribution"
      accent="risk"
      categories={categories}
      total={decisions.length}
      centerLabel="Click a segment to filter the member table by risk tier."
      activeKey={activeTier}
      onSelect={(key) => onSelectTier(activeTier === key ? null : (key as RiskTier))}
    />
  );
}

export const SAFETY_META: { key: SafetyState; label: string }[] = [
  { key: "CLEAR", label: "Clear" },
  { key: "CAUTION", label: "Caution" },
  { key: "OVERRIDE", label: "Override" },
];

export function SafetyDonut({
  decisions,
  activeSafety,
  onSelectSafety,
}: {
  decisions: FinalUC07Decision[];
  activeSafety: SafetyState | null;
  onSelectSafety: (state: SafetyState | null) => void;
}) {
  const counts = useCounts(decisions, (d) => d.safety.state);
  const overrideCount = counts.get("OVERRIDE") ?? 0;
  const categories: CategoryDatum[] = SAFETY_META.map((s) => ({
    key: s.key,
    label: s.label,
    count: counts.get(s.key) ?? 0,
    color: `var(--safety-${s.key.toLowerCase()})`,
  }));

  return (
    <DonutChart
      title="Safety Distribution"
      accent="safety"
      categories={categories}
      total={decisions.length}
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

export const NAV_META: { key: NavigationDestination; label: string }[] = NAVIGATION_DESTINATION_ORDER.map((key) => ({
  key,
  label: NAVIGATION_DESTINATION_LABEL[key],
}));

export function NavigationBar({
  decisions,
  activeDestination,
  onSelectDestination,
}: {
  decisions: FinalUC07Decision[];
  activeDestination: NavigationDestination | null;
  onSelectDestination: (destination: NavigationDestination | null) => void;
}) {
  const counts = useCounts(decisions, (d) => d.navigation.destination ?? "NO_PROACTIVE_NAVIGATION");
  const categories: CategoryDatum[] = NAV_META.map((n) => ({
    key: n.key,
    label: n.label,
    count: counts.get(n.key) ?? 0,
    color: NAVIGATION_DESTINATION_COLOR[n.key],
  }));

  return (
    <HorizontalBarChart
      title="Navigation Distribution"
      accent="navigation"
      categories={categories}
      total={decisions.length}
      activeKey={activeDestination}
      onSelect={(key) => onSelectDestination(activeDestination === key ? null : (key as NavigationDestination))}
    />
  );
}

interface ProbabilityBin {
  key: string;
  label: string;
  min: number;
  max: number | null;
  count: number;
}

export const BIN_EDGES = [0, 10, 20, 30, 40, 50];

function binForPercent(pctValue: number): number {
  for (let i = BIN_EDGES.length - 1; i >= 0; i--) {
    if (pctValue >= BIN_EDGES[i]) return i;
  }
  return 0;
}

export function ProbabilityHistogram({
  decisions,
  activeBin,
  onSelectBin,
}: {
  decisions: FinalUC07Decision[];
  activeBin: string | null;
  onSelectBin: (bin: { min: number; max: number | null } | null) => void;
}) {
  const titleId = useId();

  const bins: ProbabilityBin[] = useMemo(() => {
    const counts = new Array(BIN_EDGES.length).fill(0);
    for (const d of decisions) {
      counts[binForPercent(d.risk.probability * 100)] += 1;
    }
    return BIN_EDGES.map((edge, i) => ({
      key: String(edge),
      label: i === BIN_EDGES.length - 1 ? `${edge}%+` : `${edge}–${BIN_EDGES[i + 1] - 1}%`,
      min: edge,
      max: i === BIN_EDGES.length - 1 ? null : BIN_EDGES[i + 1],
      count: counts[i],
    }));
  }, [decisions]);

  // Part 9: threshold markers are shown ONLY because the backend already
  // returns moderate_threshold/high_threshold on every decision -- never
  // recreated or hard-coded. Every decision in one batch shares the same
  // trained model, so the first decision's thresholds represent the
  // whole population.
  const moderateThresholdPct = decisions[0] ? decisions[0].risk.moderate_threshold * 100 : null;
  const highThresholdPct = decisions[0] ? decisions[0].risk.high_threshold * 100 : null;
  const moderateBinKey = moderateThresholdPct !== null ? bins[binForPercent(moderateThresholdPct)]?.key : null;
  const highBinKey = highThresholdPct !== null ? bins[binForPercent(highThresholdPct)]?.key : null;

  const total = decisions.length;
  const summary = bins.map((b) => `${b.label}: ${b.count} members`).join(", ");

  return (
    <div className="analytics-card analytics-card--probability" aria-labelledby={titleId}>
      <ChartTitle id={titleId} accent="probability">
        Probability Distribution
      </ChartTitle>
      {total === 0 ? (
        <p className="analytics-card__empty">No members in the current population.</p>
      ) : (
        <>
          <div className="analytics-bar-chart" role="img" aria-label={`Probability distribution: ${summary}.`}>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={bins} margin={{ top: 4, right: 8, left: -20, bottom: 4 }}>
                <XAxis dataKey="label" tick={{ fontSize: 10, fill: "var(--text-secondary)" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "var(--text-secondary)" }} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip
                  formatter={(value) => {
                    const n = Number(value) || 0;
                    return [`${n} member${n === 1 ? "" : "s"}`, "Count"];
                  }}
                  contentStyle={{
                    background: "var(--surface-elevated)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius-sm)",
                    fontSize: "0.78rem",
                  }}
                />
                <Bar
                  dataKey="count"
                  radius={[4, 4, 0, 0]}
                  isAnimationActive={false}
                  cursor="pointer"
                  onClick={(entry) => {
                    const key = chartClickKey(entry);
                    const bin = bins.find((b) => b.key === key);
                    if (!bin) return;
                    onSelectBin(activeBin === bin.key ? null : { min: bin.min, max: bin.max });
                  }}
                >
                  {bins.map((b) => (
                    <Cell
                      key={b.key}
                      fill="var(--accent)"
                      opacity={activeBin === null || activeBin === b.key ? 1 : 0.35}
                      stroke={b.key === moderateBinKey || b.key === highBinKey ? "var(--risk-high)" : undefined}
                      strokeWidth={b.key === moderateBinKey || b.key === highBinKey ? 2 : 0}
                      strokeDasharray={b.key === moderateBinKey || b.key === highBinKey ? "3 2" : undefined}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <ul className="analytics-legend analytics-legend--histogram">
            {bins.map((b) => {
              const isActive = activeBin === b.key;
              const isDimmed = activeBin !== null && !isActive;
              return (
                <li key={b.key}>
                  <button
                    type="button"
                    className={`analytics-legend__item${isActive ? " analytics-legend__item--active" : ""}${isDimmed ? " analytics-legend__item--dimmed" : ""}`}
                    onClick={() => onSelectBin(isActive ? null : { min: b.min, max: b.max })}
                    aria-pressed={isActive}
                    aria-label={`Probability ${b.label}: ${b.count} members${isActive ? ", filter active" : ""}`}
                  >
                    <span className="analytics-legend__label" aria-hidden="true">
                      {b.label}
                    </span>
                    <span className="analytics-legend__count tabular" aria-hidden="true">
                      {b.count}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
          {(moderateThresholdPct !== null || highThresholdPct !== null) && (
            <p className="analytics-card__hint">
              {moderateThresholdPct !== null && `Moderate threshold ≈ ${moderateThresholdPct.toFixed(1)}%`}
              {moderateThresholdPct !== null && highThresholdPct !== null && " · "}
              {highThresholdPct !== null && `High threshold ≈ ${highThresholdPct.toFixed(1)}%`}
              {" (from the model's own reported thresholds; the marked bin contains it.)"}
            </p>
          )}
        </>
      )}
    </div>
  );
}
