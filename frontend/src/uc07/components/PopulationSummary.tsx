import type { ReactElement } from "react";
import type { FinalUC07Decision, NavigationDestination, RiskTier, SafetyState } from "../types";
import type { MemberFiltersState } from "../tableState";
import { NavigationBar, ProbabilityHistogram, RiskDonut, SafetyDonut } from "./AnalyticsCharts";
import "./PopulationSummary.css";

type KpiTone = "neutral" | "warning" | "critical" | "teal";

const ICON_PATHS: Record<string, ReactElement> = {
  members: (
    <>
      <path d="M9 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" />
      <path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6" />
      <path d="M16 8.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z" />
      <path d="M15 14.2c2.9.4 5 2.7 5 5.8" />
    </>
  ),
  alert: (
    <>
      <path d="M12 3 2 20h20L12 3Z" />
      <path d="M12 10v4" />
      <path d="M12 17h.01" />
    </>
  ),
  pulse: (
    <>
      <path d="M3 12h4l2-7 4 14 2-7h6" />
    </>
  ),
  compass: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="m14.5 9.5-2 5-5 2 2-5 5-2Z" />
    </>
  ),
  shieldWarn: (
    <>
      <path d="M12 3 4.5 6v6.2c0 4.6 3.2 7.7 7.5 8.8 4.3-1.1 7.5-4.2 7.5-8.8V6L12 3Z" />
      <path d="M12 8.5v4" />
      <path d="M12 15.2h.01" />
    </>
  ),
  shieldOff: (
    <>
      <path d="M12 3 4.5 6v6.2c0 4.6 3.2 7.7 7.5 8.8 4.3-1.1 7.5-4.2 7.5-8.8V6L12 3Z" />
      <path d="m9.5 9.5 5 5M14.5 9.5l-5 5" />
    </>
  ),
};

function KpiIcon({ name }: { name: keyof typeof ICON_PATHS }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      {ICON_PATHS[name]}
    </svg>
  );
}

function KpiCard({
  label,
  value,
  total,
  icon,
  tone = "neutral",
}: {
  label: string;
  value: number;
  total: number;
  icon: keyof typeof ICON_PATHS;
  tone?: KpiTone;
}) {
  const percent = total > 0 ? Math.round((value / total) * 100) : 0;
  return (
    <div className={`population-summary__kpi population-summary__kpi--${tone}`}>
      <span className="population-summary__kpi-icon" aria-hidden="true">
        <KpiIcon name={icon} />
      </span>
      <span className="population-summary__kpi-value tabular">{value.toLocaleString()}</span>
      <span className="population-summary__kpi-label">{label}</span>
      {total > 0 && <span className="population-summary__kpi-percent tabular">{percent}% of population</span>}
    </div>
  );
}

/**
 * Population Overview + interactive analytics (Part 4-9). Every number
 * on this dashboard is a tally over the already-decided
 * FinalUC07Decision[] population the backend returned -- nothing here
 * computes a risk tier, navigation destination, safety state, or
 * probability. `decisions` is deliberately the FULL (unfiltered, but
 * safety-override-merged) population, not the currently filtered
 * subset -- see AnalyticsCharts.tsx for why. Clicking a chart segment
 * calls `onFilterChange` with the SAME MemberFiltersState shape
 * MemberFilters' dropdowns already write to, so chart selections and
 * dropdown filters (and their chips) are automatically kept in sync --
 * there is only one filter state in this app, not two.
 */
export function PopulationSummary({
  decisions,
  filters,
  onFilterChange,
}: {
  decisions: FinalUC07Decision[];
  filters: MemberFiltersState;
  onFilterChange: (next: MemberFiltersState) => void;
}) {
  const total = decisions.length;
  const highRiskCount = decisions.filter((d) => d.risk.tier === "HIGH").length;
  const moderateRiskCount = decisions.filter((d) => d.risk.tier === "MODERATE").length;
  const navigationOpportunityCount = decisions.filter(
    (d) => d.navigation.destination !== null && d.navigation.destination !== "NO_PROACTIVE_NAVIGATION",
  ).length;
  const overrideCount = decisions.filter((d) => d.safety.state === "OVERRIDE").length;
  const cautionCount = decisions.filter((d) => d.safety.state === "CAUTION").length;

  const activeTier: RiskTier | null = filters.tier === "ALL" ? null : filters.tier;
  const activeNavigation: NavigationDestination | null = filters.navigation === "ALL" ? null : filters.navigation;
  const activeSafety: SafetyState | null = filters.safety === "ALL" ? null : filters.safety;
  const activeBinKey = filters.probMin !== "" ? filters.probMin : null;

  return (
    <section className="population-summary" aria-label="Population overview and analytics">
      <div className="population-summary__header">
        <h2 className="population-summary__heading">Population Overview</h2>
        <p className="population-summary__caveat">
          Risk stratification, navigation opportunities and safety status across the analyzed
          population — not a clinical outcome or diagnosis.
        </p>
      </div>

      <div className="population-summary__kpis">
        <KpiCard label="Total Members" value={total} total={total} icon="members" tone="neutral" />
        <KpiCard label="High Risk" value={highRiskCount} total={total} icon="alert" tone="critical" />
        <KpiCard label="Moderate Risk" value={moderateRiskCount} total={total} icon="pulse" tone="warning" />
        <KpiCard label="Navigation Opportunities" value={navigationOpportunityCount} total={total} icon="compass" tone="teal" />
        <KpiCard label="Safety Caution" value={cautionCount} total={total} icon="shieldWarn" tone="warning" />
        <KpiCard label="Safety Override" value={overrideCount} total={total} icon="shieldOff" tone="critical" />
      </div>

      <div className="population-summary__charts">
        <RiskDonut decisions={decisions} activeTier={activeTier} onSelectTier={(tier) => onFilterChange({ ...filters, tier: tier ?? "ALL" })} />
        <NavigationBar
          decisions={decisions}
          activeDestination={activeNavigation}
          onSelectDestination={(destination) => onFilterChange({ ...filters, navigation: destination ?? "ALL" })}
        />
        <SafetyDonut
          decisions={decisions}
          activeSafety={activeSafety}
          onSelectSafety={(state) => onFilterChange({ ...filters, safety: state ?? "ALL" })}
        />
        <ProbabilityHistogram
          decisions={decisions}
          activeBin={activeBinKey}
          onSelectBin={(bin) =>
            onFilterChange({
              ...filters,
              probMin: bin ? String(bin.min) : "",
              probMax: bin && bin.max !== null ? String(bin.max) : "",
            })
          }
        />
      </div>
    </section>
  );
}
