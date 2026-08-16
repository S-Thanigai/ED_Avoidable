import type { FinalUC07Decision, NavigationDestination, RiskTier, SafetyState } from "../types";
import type { MemberFiltersState } from "../tableState";
import { NavigationBar, ProbabilityHistogram, RiskDonut, SafetyDonut } from "./AnalyticsCharts";
import "./PopulationSummary.css";

function KpiCard({ label, value, total, tone }: { label: string; value: number; total: number; tone?: "critical" }) {
  const percent = total > 0 ? Math.round((value / total) * 100) : 0;
  return (
    <div className={`population-summary__kpi${tone ? ` population-summary__kpi--${tone}` : ""}`}>
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
  const overrideCount = decisions.filter((d) => d.safety.state === "OVERRIDE").length;

  const activeTier: RiskTier | null = filters.tier === "ALL" ? null : filters.tier;
  const activeNavigation: NavigationDestination | null = filters.navigation === "ALL" ? null : filters.navigation;
  const activeSafety: SafetyState | null = filters.safety === "ALL" ? null : filters.safety;
  const activeBinKey = filters.probMin !== "" ? filters.probMin : null;

  return (
    <section className="population-summary" aria-label="Population overview and analytics">
      <div className="population-summary__header">
        <h2 className="population-summary__heading">Population Overview</h2>
        <p className="population-summary__caveat">
          Describes UC07 model/agent output for this population, not a clinical outcome or diagnosis.
        </p>
      </div>

      <div className="population-summary__kpis">
        <KpiCard label="Total Members" value={total} total={total} />
        <KpiCard label="High Risk" value={highRiskCount} total={total} />
        <KpiCard label="Override" value={overrideCount} total={total} tone={overrideCount > 0 ? "critical" : undefined} />
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
