import type { FinalUC07Decision } from "../types";
import "./PopulationSummary.css";

const TIER_ROWS: { key: string; label: string; tone: "good" | "warning" | "critical" }[] = [
  { key: "LOW", label: "Low", tone: "good" },
  { key: "MODERATE", label: "Moderate", tone: "warning" },
  { key: "HIGH", label: "High", tone: "critical" },
];

const NAV_ROWS: { key: string; label: string }[] = [
  { key: "NO_PROACTIVE_NAVIGATION", label: "No proactive navigation" },
  { key: "PRIMARY_CARE", label: "Primary Care" },
  { key: "URGENT_CARE", label: "Urgent Care" },
  { key: "TELEHEALTH", label: "Telehealth" },
  { key: "CARE_MANAGEMENT", label: "Care Management" },
];

const SAFETY_ROWS: { key: string; label: string; tone: "good" | "warning" | "critical" }[] = [
  { key: "CLEAR", label: "Clear", tone: "good" },
  { key: "CAUTION", label: "Caution", tone: "warning" },
  { key: "OVERRIDE", label: "Override", tone: "critical" },
];

function count(values: string[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const v of values) m.set(v, (m.get(v) ?? 0) + 1);
  return m;
}

function SummaryCard({
  title,
  rows,
  counts,
}: {
  title: string;
  rows: { key: string; label: string; tone?: "good" | "warning" | "critical" }[];
  counts: Map<string, number>;
}) {
  return (
    <div className="population-summary__card">
      <span className="population-summary__card-title">{title}</span>
      <dl className="population-summary__card-rows">
        {rows.map((row) => (
          <div className="population-summary__card-row" key={row.key}>
            <dt className={row.tone ? `population-summary__dot population-summary__dot--${row.tone}` : undefined}>
              {row.tone && <span className="population-summary__dot-mark" aria-hidden="true" />}
              {row.label}
            </dt>
            <dd className="tabular">{counts.get(row.key) ?? 0}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** Population-level summary of already-computed FinalUC07Decision
 * results -- purely a tally of backend output, never a clinical
 * conclusion about the population. Shows the FILTERED set's breakdown
 * (the currently-relevant view) and, when filters are active, the total
 * analyzed count alongside it. */
export function PopulationSummary({
  decisions,
  totalCount,
  filtersActive,
}: {
  decisions: FinalUC07Decision[];
  totalCount: number;
  filtersActive: boolean;
}) {
  const tierCounts = count(decisions.map((d) => d.risk.tier));
  const destCounts = count(decisions.map((d) => d.navigation.destination ?? "NONE"));
  const safetyCounts = count(decisions.map((d) => d.safety.state));

  return (
    <section className="population-summary" aria-label="Cohort summary">
      <div className="population-summary__caveat">
        {filtersActive ? (
          <span>
            Showing summary for <strong>{decisions.length}</strong> of <strong>{totalCount}</strong> total
            analyzed members (filters active).
          </span>
        ) : (
          <span>
            <strong>{totalCount}</strong> members analyzed.
          </span>
        )}{" "}
        Counts describe UC07 model/agent output for this population, not a clinical outcome or
        diagnosis.
      </div>

      <div className="population-summary__grid">
        <SummaryCard title="Risk distribution" rows={TIER_ROWS} counts={tierCounts} />
        <SummaryCard title="Navigation" rows={NAV_ROWS} counts={destCounts} />
        <SummaryCard title="Safety" rows={SAFETY_ROWS} counts={safetyCounts} />
      </div>
    </section>
  );
}
