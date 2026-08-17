import { useMemo } from "react";
import type { PatientRow } from "../types";
import "./StatCards.css";

export function StatCards({ rows }: { rows: PatientRow[] }) {
  const counts = useMemo(() => {
    const total = rows.length;
    const high = rows.filter((r) => r.risk_category === "High").length;
    const medium = rows.filter((r) => r.risk_category === "Medium").length;
    const low = rows.filter((r) => r.risk_category === "Low").length;
    return { total, high, medium, low };
  }, [rows]);

  const pct = (n: number) => (counts.total ? Math.round((n / counts.total) * 100) : 0);

  return (
    <div className="stat-cards">
      <div className="stat-card">
        <span className="stat-card__label">Total patients scored</span>
        <span className="stat-card__value tabular">{counts.total.toLocaleString()}</span>
      </div>
      <div className="stat-card stat-card--critical">
        <span className="stat-card__label">High risk</span>
        <span className="stat-card__value tabular">{counts.high.toLocaleString()}</span>
        <span className="stat-card__sub">{pct(counts.high)}% of scored patients</span>
      </div>
      <div className="stat-card stat-card--warning">
        <span className="stat-card__label">Medium risk</span>
        <span className="stat-card__value tabular">{counts.medium.toLocaleString()}</span>
        <span className="stat-card__sub">{pct(counts.medium)}% of scored patients</span>
      </div>
      <div className="stat-card stat-card--good">
        <span className="stat-card__label">Low risk</span>
        <span className="stat-card__value tabular">{counts.low.toLocaleString()}</span>
        <span className="stat-card__sub">{pct(counts.low)}% of scored patients</span>
      </div>
    </div>
  );
}
