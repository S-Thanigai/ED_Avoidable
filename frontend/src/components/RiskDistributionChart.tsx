import { useMemo } from "react";
import type { PatientRow, RiskCategory } from "../types";
import "./RiskDistributionChart.css";

const ORDER: { key: RiskCategory; label: string; varName: string }[] = [
  { key: "Low", label: "Low", varName: "--status-good" },
  { key: "Medium", label: "Medium", varName: "--status-warning" },
  { key: "High", label: "High", varName: "--status-critical" },
];

export function RiskDistributionChart({ rows }: { rows: PatientRow[] }) {
  const data = useMemo(() => {
    const total = rows.length || 1;
    return ORDER.map((entry) => {
      const count = rows.filter((r) => r.risk_category === entry.key).length;
      return { ...entry, count, share: count / total };
    });
  }, [rows]);

  const max = Math.max(...data.map((d) => d.share), 0.0001);

  return (
    <div className="risk-chart" role="img" aria-label="Patient count by risk category">
      <h2 className="risk-chart__title">Risk distribution</h2>
      <div className="risk-chart__rows">
        {data.map((d) => (
          <div className="risk-chart__row" key={d.key}>
            <span className="risk-chart__label">{d.label}</span>
            <div className="risk-chart__track">
              <div
                className="risk-chart__bar"
                style={{
                  width: `${(d.share / max) * 100}%`,
                  background: `var(${d.varName})`,
                }}
              />
            </div>
            <span className="risk-chart__value tabular">{d.count.toLocaleString()}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
