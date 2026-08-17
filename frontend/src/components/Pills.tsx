import type { RiskCategory } from "../types";
import "./Pills.css";

const RISK_LABEL: Record<RiskCategory, string> = {
  Low: "Low risk",
  Medium: "Medium risk",
  High: "High risk",
};

export function RiskPill({ category }: { category: RiskCategory }) {
  return (
    <span className={`risk-pill risk-pill--${category.toLowerCase()}`}>
      <span className="risk-pill__dot" aria-hidden="true" />
      {RISK_LABEL[category] ?? category}
    </span>
  );
}

const CARE_CLASS: Record<string, string> = {
  PCP: "care-pill--pcp",
  "Urgent Care": "care-pill--urgent",
  Telehealth: "care-pill--telehealth",
};

export function CarePill({ care }: { care: string }) {
  const cls = CARE_CLASS[care] ?? "care-pill--pcp";
  return <span className={`care-pill ${cls}`}>{care}</span>;
}
