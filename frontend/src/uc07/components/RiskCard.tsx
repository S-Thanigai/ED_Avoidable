import type { RiskAssessment } from "../types";
import "./RiskCard.css";

const TIER_LABEL: Record<RiskAssessment["tier"], string> = {
  LOW: "Low",
  MODERATE: "Moderate",
  HIGH: "High",
};

/** Displays the Risk Detection Agent's output exactly as returned --
 * never computes or re-derives a tier from the probability itself. */
export function RiskCard({ risk }: { risk: RiskAssessment }) {
  const pct = (risk.probability * 100).toFixed(1);

  return (
    <section className="risk-card" aria-label="Risk assessment">
      <div className="risk-card__header">
        <span className="risk-card__eyebrow">Estimated risk of future potentially avoidable ED utilization</span>
        <span className={`risk-card__tier risk-card__tier--${risk.tier.toLowerCase()}`}>
          <span className="risk-card__tier-dot" aria-hidden="true" />
          {TIER_LABEL[risk.tier]}
        </span>
      </div>

      <p className="risk-card__probability">
        <span className="risk-card__probability-value tabular">{pct}%</span>
        <span className="risk-card__probability-label">predicted 90-day navigation risk</span>
      </p>

      {risk.contributing_factors.length > 0 && (
        <div className="risk-card__factors">
          <span className="risk-card__factors-title">Top contributing factors</span>
          <ul className="risk-card__factors-list">
            {risk.contributing_factors.map((factor) => (
              <li key={factor}>{factor}</li>
            ))}
          </ul>
        </div>
      )}

      <p className="risk-card__footnote">
        Model {risk.model_version} &middot; as of {risk.index_date}
      </p>
    </section>
  );
}
