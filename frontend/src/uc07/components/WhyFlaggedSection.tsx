import type { ExplanationFactor, RiskAssessment } from "../types";
import "./WhyFlaggedSection.css";

const METHOD_LABEL: Record<RiskAssessment["explanation_method"], string> = {
  SHAP_LINEAR: "SHAP (logistic regression model explanation)",
  LINEAR_CONTRIBUTION: "Logistic regression model explanation (linear contribution)",
};

function FactorRow({ factor }: { factor: ExplanationFactor }) {
  const increasing = factor.direction === "INCREASES_RISK";
  return (
    <li className={`why-flagged__factor why-flagged__factor--${increasing ? "up" : "down"}`}>
      <span className="why-flagged__arrow" aria-hidden="true">
        {increasing ? "↑" : "↓"}
      </span>
      <span className="why-flagged__label">{factor.display_name}</span>
      <span className="why-flagged__direction-text">{increasing ? "increased" : "decreased"} the model's estimate</span>
    </li>
  );
}

/**
 * "WHY THIS MEMBER WAS FLAGGED" (Phase 8C Part 13). Renders the Risk
 * Detection Agent's authoritative, structured `explanation_factors`
 * exactly as returned -- never re-derives, re-orders, or reinterprets
 * them, and never claims a factor CAUSES anything (only that it
 * "increased"/"decreased the model's estimate", matching the backend's
 * own non-causal framing). This is a purely additive section alongside
 * RiskCard's existing plain-sentence `contributing_factors`, not a
 * replacement for it.
 */
export function WhyFlaggedSection({ risk }: { risk: RiskAssessment }) {
  if (risk.explanation_factors.length === 0) return null;

  const increasing = risk.explanation_factors.filter((f) => f.direction === "INCREASES_RISK");
  const decreasing = risk.explanation_factors.filter((f) => f.direction === "DECREASES_RISK");

  return (
    <section className="why-flagged" aria-label="Why this member was flagged">
      <h3 className="why-flagged__heading">Why This Member Was Flagged</h3>
      <p className="why-flagged__note">
        These factors contributed to the model's own risk estimate. They describe patterns in the data, not a
        certainty about what will happen.
      </p>

      {increasing.length > 0 && (
        <ul className="why-flagged__list">
          {increasing.map((f) => (
            <FactorRow key={f.feature} factor={f} />
          ))}
        </ul>
      )}
      {decreasing.length > 0 && (
        <ul className="why-flagged__list">
          {decreasing.map((f) => (
            <FactorRow key={f.feature} factor={f} />
          ))}
        </ul>
      )}

      <p className="why-flagged__method">Explanation method: {METHOD_LABEL[risk.explanation_method]}</p>
      {/* Phase 8D Part 10 -- SHAP math is unchanged; this only adds a UI
          caveat about how to READ correlation-aware attribution values
          (see docs/DECISION_LOG.md #117: SHAP's correlation-aware masker
          can attribute a factor's contribution differently than a naive
          per-feature reading when features are correlated). */}
      <p className="why-flagged__caveat">
        Model contribution values are attribution signals and may reflect correlated features; they should not be
        interpreted as causal effects.
      </p>
    </section>
  );
}
