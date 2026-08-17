import type { ExplanationFactor, RiskAssessment } from "../types";
import "./WhyFlaggedSection.css";

const METHOD_LABEL: Record<RiskAssessment["explanation_method"], string> = {
  SHAP_LINEAR: "SHAP",
  LINEAR_CONTRIBUTION: "Logistic regression (linear contribution)",
};

/** Horizontal diverging bar centered on zero -- increasing contributions
 * extend right, decreasing extend left. Bar length is scaled relative to
 * the largest |contribution| among ALL factors shown, so bars stay
 * comparable to each other within this one member's list (contribution
 * scale is not comparable ACROSS members or across explanation_method
 * values -- see contracts.py's ExplanationFactor docstring). */
function FactorBar({ factor, maxAbs }: { factor: ExplanationFactor; maxAbs: number }) {
  const increasing = factor.direction === "INCREASES_RISK";
  const magnitude = Math.abs(factor.contribution);
  const widthPct = maxAbs > 0 ? (magnitude / maxAbs) * 100 : 0;

  return (
    <li className="why-flagged__row">
      <span className="why-flagged__row-label">{factor.display_name}</span>
      <span className="why-flagged__row-track" aria-hidden="true">
        <span className="why-flagged__row-zero" />
        {increasing ? (
          <span
            className="why-flagged__row-bar why-flagged__row-bar--up"
            style={{ width: `${widthPct / 2}%` }}
          />
        ) : (
          <span
            className="why-flagged__row-bar why-flagged__row-bar--down"
            style={{ width: `${widthPct / 2}%` }}
          />
        )}
      </span>
      <span className={`why-flagged__row-value why-flagged__row-value--${increasing ? "up" : "down"}`}>
        {increasing ? "+" : "−"}
        {magnitude.toFixed(3)}
      </span>
    </li>
  );
}

/**
 * "WHY THIS MEMBER WAS FLAGGED" (Phase 8C Part 13, redesigned for the
 * Phase 9 UI refinement). Renders the Risk Detection Agent's
 * authoritative, structured `explanation_factors` exactly as returned --
 * never re-derives, re-orders, or reinterprets them, and never claims a
 * factor CAUSES anything. This is purely a presentation change: the
 * underlying SHAP/linear-contribution math is untouched, only how the
 * signed contribution values are drawn (a centered diverging bar instead
 * of a plain arrow) changed.
 */
export function WhyFlaggedSection({ risk }: { risk: RiskAssessment }) {
  if (risk.explanation_factors.length === 0) return null;

  const increasing = risk.explanation_factors.filter((f) => f.direction === "INCREASES_RISK");
  const decreasing = risk.explanation_factors.filter((f) => f.direction === "DECREASES_RISK");
  const maxAbs = Math.max(...risk.explanation_factors.map((f) => Math.abs(f.contribution)), 0);

  return (
    <section className="why-flagged" aria-label="Why this member was flagged">
      <div className="why-flagged__heading-row">
        <span className="why-flagged__icon" aria-hidden="true">
          ◆
        </span>
        <div>
          <h3 className="why-flagged__heading">Why This Member Was Flagged</h3>
          <p className="why-flagged__subtitle">Model feature contributions — not causal explanations.</p>
        </div>
        <span
          className="why-flagged__info"
          tabIndex={0}
          role="note"
          aria-label="These values describe how model features contributed to this prediction. They do not establish causation."
          title="These values describe how model features contributed to this prediction. They do not establish causation."
        >
          ⓘ
        </span>
      </div>

      {increasing.length > 0 && (
        <div className="why-flagged__group">
          <span className="why-flagged__group-label why-flagged__group-label--up">Increased estimate</span>
          <ul className="why-flagged__list">
            {increasing.map((f) => (
              <FactorBar key={f.feature} factor={f} maxAbs={maxAbs} />
            ))}
          </ul>
        </div>
      )}
      {decreasing.length > 0 && (
        <div className="why-flagged__group">
          <span className="why-flagged__group-label why-flagged__group-label--down">Decreased estimate</span>
          <ul className="why-flagged__list">
            {decreasing.map((f) => (
              <FactorBar key={f.feature} factor={f} maxAbs={maxAbs} />
            ))}
          </ul>
        </div>
      )}

      <p className="why-flagged__method">
        Explanation method: <span className="why-flagged__method-value">{METHOD_LABEL[risk.explanation_method]}</span>
      </p>
      <p className="why-flagged__caveat">
        Model contribution values are attribution signals and may reflect correlated features; they should not be
        interpreted as causal effects.
      </p>
    </section>
  );
}
