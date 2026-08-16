import type { SafetyDecision } from "../types";
import { ContextStatus } from "./ContextStatus";
import "./SafetyCard.css";

const STATE_LABEL: Record<SafetyDecision["state"], string> = {
  CLEAR: "Clear",
  CAUTION: "Caution",
  OVERRIDE: "Emergency safety override",
};

/** Displays the Safety & Policy Agent's output -- the final,
 * non-bypassable authority. This component never computes a safety
 * state itself; it only renders what the backend returned, with
 * OVERRIDE always given the strongest visual priority on the page. */
export function SafetyCard({ safety }: { safety: SafetyDecision }) {
  return (
    <section
      className={`safety-card safety-card--${safety.state.toLowerCase()}`}
      aria-label="Safety status"
      role={safety.state === "OVERRIDE" ? "alert" : undefined}
    >
      <div className="safety-card__header">
        <span className="safety-card__eyebrow">Safety status</span>
        <span className="safety-card__state">
          <span className="safety-card__state-icon" aria-hidden="true">
            {safety.state === "OVERRIDE" ? "⛔" : safety.state === "CAUTION" ? "⚠" : "✓"}
          </span>
          {STATE_LABEL[safety.state]}
        </span>
      </div>

      <p className="safety-card__message">{safety.message}</p>

      <ContextStatus completeness={safety.context_completeness} source={safety.context_source} />
    </section>
  );
}
