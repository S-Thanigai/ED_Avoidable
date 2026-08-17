import type { ReactElement } from "react";
import type { SafetyDecision } from "../types";
import { ContextStatus } from "./ContextStatus";
import "./SafetyCard.css";

const STATE_LABEL: Record<SafetyDecision["state"], string> = {
  CLEAR: "Clear",
  CAUTION: "Caution",
  OVERRIDE: "Emergency safety override",
};

const STATE_ICON_PATH: Record<SafetyDecision["state"], ReactElement> = {
  CLEAR: <path d="M4 12.5 9 17.5 20 6.5" />,
  CAUTION: (
    <>
      <path d="M12 3 2 20h20L12 3Z" />
      <path d="M12 10v4M12 17h.01" />
    </>
  ),
  OVERRIDE: (
    <>
      <path d="M12 3 4.5 6v6.2c0 4.6 3.2 7.7 7.5 8.8 4.3-1.1 7.5-4.2 7.5-8.8V6L12 3Z" />
      <path d="m9.5 9.5 5 5M14.5 9.5l-5 5" />
    </>
  ),
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
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              {STATE_ICON_PATH[safety.state]}
            </svg>
          </span>
          {STATE_LABEL[safety.state]}
        </span>
      </div>

      <p className="safety-card__message">{safety.message}</p>

      <ContextStatus completeness={safety.context_completeness} source={safety.context_source} />
    </section>
  );
}
