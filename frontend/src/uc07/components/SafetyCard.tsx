import type { ReactElement } from "react";
import type { SafetyDecision } from "../types";
import { ContextStatus } from "./ContextStatus";
import "./SafetyCard.css";

const STATE_LABEL: Record<SafetyDecision["state"], string> = {
  CLEAR: "Clear",
  CAUTION: "Caution",
  OVERRIDE: "Emergency safety override",
};

// CAUTION-specific display text (UI-only simplification): the backend's
// safety.message for CAUTION includes emergency/911 instructions that
// belong in the actual emergency-detection OVERRIDE state, not in a
// routine "we don't have current information yet" state -- showing them
// here reads as an emergency alert when none has been detected. This
// does not change what the Safety Agent decides (CLEAR/CAUTION/OVERRIDE,
// override triggers, blocked phrases are all unchanged) -- only what
// text this card displays for the CAUTION case specifically. CLEAR and
// OVERRIDE continue to render the backend's own safety.message verbatim.
const CAUTION_DISPLAY_MESSAGE =
  "Current safety information is unavailable for this encounter. Review the member's current " +
  "clinical context before making a lower-acuity care navigation recommendation.";
const CAUTION_SUPPORTING_LINE =
  "Current information was not supplied, so the system cannot confirm an appropriate lower-acuity pathway.";

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

      <p className="safety-card__message">
        {safety.state === "CAUTION" ? CAUTION_DISPLAY_MESSAGE : safety.message}
      </p>
      {safety.state === "CAUTION" && (
        <p className="safety-card__supporting">{CAUTION_SUPPORTING_LINE}</p>
      )}

      <ContextStatus completeness={safety.context_completeness} source={safety.context_source} />
    </section>
  );
}
