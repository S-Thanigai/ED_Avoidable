import { useState } from "react";
import type { FinalNavigationView, SafetyState } from "../types";
import { NAVIGATION_DESTINATION_COLOR, NAVIGATION_DESTINATION_LABEL } from "../navigationDisplay";
import "./NavigationCard.css";

const REASON_LABEL: Record<string, string> = {
  ELEVATED_FUTURE_RISK: "Elevated predicted future risk",
  REPEATED_LOWER_ACUITY_HISTORY: "Repeated lower-acuity utilization history",
  TRANSPORTATION_BARRIER: "Reported transportation barrier",
  LIMITED_PCP_ACCESS: "Limited primary care access",
  TELEHEALTH_AVAILABLE: "Telehealth available",
  CHRONIC_COMPLEXITY: "Chronic condition complexity",
  PRIOR_CM_ENGAGEMENT: "Prior Care Management engagement",
  OUTPATIENT_CONTINUITY_OPPORTUNITY: "Outpatient continuity opportunity",
  URGENT_CARE_ACCESS_ADVANTAGE: "Better access to urgent care than primary care",
  NO_OPPORTUNITY_IDENTIFIED: "No meaningful navigation opportunity identified",
};

/** Displays the Care Navigation Agent's output exactly as returned by
 * the Safety-reviewed FinalNavigationView -- never recomputes a
 * destination or generates new rationale text client-side. */
export function NavigationCard({
  navigation,
  safetyState,
}: {
  navigation: FinalNavigationView;
  safetyState: SafetyState;
}) {
  const [expanded, setExpanded] = useState(false);
  const suppressed = safetyState === "OVERRIDE";

  return (
    <section className="navigation-card" aria-label="Navigation recommendation">
      <span className="navigation-card__eyebrow">Navigation recommendation</span>

      {suppressed ? (
        <div className="navigation-card__suppressed">
          <span className="navigation-card__suppressed-badge">Navigation logic suppressed by safety override</span>
          <p className="navigation-card__suppressed-text">
            No non-emergency navigation option is offered for this encounter. See the safety
            status above.
          </p>
        </div>
      ) : (
        <>
          <span className="navigation-card__destination">
            {navigation.destination && (
              <span
                className="navigation-card__destination-dot"
                aria-hidden="true"
                style={{ background: NAVIGATION_DESTINATION_COLOR[navigation.destination] }}
              />
            )}
            {navigation.destination ? NAVIGATION_DESTINATION_LABEL[navigation.destination] ?? navigation.destination : "—"}
          </span>
          <p className="navigation-card__explanation">{navigation.explanation}</p>

          {navigation.reason_codes.length > 0 && (
            <div className="navigation-card__why">
              <button
                type="button"
                className="navigation-card__why-toggle"
                onClick={() => setExpanded((v) => !v)}
                aria-expanded={expanded}
              >
                {expanded ? "Hide" : "Why this recommendation?"}
              </button>
              {expanded && (
                <ul className="navigation-card__reason-list">
                  {navigation.reason_codes.map((code) => (
                    <li key={code}>{REASON_LABEL[code] ?? code}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </>
      )}
    </section>
  );
}
