import "./Uc07DisclaimerBanner.css";

/** Persistent safety disclaimer for the UC07 view, wording consistent
 * with backend/agents/safety_policy.py's BASE_DISCLAIMER and the
 * Phase 5-7 documentation. Always visible, independent of whether a
 * decision has loaded yet. */
export function Uc07DisclaimerBanner() {
  return (
    <div className="uc07-disclaimer" role="note">
      <span className="uc07-disclaimer__icon" aria-hidden="true">
        ⚠
      </span>
      <p className="uc07-disclaimer__text">
        <strong>For care navigation only — never a reason to delay care.</strong> These
        recommendations apply only to non-emergency situations. Emergency care should never be
        delayed when emergency symptoms or high-acuity conditions are present. If you or a
        member may be experiencing a medical emergency, call 911 or go to the nearest emergency
        department immediately.
      </p>
    </div>
  );
}
