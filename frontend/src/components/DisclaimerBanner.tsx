import "./DisclaimerBanner.css";

export function DisclaimerBanner() {
  return (
    <div className="disclaimer" role="note">
      <span className="disclaimer__icon" aria-hidden="true">
        ⚠
      </span>
      <p className="disclaimer__text">
        <strong>For care navigation only — never a reason to delay care.</strong> This
        tool flags patterns of potentially avoidable ED use and suggests a lower-acuity
        next step. It does not diagnose, triage, or determine medical necessity, and no
        output here should ever be read as discouraging emergency care. If you or a
        member may be experiencing a medical emergency, call 911 or go to the nearest
        emergency department immediately.
      </p>
    </div>
  );
}
