import type { ContextCompleteness, ContextSource } from "../types";
import "./ContextStatus.css";

const COMPLETENESS_LABEL: Record<ContextCompleteness, string> = {
  COMPLETE: "Complete",
  PARTIAL: "Partial",
  ABSENT: "Not available",
};

const COMPLETENESS_HELP: Record<ContextCompleteness, string> = {
  COMPLETE: "Every current-safety field was explicitly supplied.",
  PARTIAL: "Some, but not all, current-safety fields were supplied. The remaining fields are unknown, not assumed safe.",
  ABSENT: "No current-safety information was supplied for this encounter. Unknown is never treated as safe.",
};

// Never call CALLER_SUPPLIED "verified" -- it is only a provenance
// label, not a confirmation that the information is accurate.
const SOURCE_LABEL: Record<ContextSource, string> = {
  CALLER_SUPPLIED: "User/caller supplied",
  SYSTEM_DERIVED: "System derived",
  NOT_AVAILABLE: "Not available",
};

export function ContextStatus({
  completeness,
  source,
}: {
  completeness: ContextCompleteness;
  source: ContextSource;
}) {
  return (
    <div className="context-status" aria-label="Current safety context status">
      <div className="context-status__row">
        <span className="context-status__label">Current safety context:</span>
        <span className={`context-status__badge context-status__badge--${completeness.toLowerCase()}`}>
          {COMPLETENESS_LABEL[completeness]}
        </span>
      </div>
      <p className="context-status__help">{COMPLETENESS_HELP[completeness]}</p>
      <div className="context-status__row">
        <span className="context-status__label">Source:</span>
        <span className="context-status__source">{SOURCE_LABEL[source]}</span>
      </div>
    </div>
  );
}
