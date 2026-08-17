import { useState } from "react";
import type { UploadFiles } from "../../types";
import type { FinalUC07Decision } from "../types";
import { decideUC07, UC07ApiError } from "../api";
import { ContextStatus } from "./ContextStatus";
import { SafetyContextForm, type SafetyContextFormValue } from "./SafetyContextForm";
import "./CurrentSafetyContextSection.css";

const STATE_LABEL: Record<FinalUC07Decision["safety"]["state"], string> = {
  CLEAR: "Clear",
  CAUTION: "Caution",
  OVERRIDE: "Override",
};

// Concise, non-clinical outcome descriptions (Section 16). CLEAR never
// implies "clinically cleared" -- it describes only what current
// information was supplied and whether a configured rule fired.
const STATE_DESCRIPTION: Record<FinalUC07Decision["safety"]["state"], string> = {
  CLEAR: "Complete current context with no override indicator.",
  CAUTION: "Current context is missing or incomplete.",
  OVERRIDE: "Current information contains an emergency/high-acuity indicator.",
};

function yesNoUnknown(value: 0 | 1 | undefined): string {
  if (value === undefined) return "Unknown";
  return value === 1 ? "Yes" : "No";
}

/**
 * "Current Safety Context" section for the member details view (used in
 * both the batch drawer and the single-member inline view). This is a
 * pure collect-and-display component: entering values and clicking
 * "Evaluate Current Safety" sends them to the SAME authoritative
 * POST /uc07/decide endpoint, scoped to this one member, and displays
 * whatever CLEAR/CAUTION/OVERRIDE the backend Safety Agent returns.
 * This component never itself decides a safety state.
 */
export function CurrentSafetyContextSection({
  decision,
  files,
  indexDate,
  onEvaluated,
}: {
  decision: FinalUC07Decision;
  files: UploadFiles;
  indexDate: string;
  onEvaluated: (updated: FinalUC07Decision) => void;
}) {
  const [formValue, setFormValue] = useState<SafetyContextFormValue>({});
  const [submitted, setSubmitted] = useState<SafetyContextFormValue | null>(null);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleEvaluate = async () => {
    setEvaluating(true);
    setError(null);
    try {
      const hasAnyField = Object.values(formValue).some((v) => v !== undefined);
      const result = await decideUC07({
        files,
        memberId: decision.member_id,
        indexDate,
        currentSafetyContext: hasAnyField ? { [decision.member_id]: formValue } : undefined,
      });
      const updated = result.decisions[0];
      if (!updated) throw new UC07ApiError("Backend returned no decision for this member.", null);
      setSubmitted({ ...formValue });
      onEvaluated(updated);
    } catch (err) {
      setError(err instanceof UC07ApiError || err instanceof Error ? err.message : "Could not evaluate safety context.");
    } finally {
      setEvaluating(false);
    }
  };

  const { safety } = decision;

  return (
    <section className="current-safety-context" aria-label="Current safety context">
      <h3 className="current-safety-context__heading">Current Safety Context</h3>

      <div className="current-safety-context__summary">
        <div className="current-safety-context__summary-row">
          <span>Status</span>
          <strong className={`current-safety-context__state current-safety-context__state--${safety.state.toLowerCase()}`}>
            {STATE_LABEL[safety.state]}
          </strong>
        </div>
        <ContextStatus completeness={safety.context_completeness} source={safety.context_source} />
      </div>

      <dl className="current-safety-context__legend" aria-label="Possible safety outcomes">
        {(["CLEAR", "CAUTION", "OVERRIDE"] as const).map((state) => (
          <div
            key={state}
            className={`current-safety-context__legend-row${state === safety.state ? " current-safety-context__legend-row--active" : ""}`}
          >
            <dt className={`current-safety-context__legend-term current-safety-context__legend-term--${state.toLowerCase()}`}>
              {STATE_LABEL[state]}
            </dt>
            <dd>{STATE_DESCRIPTION[state]}</dd>
          </div>
        ))}
      </dl>

      {submitted && (
        <dl className="current-safety-context__fields">
          <div>
            <dt>Red flag</dt>
            <dd>{yesNoUnknown(submitted.red_flag)}</dd>
          </div>
          <div>
            <dt>ICU</dt>
            <dd>{yesNoUnknown(submitted.icu)}</dd>
          </div>
          <div>
            <dt>Admitted</dt>
            <dd>{yesNoUnknown(submitted.admitted)}</dd>
          </div>
          <div>
            <dt>Major procedure</dt>
            <dd>{yesNoUnknown(submitted.major_procedure)}</dd>
          </div>
          <div>
            <dt>Triage level</dt>
            <dd>{submitted.triage_level ?? "Unknown"}</dd>
          </div>
        </dl>
      )}

      <SafetyContextForm value={formValue} onChange={setFormValue} />

      {error && (
        <p className="current-safety-context__error" role="alert">
          {error}
        </p>
      )}

      <button
        type="button"
        className="current-safety-context__evaluate"
        onClick={handleEvaluate}
        disabled={evaluating}
      >
        {evaluating ? "Evaluating…" : "Evaluate Current Safety"}
      </button>
    </section>
  );
}
