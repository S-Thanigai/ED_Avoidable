import { useRef } from "react";
import "./SafetyContextCsvUpload.css";

/** Optional fourth upload for a batch current_safety_context.csv
 * (member_id, red_flag, icu, admitted, major_procedure, triage_level).
 * The three historical CSVs remain required and unaffected; this file
 * is used only by the Safety Agent and never influences ML risk
 * scoring -- see docs/08B_CURRENT_SAFETY_CONTEXT_WORKFLOW.md. */
export function SafetyContextCsvUpload({
  file,
  onChange,
  disabled,
}: {
  file: File | null;
  onChange: (file: File | null) => void;
  disabled: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="safety-csv-upload">
      <div className="safety-csv-upload__label-row">
        <span className="safety-csv-upload__title">Current safety context</span>
        <span className="safety-csv-upload__badge">Optional</span>
      </div>
      <p className="safety-csv-upload__help">
        Optional current-encounter information used only by the Safety Agent. It does not affect
        ML risk prediction. Columns: member_id, red_flag, icu, admitted, major_procedure,
        triage_level. A member with no row here is treated as having no current safety
        information (CAUTION).
      </p>
      <div className="safety-csv-upload__control">
        <input
          ref={inputRef}
          id="safety-context-csv-input"
          type="file"
          accept=".csv"
          disabled={disabled}
          onChange={(e) => onChange(e.target.files?.[0] ?? null)}
        />
        <label htmlFor="safety-context-csv-input" className="safety-csv-upload__button">
          {file ? "Replace file" : "Choose current_safety_context.csv"}
        </label>
        {file && (
          <span className="safety-csv-upload__filename">
            {file.name}
            <button
              type="button"
              className="safety-csv-upload__remove"
              onClick={() => {
                onChange(null);
                if (inputRef.current) inputRef.current.value = "";
              }}
              aria-label="Remove current safety context file"
              disabled={disabled}
            >
              ×
            </button>
          </span>
        )}
      </div>
    </div>
  );
}
