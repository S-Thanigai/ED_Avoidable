import "./SafetyContextForm.css";

export interface SafetyContextFormValue {
  red_flag?: 0 | 1;
  icu?: 0 | 1;
  admitted?: 0 | 1;
  major_procedure?: 0 | 1;
  triage_level?: 1 | 2 | 3 | 4 | 5;
}

const BINARY_FIELDS: { key: keyof Omit<SafetyContextFormValue, "triage_level">; label: string }[] = [
  { key: "red_flag", label: "Red-flag symptom present" },
  { key: "icu", label: "ICU" },
  { key: "admitted", label: "Admitted" },
  { key: "major_procedure", label: "Major procedure in progress" },
];

// Tri-state: "" means UNKNOWN (field omitted from the request entirely,
// never sent as 0) -- this component must never collapse "unknown" into
// "false". Intended audience: care-management/clinical staff entering
// structured current-encounter context, not member self-report.
export function SafetyContextForm({
  value,
  onChange,
}: {
  value: SafetyContextFormValue;
  onChange: (next: SafetyContextFormValue) => void;
}) {
  const setBinary = (key: keyof Omit<SafetyContextFormValue, "triage_level">, raw: string) => {
    const next = { ...value };
    if (raw === "") delete next[key];
    else next[key] = (raw === "1" ? 1 : 0) as 0 | 1;
    onChange(next);
  };

  const setTriage = (raw: string) => {
    const next = { ...value };
    if (raw === "") delete next.triage_level;
    else next.triage_level = Number(raw) as 1 | 2 | 3 | 4 | 5;
    onChange(next);
  };

  return (
    <fieldset className="safety-context-form">
      <legend className="safety-context-form__legend">Current safety context (optional)</legend>
      <p className="safety-context-form__help">
        For care-management/clinical staff use. Enter only fields that are actually known for
        this encounter right now — leave a field on "Unknown" if it is not known. Unknown fields
        are never treated as safe/false.
      </p>

      <div className="safety-context-form__grid">
        {BINARY_FIELDS.map(({ key, label }) => (
          <label className="safety-context-form__field" key={key}>
            <span>{label}</span>
            <select
              value={value[key] === undefined ? "" : String(value[key])}
              onChange={(e) => setBinary(key, e.target.value)}
            >
              <option value="">Unknown</option>
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
          </label>
        ))}

        <label className="safety-context-form__field">
          <span>Triage level</span>
          <select
            value={value.triage_level === undefined ? "" : String(value.triage_level)}
            onChange={(e) => setTriage(e.target.value)}
          >
            <option value="">Unknown</option>
            <option value="1">1 (highest acuity)</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
            <option value="5">5 (lowest acuity)</option>
          </select>
        </label>
      </div>
    </fieldset>
  );
}
