import "./DecisionLoading.css";

export function DecisionLoading({ label = "Requesting decision from the ED Navigator backend…" }: { label?: string }) {
  return (
    <div className="decision-loading" role="status" aria-live="polite">
      <span className="decision-loading__spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}
