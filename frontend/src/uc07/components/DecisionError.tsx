import { UC07ApiError } from "../api";
import "./DecisionError.css";

/** Renders a UC07ApiError safely -- never a raw stack trace, never a
 * fabricated recommendation. If the backend cannot be reached or
 * returns an error, this is the ONLY thing shown in place of a
 * decision; no client-side navigation/risk/safety guess is ever
 * substituted. */
export function DecisionError({ error, onDismiss }: { error: UC07ApiError | Error; onDismiss?: () => void }) {
  const status = error instanceof UC07ApiError ? error.status : null;
  const kind = classify(status);

  return (
    <div className="decision-error" role="alert">
      <div className="decision-error__header">
        <span className="decision-error__icon" aria-hidden="true">
          ⛔
        </span>
        <span className="decision-error__title">{kind.title}</span>
      </div>
      <p className="decision-error__message">{error.message}</p>
      <p className="decision-error__safety-note">
        Decision unavailable — no navigation recommendation is being shown. This is not a
        clinical judgment; if this is a medical emergency, call 911 or go to the nearest
        emergency department immediately.
      </p>
      {onDismiss && (
        <button type="button" className="decision-error__dismiss" onClick={onDismiss}>
          Dismiss
        </button>
      )}
    </div>
  );
}

function classify(status: number | null): { title: string } {
  if (status === null) return { title: "Backend unavailable" };
  if (status === 404) return { title: "Member not found" };
  if (status === 422) return { title: "Invalid request data" };
  if (status === 503) return { title: "UC07 model unavailable" };
  if (status >= 500) return { title: "Backend error" };
  return { title: "Request failed" };
}
