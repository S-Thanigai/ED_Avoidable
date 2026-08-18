import { useState } from "react";
import { deletePopulation, PopulationsApiError } from "./api";
import "./DeletePopulationDialog.css";

/** Explicit, named confirmation before an irreversible delete -- clearly
 * identifies what's being deleted (population name + member count),
 * requires a distinct confirm click, and never affects another user's
 * data (ownership is enforced server-side regardless of what this
 * dialog shows). */
export function DeletePopulationDialog({
  populationId,
  populationName,
  memberCount,
  onCancel,
  onDeleted,
}: {
  populationId: number;
  populationName: string;
  memberCount: number;
  onCancel: () => void;
  onDeleted: () => void;
}) {
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDelete = async () => {
    setDeleting(true);
    setError(null);
    try {
      await deletePopulation(populationId);
      onDeleted();
    } catch (err) {
      setError(err instanceof PopulationsApiError || err instanceof Error ? err.message : "Delete failed.");
      setDeleting(false);
    }
  };

  return (
    <div className="delete-population-dialog__overlay" onClick={onCancel}>
      <div
        className="delete-population-dialog"
        onClick={(e) => e.stopPropagation()}
        role="alertdialog"
        aria-modal="true"
        aria-label="Confirm delete population"
      >
        <h3>Delete "{populationName}"?</h3>
        <p>
          This permanently removes this saved population, its {memberCount.toLocaleString()} member records, and
          all associated analysis results. This cannot be undone.
        </p>
        {error && (
          <div className="delete-population-dialog__error" role="alert">
            {error}
          </div>
        )}
        <div className="delete-population-dialog__actions">
          <button type="button" onClick={onCancel} disabled={deleting}>
            Cancel
          </button>
          <button type="button" className="delete-population-dialog__confirm" onClick={handleDelete} disabled={deleting}>
            {deleting ? "Deleting…" : "Delete Population"}
          </button>
        </div>
      </div>
    </div>
  );
}
