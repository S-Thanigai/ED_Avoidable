import "./EmptyState.css";

export function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-state__icon" aria-hidden="true">
        🧭
      </div>
      <h2 className="empty-state__title">Upload data to spot avoidable ED patterns</h2>
      <p className="empty-state__text">
        Upload the members, ED visit, and care history CSVs above to score every
        member for potentially avoidable ED utilization and get a suggested
        lower-acuity next step — primary care, urgent care, telehealth, or a
        care-management follow-up.
      </p>
    </div>
  );
}
