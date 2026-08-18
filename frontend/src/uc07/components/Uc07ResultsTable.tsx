import type { FinalUC07Decision } from "../types";
import type { SortDirection, SortKey, SortState } from "../tableState";
import { NAVIGATION_DESTINATION_COLOR, NAVIGATION_DESTINATION_LABEL } from "../navigationDisplay";
import "./Uc07ResultsTable.css";

const SAFETY_LABEL: Record<string, string> = {
  CLEAR: "Clear",
  CAUTION: "Caution",
  OVERRIDE: "Override",
};

const SORTABLE_COLUMNS: { key: SortKey; label: string }[] = [
  { key: "member_id", label: "Member" },
  { key: "tier", label: "Risk Tier" },
  { key: "probability", label: "Probability" },
  { key: "navigation", label: "Navigation" },
];

function SortIndicator({ active, direction }: { active: boolean; direction: SortDirection }) {
  if (!active) return <span className="uc07-results-table__sort-icon" aria-hidden="true">↕</span>;
  return (
    <span className="uc07-results-table__sort-icon uc07-results-table__sort-icon--active" aria-hidden="true">
      {direction === "asc" ? "↑" : "↓"}
    </span>
  );
}

export function Uc07ResultsTable({
  decisions,
  onSelect,
  selectedMemberId,
  sort,
  onSortChange,
}: {
  decisions: FinalUC07Decision[];
  onSelect: (decision: FinalUC07Decision) => void;
  selectedMemberId?: string | null;
  sort: SortState;
  onSortChange: (next: SortState) => void;
}) {
  const toggleSort = (key: SortKey) => {
    if (sort.key !== key) {
      onSortChange({ key, direction: "asc" });
    } else {
      onSortChange({ key, direction: sort.direction === "asc" ? "desc" : "asc" });
    }
  };

  return (
    <div className="uc07-results-table__wrap">
      <table className="uc07-results-table">
        <caption className="sr-only">UC07 decisions by member</caption>
        <thead>
          <tr>
            {SORTABLE_COLUMNS.map((col) => (
              <th key={col.key} scope="col" aria-sort={sort.key === col.key ? (sort.direction === "asc" ? "ascending" : "descending") : "none"}>
                <button type="button" className="uc07-results-table__sort-button" onClick={() => toggleSort(col.key)}>
                  {col.label}
                  <SortIndicator active={sort.key === col.key} direction={sort.direction} />
                </button>
              </th>
            ))}
            <th scope="col">Safety</th>
            <th scope="col">Action</th>
          </tr>
        </thead>
        <tbody>
          {decisions.map((decision) => {
            const isSelected = decision.member_id === selectedMemberId;
            const pct = decision.risk.probability * 100;
            return (
              <tr
                key={decision.member_id}
                className={isSelected ? "uc07-results-table__row--selected" : undefined}
                aria-selected={isSelected}
              >
                <td>
                  <button
                    type="button"
                    className="uc07-results-table__member-link"
                    onClick={() => onSelect(decision)}
                  >
                    {decision.member_id}
                  </button>
                </td>
                <td>
                  <span className={`uc07-results-table__tier uc07-results-table__tier--${decision.risk.tier.toLowerCase()}`}>
                    {decision.risk.tier}
                  </span>
                </td>
                <td className="tabular">
                  <div className="uc07-results-table__prob">
                    <span className="uc07-results-table__prob-value">{pct.toFixed(1)}%</span>
                    <span className="uc07-results-table__prob-bar" aria-hidden="true">
                      <span
                        className={`uc07-results-table__prob-bar-fill uc07-results-table__prob-bar-fill--${decision.risk.tier.toLowerCase()}`}
                        style={{ width: `${Math.min(100, pct)}%` }}
                      />
                    </span>
                  </div>
                </td>
                <td>
                  {decision.navigation.destination ? (
                    <span className="uc07-results-table__nav">
                      <span
                        className="uc07-results-table__nav-dot"
                        aria-hidden="true"
                        style={{ background: NAVIGATION_DESTINATION_COLOR[decision.navigation.destination] }}
                      />
                      {NAVIGATION_DESTINATION_LABEL[decision.navigation.destination] ?? decision.navigation.destination}
                    </span>
                  ) : (
                    "Suppressed by override"
                  )}
                </td>
                <td>
                  <span className={`uc07-results-table__safety uc07-results-table__safety--${decision.safety.state.toLowerCase()}`}>
                    {decision.safety.state === "OVERRIDE" && (
                      <svg
                        className="uc07-results-table__safety-icon"
                        aria-hidden="true"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.4"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <path d="M12 3 2 20h20L12 3Z" />
                        <path d="M12 10v4M12 17h.01" />
                      </svg>
                    )}
                    {SAFETY_LABEL[decision.safety.state] ?? decision.safety.state}
                  </span>
                </td>
                <td>
                  <button
                    type="button"
                    className="uc07-results-table__details-button"
                    onClick={() => onSelect(decision)}
                    aria-label={`View details for member ${decision.member_id}`}
                  >
                    View details <span aria-hidden="true">→</span>
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
