import type { FinalUC07Decision } from "../types";
import type { SortDirection, SortKey, SortState } from "../tableState";
import "./Uc07ResultsTable.css";

const DEST_LABEL: Record<string, string> = {
  PRIMARY_CARE: "Primary Care",
  URGENT_CARE: "Urgent Care",
  TELEHEALTH: "Telehealth",
  CARE_MANAGEMENT: "Care Management",
  NO_PROACTIVE_NAVIGATION: "No proactive navigation",
};

const SORTABLE_COLUMNS: { key: SortKey; label: string }[] = [
  { key: "member_id", label: "Member" },
  { key: "tier", label: "Risk tier" },
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
            <th scope="col">Details</th>
          </tr>
        </thead>
        <tbody>
          {decisions.map((decision) => {
            const isSelected = decision.member_id === selectedMemberId;
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
                <td className="tabular">{(decision.risk.probability * 100).toFixed(1)}%</td>
                <td>
                  {decision.navigation.destination
                    ? DEST_LABEL[decision.navigation.destination] ?? decision.navigation.destination
                    : "Suppressed by override"}
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
