import { useMemo, useState } from "react";
import type { PatientRow, RiskCategory } from "../types";
import { CarePill, RiskPill } from "./Pills";
import "./LegacyPatientTable.css";

type SortKey = "member_id" | "age" | "risk_score";
type SortDir = "asc" | "desc";

const PAGE_SIZE = 25;

interface PatientTableProps {
  rows: PatientRow[];
  onSelect: (row: PatientRow) => void;
}

export function PatientTable({ rows, onSelect }: PatientTableProps) {
  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState<RiskCategory | "All">("All");
  const [sortKey, setSortKey] = useState<SortKey>("risk_score");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [page, setPage] = useState(1);

  const filtered = useMemo(() => {
    let next = rows;
    if (riskFilter !== "All") {
      next = next.filter((r) => r.risk_category === riskFilter);
    }
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      next = next.filter((r) => r.member_id.toLowerCase().includes(q));
    }
    const sorted = [...next].sort((a, b) => {
      let cmp = 0;
      if (sortKey === "member_id") cmp = a.member_id.localeCompare(b.member_id);
      else if (sortKey === "age") cmp = a.age - b.age;
      else cmp = a.risk_score - b.risk_score;
      return sortDir === "asc" ? cmp : -cmp;
    });
    return sorted;
  }, [rows, riskFilter, search, sortKey, sortDir]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const clampedPage = Math.min(page, pageCount);
  const pageRows = filtered.slice((clampedPage - 1) * PAGE_SIZE, clampedPage * PAGE_SIZE);

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "member_id" ? "asc" : "desc");
    }
    setPage(1);
  };

  const sortIndicator = (key: SortKey) =>
    key === sortKey ? (sortDir === "asc" ? "▲" : "▼") : "";

  return (
    <section className="patient-table-wrap">
      <div className="patient-table__controls">
        <input
          type="search"
          placeholder="Search by member ID…"
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(1);
          }}
          className="patient-table__search"
        />
        <div className="patient-table__filters" role="group" aria-label="Filter by risk">
          {(["All", "Low", "Medium", "High"] as const).map((opt) => (
            <button
              key={opt}
              type="button"
              className={`filter-chip${riskFilter === opt ? " filter-chip--active" : ""}`}
              onClick={() => {
                setRiskFilter(opt);
                setPage(1);
              }}
            >
              {opt}
            </button>
          ))}
        </div>
        <span className="patient-table__count">
          {filtered.length.toLocaleString()} of {rows.length.toLocaleString()} patients
        </span>
      </div>

      <div className="patient-table__scroll">
        <table className="patient-table">
          <thead>
            <tr>
              <th className="sortable" onClick={() => toggleSort("member_id")}>
                Member {sortIndicator("member_id")}
              </th>
              <th className="sortable" onClick={() => toggleSort("age")}>
                Age / gender {sortIndicator("age")}
              </th>
              <th className="sortable" onClick={() => toggleSort("risk_score")}>
                Risk score {sortIndicator("risk_score")}
              </th>
              <th>Risk category</th>
              <th>Recommended next step</th>
              <th aria-label="Actions" />
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row) => (
              <tr key={row.member_id} onClick={() => onSelect(row)} className="patient-row">
                <td className="patient-table__id">{row.member_id}</td>
                <td className="tabular">
                  {row.age} · {row.gender}
                </td>
                <td className="tabular patient-table__score">{row.risk_score.toFixed(1)}</td>
                <td>
                  <RiskPill category={row.risk_category} />
                </td>
                <td>
                  <CarePill care={row.recommended_alternative_care} />
                </td>
                <td>
                  <span className="patient-table__view">View →</span>
                </td>
              </tr>
            ))}
            {pageRows.length === 0 && (
              <tr>
                <td colSpan={6} className="patient-table__empty">
                  No patients match the current search/filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {pageCount > 1 && (
        <div className="patient-table__pagination">
          <button disabled={clampedPage === 1} onClick={() => setPage((p) => p - 1)}>
            ← Prev
          </button>
          <span>
            Page {clampedPage} of {pageCount}
          </span>
          <button disabled={clampedPage === pageCount} onClick={() => setPage((p) => p + 1)}>
            Next →
          </button>
        </div>
      )}
    </section>
  );
}
