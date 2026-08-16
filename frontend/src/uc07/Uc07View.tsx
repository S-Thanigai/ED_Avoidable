import { useEffect, useMemo, useState } from "react";
import { UploadPanel } from "../components/UploadPanel";
import type { UploadFiles } from "../types";
import { decideUC07, UC07ApiError } from "./api";
import type { FinalUC07Decision, UC07DecideResponse } from "./types";
import { Uc07DisclaimerBanner } from "./components/Uc07DisclaimerBanner";
import { SyntheticDisclosure } from "./components/SyntheticDisclosure";
import { ModelInfo } from "./components/ModelInfo";
import { DecisionLoading } from "./components/DecisionLoading";
import { DecisionError } from "./components/DecisionError";
import { PopulationSummary } from "./components/PopulationSummary";
import { MemberFilters } from "./components/MemberFilters";
import { Uc07ResultsTable } from "./components/Uc07ResultsTable";
import { Pagination, PAGE_SIZE } from "./components/Pagination";
import { Uc07DecisionPanel } from "./components/Uc07DecisionPanel";
import { MemberDataSections } from "./components/MemberDataSections";
import { MemberDetailsDrawer } from "./components/MemberDetailsDrawer";
import { CurrentSafetyContextSection } from "./components/CurrentSafetyContextSection";
import { SafetyContextCsvUpload } from "./components/SafetyContextCsvUpload";
import { readAndParseUc07Files, type Uc07MemberDataLookups } from "./csvUtils";
import {
  DEFAULT_FILTERS,
  DEFAULT_SORT,
  filterDecisions,
  isFiltersActive,
  sortDecisions,
  type MemberFiltersState,
  type SortState,
} from "./tableState";
import "./Uc07View.css";

const EMPTY_FILES: UploadFiles = { members: null, edVisits: null, care: null };

/** Top-level orchestration for the authoritative UC07 flow. This
 * component and everything it renders is a VIEW/INTERACTION layer only
 * -- every risk tier, navigation destination, and safety state comes
 * from decideUC07() (POST /uc07/decide); nothing here computes one.
 * Filtering/sorting/pagination/selection are all presentation-layer
 * concerns over the already-decided batch (see tableState.ts). Current
 * safety context (single-member "Evaluate Current Safety" and the
 * optional batch CSV) is likewise collected here and handed to the
 * backend unmodified -- see CurrentSafetyContextSection.tsx and
 * docs/08B_CURRENT_SAFETY_CONTEXT_WORKFLOW.md. */
export function Uc07View() {
  const [files, setFiles] = useState<UploadFiles>(EMPTY_FILES);
  const [safetyContextCsvFile, setSafetyContextCsvFile] = useState<File | null>(null);
  const [memberId, setMemberId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<UC07ApiError | Error | null>(null);
  const [response, setResponse] = useState<UC07DecideResponse | null>(null);

  const [filters, setFiltersState] = useState<MemberFiltersState>(DEFAULT_FILTERS);
  const [sort, setSort] = useState<SortState>(DEFAULT_SORT);
  const [page, setPage] = useState(1);
  const [selectedMemberId, setSelectedMemberId] = useState<string | null>(null);

  // Per-member decisions refreshed via "Evaluate Current Safety" --
  // supersede the original batch decision for that member_id only, in
  // every downstream view (table row, summary counts, drawer). Reset on
  // every new decideUC07() run (a different batch is a different dataset).
  const [safetyOverrides, setSafetyOverrides] = useState<Map<string, FinalUC07Decision>>(new Map());

  const [lookups, setLookups] = useState<Uc07MemberDataLookups | null>(null);
  const [lookupsLoading, setLookupsLoading] = useState(false);

  const handleFileChange = (key: keyof UploadFiles, file: File | null) => {
    setFiles((prev) => ({ ...prev, [key]: file }));
  };

  // Changing a filter always resets pagination to page 1.
  const updateFilters = (next: MemberFiltersState) => {
    setFiltersState(next);
    setPage(1);
  };

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    // A fresh result set is a different dataset -- start clean rather
    // than carrying over filters/sort/page/selection from a previous run.
    setResponse(null);
    setSelectedMemberId(null);
    setFiltersState(DEFAULT_FILTERS);
    setSort(DEFAULT_SORT);
    setPage(1);
    setLookups(null);
    setSafetyOverrides(new Map());
    try {
      const trimmedMemberId = memberId.trim();
      const result = await decideUC07({
        files,
        memberId: trimmedMemberId || undefined,
        safetyContextFile: safetyContextCsvFile ?? undefined,
      });
      setResponse(result);
      if (result.decisions.length === 1) setSelectedMemberId(result.decisions[0].member_id);
    } catch (err) {
      setError(err instanceof UC07ApiError || err instanceof Error ? err : new Error("Something went wrong."));
    } finally {
      setLoading(false);
    }
  };

  // Called after a successful "Evaluate Current Safety" -- the backend
  // Safety Agent already made the determination; this just stores the
  // fresh decision so every view (table, summary, drawer) reflects it.
  const handleSafetyEvaluated = (updated: FinalUC07Decision) => {
    setSafetyOverrides((prev) => {
      const next = new Map(prev);
      next.set(updated.member_id, updated);
      return next;
    });
  };

  // Parse the same three CSV files already uploaded for this decision
  // batch, client-side, so the details view can show profile/
  // utilization/access/care-history data without any backend/API change.
  useEffect(() => {
    if (!response) return;
    let cancelled = false;
    setLookupsLoading(true);
    readAndParseUc07Files(files)
      .then((result) => {
        if (!cancelled) setLookups(result);
      })
      .catch(() => {
        if (!cancelled) setLookups(null);
      })
      .finally(() => {
        if (!cancelled) setLookupsLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // `files` deliberately excluded: only the successful response should
    // trigger a re-parse, not every keystroke while choosing new files.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [response]);

  const allDecisions = useMemo(() => response?.decisions ?? [], [response]);
  // Apply any per-member "Evaluate Current Safety" overrides on top of
  // the original batch -- every downstream view uses this merged set.
  const effectiveDecisions = useMemo(
    () => allDecisions.map((d) => safetyOverrides.get(d.member_id) ?? d),
    [allDecisions, safetyOverrides],
  );
  const filtered = useMemo(() => filterDecisions(effectiveDecisions, filters), [effectiveDecisions, filters]);
  const sorted = useMemo(() => sortDecisions(filtered, sort), [filtered, sort]);
  const paged = useMemo(() => sorted.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE), [sorted, page]);
  const filtersActive = isFiltersActive(filters);

  const selectedDecision: FinalUC07Decision | null = selectedMemberId
    ? (effectiveDecisions.find((d) => d.member_id === selectedMemberId) ?? null)
    : null;

  const allSelected = Boolean(files.members && files.edVisits && files.care);
  const isBatch = allDecisions.length > 1;

  return (
    <div className="uc07-view">
      <Uc07DisclaimerBanner />

      <div className="uc07-view__intro">
        <SyntheticDisclosure modelVersion={response?.model_version} />
        <ModelInfo />
      </div>

      <h2 className="uc07-view__section-heading">Historical data <span className="uc07-view__required-badge">Required</span></h2>
      <UploadPanel files={files} onFileChange={handleFileChange} onRun={handleRun} loading={loading} />

      <SafetyContextCsvUpload file={safetyContextCsvFile} onChange={setSafetyContextCsvFile} disabled={loading} />

      <section className="uc07-view__options">
        <label className="uc07-view__member-field">
          <span>Member ID (optional — leave blank to score the full population)</span>
          <input
            type="text"
            value={memberId}
            onChange={(e) => setMemberId(e.target.value)}
            placeholder="e.g. M00001"
            disabled={loading}
          />
        </label>
      </section>

      <button
        type="button"
        className="run-button uc07-view__run"
        disabled={!allSelected || loading}
        onClick={handleRun}
      >
        {loading ? "Getting decision…" : "Get UC07 decision"}
      </button>

      {loading && <DecisionLoading />}
      {error && !loading && <DecisionError error={error} onDismiss={() => setError(null)} />}

      {!loading && !error && response && (
        <>
          {isBatch && (
            <>
              <PopulationSummary decisions={filtered} totalCount={effectiveDecisions.length} filtersActive={filtersActive} />
              <MemberFilters
                filters={filters}
                onChange={updateFilters}
                totalCount={effectiveDecisions.length}
                filteredCount={filtered.length}
              />

              {sorted.length === 0 ? (
                <div className="uc07-view__empty">
                  <p>No members match the selected filters.</p>
                  <button type="button" className="uc07-view__empty-clear" onClick={() => updateFilters(DEFAULT_FILTERS)}>
                    Clear filters
                  </button>
                </div>
              ) : (
                <>
                  <Uc07ResultsTable
                    decisions={paged}
                    onSelect={(d) => setSelectedMemberId(d.member_id)}
                    selectedMemberId={selectedMemberId}
                    sort={sort}
                    onSortChange={setSort}
                  />
                  <Pagination page={page} totalItems={sorted.length} onPageChange={setPage} />
                </>
              )}
            </>
          )}

          {!isBatch && selectedDecision && (
            <>
              <Uc07DecisionPanel decision={selectedDecision} />
              <MemberDataSections memberId={selectedDecision.member_id} lookups={lookups} loading={lookupsLoading} />
              <CurrentSafetyContextSection
                decision={selectedDecision}
                files={files}
                indexDate={response.index_date}
                onEvaluated={handleSafetyEvaluated}
              />
            </>
          )}
        </>
      )}

      {isBatch && selectedDecision && response && (
        <MemberDetailsDrawer
          decision={selectedDecision}
          lookups={lookups}
          lookupsLoading={lookupsLoading}
          files={files}
          indexDate={response.index_date}
          onSafetyEvaluated={handleSafetyEvaluated}
          onClose={() => setSelectedMemberId(null)}
        />
      )}
    </div>
  );
}
