import { useEffect, useMemo, useState } from "react";
import { UploadPanel } from "../components/UploadPanel";
import type { UploadFiles } from "../types";
import { decideUC07, UC07ApiError } from "./api";
import type { FinalUC07Decision, UC07DecideResponse } from "./types";
import { Uc07DisclaimerBanner } from "./components/Uc07DisclaimerBanner";
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
import { SaveAnalysisPrompt } from "../populations/SaveAnalysisPrompt";
import {
  DEFAULT_FILTERS,
  DEFAULT_SORT,
  filterDecisions,
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
  // Setup -> Results transition (Section 8): once a run succeeds, the
  // Data Inputs section compacts into a one-line summary so the
  // population results -- not the upload UI -- occupy the top of the
  // viewport. "Change files" re-expands it without discarding the
  // results currently on screen; only running again replaces them.
  const [inputsExpanded, setInputsExpanded] = useState(true);

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
      setInputsExpanded(false);
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

  // Phase 8D Part 8 -- `sorted` can shrink for reasons OTHER than an
  // explicit filter change (which already resets to page 1 via
  // updateFilters): most notably, "Evaluate Current Safety" can change a
  // member's safety.state such that they no longer match an ACTIVE
  // filter, silently shrinking `sorted` while `page` stays wherever the
  // user was. Without this clamp, `sorted.slice(...)` for a page beyond
  // the new end returns `[]` while `sorted.length` is still > 0, so the
  // "no members match" empty state (keyed off `sorted.length === 0`)
  // never fires either -- the table just renders blank with no
  // explanation. `clampedPage` (not raw `page`) is used to compute
  // `paged` below AND is what's passed to <Pagination>, so a blank page
  // is never rendered even for the one render cycle before the
  // corrective effect below commits the clamped value back into `page`
  // state (needed so a later "Next"/"Previous" click starts from the
  // right place, not the stale one).
  const maxPage = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const clampedPage = Math.min(page, maxPage);
  useEffect(() => {
    if (page !== clampedPage) setPage(clampedPage);
  }, [clampedPage, page]);

  const paged = useMemo(() => sorted.slice((clampedPage - 1) * PAGE_SIZE, clampedPage * PAGE_SIZE), [sorted, clampedPage]);

  const selectedDecision: FinalUC07Decision | null = selectedMemberId
    ? (effectiveDecisions.find((d) => d.member_id === selectedMemberId) ?? null)
    : null;

  const isBatch = allDecisions.length > 1;

  return (
    <div className="uc07-view">
      <Uc07DisclaimerBanner />

      {inputsExpanded || !response ? (
        <section className="uc07-view__data-inputs" aria-label="Data inputs setup">
          <h2 className="uc07-view__section-heading">
            Data Inputs <span className="uc07-view__required-badge">Required</span>
          </h2>

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

          <UploadPanel files={files} onFileChange={handleFileChange} onRun={handleRun} loading={loading} />

          <SafetyContextCsvUpload file={safetyContextCsvFile} onChange={setSafetyContextCsvFile} disabled={loading} />
        </section>
      ) : (
        <section className="uc07-view__data-inputs-summary" aria-label="Data sources (loaded)">
          <span className="uc07-view__data-inputs-summary-heading">Data Sources</span>
          <span className="uc07-view__data-inputs-summary-item uc07-view__data-inputs-summary-item--members">
            <span className="uc07-view__data-inputs-summary-check" aria-hidden="true">
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3.5 8.5 6.5 11.5 12.5 4.5" />
              </svg>
            </span>
            <span className="uc07-view__data-inputs-summary-text">
              <strong>Members</strong>
              <span>{allDecisions.length.toLocaleString()} loaded{files.members ? ` · ${files.members.name}` : ""}</span>
            </span>
          </span>
          <span className="uc07-view__data-inputs-summary-item uc07-view__data-inputs-summary-item--ed">
            <span className="uc07-view__data-inputs-summary-check" aria-hidden="true">
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3.5 8.5 6.5 11.5 12.5 4.5" />
              </svg>
            </span>
            <span className="uc07-view__data-inputs-summary-text">
              <strong>ED Visits</strong>
              <span>{files.edVisits ? files.edVisits.name : "loaded"}</span>
            </span>
          </span>
          <span className="uc07-view__data-inputs-summary-item uc07-view__data-inputs-summary-item--care">
            <span className="uc07-view__data-inputs-summary-check" aria-hidden="true">
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3.5 8.5 6.5 11.5 12.5 4.5" />
              </svg>
            </span>
            <span className="uc07-view__data-inputs-summary-text">
              <strong>Care History</strong>
              <span>{files.care ? files.care.name : "loaded"}</span>
            </span>
          </span>
          {safetyContextCsvFile && (
            <span className="uc07-view__data-inputs-summary-item uc07-view__data-inputs-summary-item--optional">
              <span className="uc07-view__data-inputs-summary-check" aria-hidden="true">
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M3.5 8.5 6.5 11.5 12.5 4.5" />
                </svg>
              </span>
              <span className="uc07-view__data-inputs-summary-text">
                <strong>Current Safety</strong>
                <span>{safetyContextCsvFile.name}</span>
              </span>
            </span>
          )}
          <button type="button" className="uc07-view__change-files" onClick={() => setInputsExpanded(true)}>
            Change files
          </button>
        </section>
      )}

      {loading && <DecisionLoading />}
      {error && !loading && <DecisionError error={error} onDismiss={() => setError(null)} />}

      {!loading && !error && response && (
        <>
          <SaveAnalysisPrompt
            files={files}
            indexDate={response.index_date}
            safetyContextFile={safetyContextCsvFile}
            memberCount={allDecisions.length}
          />

          {isBatch && (
            <>
              <PopulationSummary decisions={effectiveDecisions} filters={filters} onFilterChange={updateFilters} />
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
                  <Pagination page={clampedPage} totalItems={sorted.length} onPageChange={setPage} />
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
