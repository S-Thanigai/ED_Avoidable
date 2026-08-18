import { useEffect, useState } from "react";
import type { FinalUC07Decision, NavigationDestination, RiskTier, SafetyState } from "../uc07/types";
import { MemberFilters } from "../uc07/components/MemberFilters";
import { Uc07ResultsTable } from "../uc07/components/Uc07ResultsTable";
import { Pagination, PAGE_SIZE } from "../uc07/components/Pagination";
import { MemberDetailsDrawer } from "../uc07/components/MemberDetailsDrawer";
import { DEFAULT_FILTERS, DEFAULT_SORT, type MemberFiltersState, type SortState } from "../uc07/tableState";
import type { UploadFiles } from "../types";
import { getMember, getPopulation, listMembers, PopulationsApiError } from "./api";
import type { MemberDetail, PaginatedMembers, PopulationDetail } from "./types";
import { memberDetailToLookups } from "./lookupsAdapter";
import { SavedNavigationBar, SavedProbabilityHistogram, SavedRiskDonut, SavedSafetyDonut } from "./SavedAnalyticsCharts";
import { DeletePopulationDialog } from "./DeletePopulationDialog";
import "../uc07/components/PopulationSummary.css";
import "./SavedPopulationView.css";

// No CSV is ever re-uploaded for a saved population -- CurrentSafetyContextSection
// (rendered inside the reused MemberDetailsDrawer) degrades gracefully
// when asked to "Evaluate Current Safety" without files: decideUC07()
// rejects immediately with a clear, non-crashing message ("Members, ED
// visits, and care history CSVs are all required."), since it never
// even reaches the network. Every OTHER part of the drawer (risk,
// navigation, persisted safety state, SHAP, AI explanation, PDF, email)
// works identically to the live CSV pathway.
const NO_FILES: UploadFiles = { members: null, edVisits: null, care: null };

function toApiParams(filters: MemberFiltersState, sort: SortState, page: number) {
  return {
    page,
    page_size: PAGE_SIZE,
    search: filters.search.trim() || undefined,
    tier: filters.tier !== "ALL" ? filters.tier : undefined,
    navigation: filters.navigation !== "ALL" ? filters.navigation : undefined,
    safety: filters.safety !== "ALL" ? filters.safety : undefined,
    prob_min: filters.probMin !== "" ? Number(filters.probMin) / 100 : undefined,
    prob_max: filters.probMax !== "" ? Number(filters.probMax) / 100 : undefined,
    sort_key: sort.key ?? undefined,
    sort_dir: sort.direction,
  };
}

export function SavedPopulationView({
  populationId,
  onBack,
  onDeleted,
}: {
  populationId: number;
  onBack: () => void;
  onDeleted: () => void;
}) {
  const [population, setPopulation] = useState<PopulationDetail | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  const [filters, setFiltersState] = useState<MemberFiltersState>(DEFAULT_FILTERS);
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [sort, setSort] = useState<SortState>(DEFAULT_SORT);
  const [page, setPage] = useState(1);

  const [membersResult, setMembersResult] = useState<PaginatedMembers | null>(null);
  const [membersLoading, setMembersLoading] = useState(true);
  const [membersError, setMembersError] = useState<string | null>(null);

  const [selectedMemberId, setSelectedMemberId] = useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<MemberDetail | null>(null);
  const [selectedLoading, setSelectedLoading] = useState(false);

  const [deleteOpen, setDeleteOpen] = useState(false);

  // Debounce the free-text search so typing doesn't fire a server
  // request per keystroke -- every other filter (tier/navigation/
  // safety/probability) is a discrete select/number input and applies
  // immediately.
  useEffect(() => {
    const t = window.setTimeout(() => setDebouncedSearch(filters.search), 300);
    return () => window.clearTimeout(t);
  }, [filters.search]);

  useEffect(() => {
    let cancelled = false;
    getPopulation(populationId)
      .then((result) => {
        if (!cancelled) setPopulation(result);
      })
      .catch((err) => {
        if (!cancelled) {
          setSummaryError(err instanceof PopulationsApiError || err instanceof Error ? err.message : "Failed to load.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [populationId]);

  useEffect(() => {
    let cancelled = false;
    setMembersLoading(true);
    setMembersError(null);
    const params = toApiParams({ ...filters, search: debouncedSearch }, sort, page);
    listMembers(populationId, params)
      .then((result) => {
        if (cancelled) return;
        setMembersResult(result);
        setMembersLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setMembersError(err instanceof PopulationsApiError || err instanceof Error ? err.message : "Failed to load members.");
        setMembersLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // Deliberately depends on individual filter fields (not `filters` as
    // a whole) plus `debouncedSearch` rather than `filters.search`, so
    // typing in the search box doesn't fire a request per keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [populationId, filters.tier, filters.navigation, filters.safety, filters.probMin, filters.probMax, debouncedSearch, sort, page]);

  useEffect(() => {
    if (!selectedMemberId) {
      setSelectedDetail(null);
      return;
    }
    let cancelled = false;
    setSelectedLoading(true);
    getMember(populationId, selectedMemberId)
      .then((detail) => {
        if (!cancelled) setSelectedDetail(detail);
      })
      .catch(() => {
        if (!cancelled) setSelectedDetail(null);
      })
      .finally(() => {
        if (!cancelled) setSelectedLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [populationId, selectedMemberId]);

  const updateFilters = (next: MemberFiltersState) => {
    setFiltersState(next);
    setPage(1);
  };

  const items = membersResult?.items ?? [];
  const totalItems = membersResult?.total_items ?? 0;

  return (
    <div className="saved-population-view">
      <div className="saved-population-view__header">
        <button type="button" className="saved-population-view__back" onClick={onBack}>
          ← Back to Dashboard
        </button>
        {population && (
          <button type="button" className="saved-population-view__delete" onClick={() => setDeleteOpen(true)}>
            Delete Population
          </button>
        )}
      </div>

      {summaryError && (
        <div className="saved-population-view__error" role="alert">
          {summaryError}
        </div>
      )}

      {population && (
        <>
          <div className="saved-population-view__title-row">
            <h1>{population.name}</h1>
            <span className="saved-population-view__meta">
              {population.member_count.toLocaleString()} members · analyzed as of {population.index_date} ·{" "}
              {population.synthetic_model ? "Synthetic demonstration model" : population.model_version}
            </span>
          </div>

          <div className="population-summary__charts">
            <SavedRiskDonut
              tierCounts={population.tier_counts}
              activeTier={filters.tier === "ALL" ? null : (filters.tier as RiskTier)}
              onSelectTier={(tier) => updateFilters({ ...filters, tier: tier ?? "ALL" })}
            />
            <SavedNavigationBar
              navigationCounts={population.navigation_counts}
              activeDestination={filters.navigation === "ALL" ? null : (filters.navigation as NavigationDestination)}
              onSelectDestination={(destination) => updateFilters({ ...filters, navigation: destination ?? "ALL" })}
            />
            <SavedSafetyDonut
              safetyCounts={population.safety_counts}
              activeSafety={filters.safety === "ALL" ? null : (filters.safety as SafetyState)}
              onSelectSafety={(state) => updateFilters({ ...filters, safety: state ?? "ALL" })}
            />
            <SavedProbabilityHistogram
              bins={population.probability_bins}
              moderateThreshold={population.moderate_threshold}
              highThreshold={population.high_threshold}
              activeBin={filters.probMin !== "" ? filters.probMin : null}
              onSelectBin={(bin) =>
                updateFilters({
                  ...filters,
                  probMin: bin ? String(bin.min) : "",
                  probMax: bin && bin.max !== null ? String(bin.max) : "",
                })
              }
            />
          </div>
        </>
      )}

      <MemberFilters filters={filters} onChange={updateFilters} totalCount={population?.member_count ?? 0} filteredCount={totalItems} />

      {membersError && (
        <div className="saved-population-view__error" role="alert">
          {membersError}
        </div>
      )}

      {membersLoading ? (
        <p className="saved-population-view__status">Loading members…</p>
      ) : items.length === 0 ? (
        <div className="saved-population-view__empty">
          <p>No members match the selected filters.</p>
          <button type="button" onClick={() => updateFilters(DEFAULT_FILTERS)}>
            Clear filters
          </button>
        </div>
      ) : (
        <>
          <Uc07ResultsTable
            decisions={items}
            onSelect={(d: FinalUC07Decision) => setSelectedMemberId(d.member_id)}
            selectedMemberId={selectedMemberId}
            sort={sort}
            onSortChange={(next) => {
              setSort(next);
              setPage(1);
            }}
          />
          <Pagination page={membersResult?.page ?? 1} totalItems={totalItems} onPageChange={setPage} />
        </>
      )}

      {selectedMemberId && selectedDetail && (
        <MemberDetailsDrawer
          decision={selectedDetail.decision}
          lookups={memberDetailToLookups(selectedDetail)}
          lookupsLoading={selectedLoading}
          files={NO_FILES}
          indexDate={population?.index_date ?? selectedDetail.decision.risk.index_date}
          onSafetyEvaluated={() => {
            /* saved-population "Evaluate Current Safety" has no CSV to
               re-run against -- decideUC07() itself rejects cleanly
               (see NO_FILES comment above), so this callback never
               actually fires with new data here. */
          }}
          onClose={() => setSelectedMemberId(null)}
        />
      )}

      {deleteOpen && population && (
        <DeletePopulationDialog
          populationId={population.id}
          populationName={population.name}
          memberCount={population.member_count}
          onCancel={() => setDeleteOpen(false)}
          onDeleted={onDeleted}
        />
      )}
    </div>
  );
}
