// Single source of truth for filtering/sorting the already-decided
// FinalUC07Decision[] batch. Purely a presentation-layer concern -- it
// never recomputes a tier, destination, or safety state; it only
// selects/orders decisions the backend already returned.
import type { FinalUC07Decision, NavigationDestination, RiskTier, SafetyState } from "./types";

export interface MemberFiltersState {
  search: string;
  tier: "ALL" | RiskTier;
  navigation: "ALL" | NavigationDestination;
  safety: "ALL" | SafetyState;
  probMin: string;
  probMax: string;
}

export const DEFAULT_FILTERS: MemberFiltersState = {
  search: "",
  tier: "ALL",
  navigation: "ALL",
  safety: "ALL",
  probMin: "",
  probMax: "",
};

export type SortKey = "member_id" | "tier" | "probability" | "navigation";
export type SortDirection = "asc" | "desc";

export interface SortState {
  key: SortKey | null;
  direction: SortDirection;
}

export const DEFAULT_SORT: SortState = { key: null, direction: "asc" };

const TIER_RANK: Record<RiskTier, number> = { LOW: 0, MODERATE: 1, HIGH: 2 };

export function isFiltersActive(filters: MemberFiltersState): boolean {
  return (
    filters.search.trim() !== "" ||
    filters.tier !== "ALL" ||
    filters.navigation !== "ALL" ||
    filters.safety !== "ALL" ||
    filters.probMin !== "" ||
    filters.probMax !== ""
  );
}

export function filterDecisions(decisions: FinalUC07Decision[], filters: MemberFiltersState): FinalUC07Decision[] {
  const search = filters.search.trim().toLowerCase();
  const min = filters.probMin === "" ? null : Number(filters.probMin) / 100;
  const max = filters.probMax === "" ? null : Number(filters.probMax) / 100;

  return decisions.filter((d) => {
    if (search && !d.member_id.toLowerCase().includes(search)) return false;
    if (filters.tier !== "ALL" && d.risk.tier !== filters.tier) return false;
    if (filters.navigation !== "ALL") {
      const dest = d.navigation.destination ?? "NONE";
      if (dest !== filters.navigation) return false;
    }
    if (filters.safety !== "ALL" && d.safety.state !== filters.safety) return false;
    if (min !== null && !Number.isNaN(min) && d.risk.probability < min) return false;
    if (max !== null && !Number.isNaN(max) && d.risk.probability > max) return false;
    return true;
  });
}

export function sortDecisions(decisions: FinalUC07Decision[], sort: SortState): FinalUC07Decision[] {
  if (!sort.key) return decisions;
  const sorted = [...decisions].sort((a, b) => {
    let cmp = 0;
    switch (sort.key) {
      case "member_id":
        cmp = a.member_id.localeCompare(b.member_id);
        break;
      case "tier":
        cmp = TIER_RANK[a.risk.tier] - TIER_RANK[b.risk.tier];
        break;
      case "probability":
        cmp = a.risk.probability - b.risk.probability;
        break;
      case "navigation":
        cmp = (a.navigation.destination ?? "").localeCompare(b.navigation.destination ?? "");
        break;
    }
    return sort.direction === "asc" ? cmp : -cmp;
  });
  return sorted;
}

export interface ActiveFilterChip {
  id: string;
  label: string;
}

const NAV_CHIP_LABEL: Record<string, string> = {
  PRIMARY_CARE: "Primary Care",
  URGENT_CARE: "Urgent Care",
  TELEHEALTH: "Telehealth",
  CARE_MANAGEMENT: "Care Management",
  NO_PROACTIVE_NAVIGATION: "No proactive navigation",
};

export function describeActiveFilters(filters: MemberFiltersState): ActiveFilterChip[] {
  const chips: ActiveFilterChip[] = [];
  if (filters.search.trim()) chips.push({ id: "search", label: `Member: "${filters.search.trim()}"` });
  if (filters.tier !== "ALL") chips.push({ id: "tier", label: filters.tier });
  if (filters.navigation !== "ALL") chips.push({ id: "navigation", label: NAV_CHIP_LABEL[filters.navigation] ?? filters.navigation });
  if (filters.safety !== "ALL") chips.push({ id: "safety", label: filters.safety });
  if (filters.probMin !== "") chips.push({ id: "probMin", label: `Probability ≥ ${filters.probMin}%` });
  if (filters.probMax !== "") chips.push({ id: "probMax", label: `Probability ≤ ${filters.probMax}%` });
  return chips;
}

export function clearFilterField(filters: MemberFiltersState, id: string): MemberFiltersState {
  switch (id) {
    case "search":
      return { ...filters, search: "" };
    case "tier":
      return { ...filters, tier: "ALL" };
    case "navigation":
      return { ...filters, navigation: "ALL" };
    case "safety":
      return { ...filters, safety: "ALL" };
    case "probMin":
      return { ...filters, probMin: "" };
    case "probMax":
      return { ...filters, probMax: "" };
    default:
      return filters;
  }
}
