import { describe, expect, it } from "vitest";
import {
  DEFAULT_FILTERS,
  clearFilterField,
  describeActiveFilters,
  filterDecisions,
  isFiltersActive,
  sortDecisions,
  type MemberFiltersState,
} from "../tableState";
import { makeDecision } from "./fixtures";

const A = makeDecision({ member_id: "A", risk: { tier: "LOW", probability: 0.05 }, navigation: { destination: "NO_PROACTIVE_NAVIGATION" }, safety: { state: "CLEAR", context_completeness: "COMPLETE", context_source: "CALLER_SUPPLIED" } });
const B = makeDecision({ member_id: "B", risk: { tier: "MODERATE", probability: 0.15 }, navigation: { destination: "PRIMARY_CARE" }, safety: { state: "CAUTION" } });
const C = makeDecision({ member_id: "C", risk: { tier: "HIGH", probability: 0.35 }, navigation: { destination: "CARE_MANAGEMENT" }, safety: { state: "OVERRIDE", override: true } });
const ALL = [A, B, C];

describe("filterDecisions", () => {
  it("returns everything when no filters are active", () => {
    expect(filterDecisions(ALL, DEFAULT_FILTERS)).toHaveLength(3);
  });

  it("filters by partial member ID search, case-insensitively", () => {
    const result = filterDecisions(ALL, { ...DEFAULT_FILTERS, search: "a" });
    expect(result.map((d) => d.member_id)).toEqual(["A"]);
  });

  it("filters by risk tier", () => {
    expect(filterDecisions(ALL, { ...DEFAULT_FILTERS, tier: "HIGH" }).map((d) => d.member_id)).toEqual(["C"]);
  });

  it("filters by navigation destination", () => {
    expect(filterDecisions(ALL, { ...DEFAULT_FILTERS, navigation: "PRIMARY_CARE" }).map((d) => d.member_id)).toEqual(["B"]);
  });

  it("filters by safety state", () => {
    expect(filterDecisions(ALL, { ...DEFAULT_FILTERS, safety: "OVERRIDE" }).map((d) => d.member_id)).toEqual(["C"]);
  });

  it("filters by probability range", () => {
    const result = filterDecisions(ALL, { ...DEFAULT_FILTERS, probMin: "10", probMax: "30" });
    expect(result.map((d) => d.member_id)).toEqual(["B"]);
  });

  it("combines multiple filters with AND semantics", () => {
    const filters: MemberFiltersState = { ...DEFAULT_FILTERS, tier: "HIGH", navigation: "CARE_MANAGEMENT", probMin: "20" };
    expect(filterDecisions(ALL, filters).map((d) => d.member_id)).toEqual(["C"]);

    const noMatch: MemberFiltersState = { ...DEFAULT_FILTERS, tier: "HIGH", navigation: "PRIMARY_CARE" };
    expect(filterDecisions(ALL, noMatch)).toHaveLength(0);
  });
});

describe("sortDecisions", () => {
  it("leaves order unchanged when no sort key is set", () => {
    expect(sortDecisions(ALL, { key: null, direction: "asc" }).map((d) => d.member_id)).toEqual(["A", "B", "C"]);
  });

  it("sorts by probability ascending and descending", () => {
    expect(sortDecisions(ALL, { key: "probability", direction: "asc" }).map((d) => d.member_id)).toEqual(["A", "B", "C"]);
    expect(sortDecisions(ALL, { key: "probability", direction: "desc" }).map((d) => d.member_id)).toEqual(["C", "B", "A"]);
  });

  it("sorts by risk tier using LOW < MODERATE < HIGH ordinal order, not alphabetical", () => {
    // alphabetically this would be HIGH, LOW, MODERATE -- must not be that
    expect(sortDecisions(ALL, { key: "tier", direction: "asc" }).map((d) => d.member_id)).toEqual(["A", "B", "C"]);
    expect(sortDecisions(ALL, { key: "tier", direction: "desc" }).map((d) => d.member_id)).toEqual(["C", "B", "A"]);
  });

  it("sorts by member_id", () => {
    expect(sortDecisions([C, A, B], { key: "member_id", direction: "asc" }).map((d) => d.member_id)).toEqual(["A", "B", "C"]);
  });

  it("does not mutate the input array", () => {
    const copy = [...ALL];
    sortDecisions(ALL, { key: "probability", direction: "desc" });
    expect(ALL).toEqual(copy);
  });
});

describe("isFiltersActive / describeActiveFilters / clearFilterField", () => {
  it("is false for default filters", () => {
    expect(isFiltersActive(DEFAULT_FILTERS)).toBe(false);
    expect(describeActiveFilters(DEFAULT_FILTERS)).toHaveLength(0);
  });

  it("is true once any field is set, and produces a removable chip", () => {
    const filters: MemberFiltersState = { ...DEFAULT_FILTERS, tier: "HIGH" };
    expect(isFiltersActive(filters)).toBe(true);
    const chips = describeActiveFilters(filters);
    expect(chips).toEqual([{ id: "tier", label: "HIGH" }]);
  });

  it("clearFilterField removes only the targeted field", () => {
    const filters: MemberFiltersState = { ...DEFAULT_FILTERS, tier: "HIGH", navigation: "TELEHEALTH" };
    const next = clearFilterField(filters, "tier");
    expect(next.tier).toBe("ALL");
    expect(next.navigation).toBe("TELEHEALTH");
  });
});
