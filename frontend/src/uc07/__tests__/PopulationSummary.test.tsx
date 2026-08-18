import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PopulationSummary } from "../components/PopulationSummary";
import { makeDecision } from "./fixtures";
import { DEFAULT_FILTERS, type MemberFiltersState } from "../tableState";

function renderSummary(decisions: ReturnType<typeof makeDecision>[], filters: MemberFiltersState = DEFAULT_FILTERS) {
  const onFilterChange = vi.fn();
  const utils = render(<PopulationSummary decisions={decisions} filters={filters} onFilterChange={onFilterChange} />);
  return { ...utils, onFilterChange };
}

describe("PopulationSummary", () => {
  it("tallies KPI counts from the decisions array, not a hardcoded/stale value", () => {
    const decisions = [
      makeDecision({ member_id: "M1", risk: { tier: "LOW" }, safety: { state: "CLEAR", context_completeness: "COMPLETE", context_source: "CALLER_SUPPLIED" } }),
      makeDecision({ member_id: "M2", risk: { tier: "HIGH" }, safety: { state: "OVERRIDE", override: true } }),
      makeDecision({ member_id: "M3", risk: { tier: "HIGH" }, safety: { state: "CAUTION" } }),
    ];
    renderSummary(decisions);

    expect(screen.getByText("3", { selector: ".population-summary__kpi-value" })).toBeInTheDocument();
    const kpis = screen.getAllByText(/^\d+$/, { selector: ".population-summary__kpi-value" }).map((el) => el.textContent);
    // Total / High Risk / Moderate Risk / Navigation Opportunities / Safety Caution / Safety Override.
    // None of the three fixtures override `navigation`, so all default to
    // NO_PROACTIVE_NAVIGATION (fixtures.ts) -- Navigation Opportunities is 0.
    expect(kpis).toEqual(["3", "2", "0", "0", "1", "1"]);
  });

  it("never claims a clinical outcome or diagnosis", () => {
    renderSummary([]);
    expect(screen.getByText(/not a clinical outcome or/i)).toBeInTheDocument();
  });

  it("does not crash on a zero-member population and shows an empty state per chart", () => {
    renderSummary([]);
    expect(screen.getAllByText(/No members in the current population/).length).toBeGreaterThan(0);
  });

  it("does not crash when 100% of members share one category (e.g. all CAUTION)", () => {
    const decisions = Array.from({ length: 5 }, (_, i) => makeDecision({ member_id: `M${i}`, safety: { state: "CAUTION" } }));
    renderSummary(decisions);
    expect(screen.getByRole("button", { name: /Caution: 5 members \(100%\)/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Clear: 0 members \(0%\)/ })).toBeInTheDocument();
  });

  it("clicking the HIGH risk donut segment sets the tier filter (Part 27 test 4)", async () => {
    const user = userEvent.setup();
    const decisions = [makeDecision({ member_id: "M1", risk: { tier: "LOW" } }), makeDecision({ member_id: "M2", risk: { tier: "HIGH" } })];
    const { onFilterChange } = renderSummary(decisions);

    await user.click(screen.getByRole("button", { name: /^High: 1 members/ }));
    expect(onFilterChange).toHaveBeenCalledWith({ ...DEFAULT_FILTERS, tier: "HIGH" });
  });

  it("clicking an already-active segment clears that filter (toggle off)", async () => {
    const user = userEvent.setup();
    const decisions = [makeDecision({ member_id: "M1", risk: { tier: "HIGH" } })];
    const { onFilterChange } = renderSummary(decisions, { ...DEFAULT_FILTERS, tier: "HIGH" });

    await user.click(screen.getByRole("button", { name: /^High: 1 members.*filter active/ }));
    expect(onFilterChange).toHaveBeenCalledWith({ ...DEFAULT_FILTERS, tier: "ALL" });
  });

  it("clicking a navigation bar segment sets the navigation filter (Part 27 test 5)", async () => {
    const user = userEvent.setup();
    const decisions = [
      makeDecision({ member_id: "M1", navigation: { destination: "CARE_MANAGEMENT", reason_codes: [], explanation: "x" } }),
    ];
    const { onFilterChange } = renderSummary(decisions);

    await user.click(screen.getByRole("button", { name: /Care Management: 1 members/ }));
    expect(onFilterChange).toHaveBeenCalledWith({ ...DEFAULT_FILTERS, navigation: "CARE_MANAGEMENT" });
  });

  it("clicking a safety donut segment sets the safety filter (Part 27 test 6)", async () => {
    const user = userEvent.setup();
    const decisions = [makeDecision({ member_id: "M1", safety: { state: "OVERRIDE", override: true } })];
    const { onFilterChange } = renderSummary(decisions);

    await user.click(screen.getByRole("button", { name: /^Override: 1 members/ }));
    expect(onFilterChange).toHaveBeenCalledWith({ ...DEFAULT_FILTERS, safety: "OVERRIDE" });
  });

  it("shows an impossible-to-miss OVERRIDE callout only when the override count is non-zero", () => {
    const withOverride = [makeDecision({ member_id: "M1", safety: { state: "OVERRIDE", override: true } })];
    const { rerender } = renderSummary(withOverride);
    expect(screen.getByText(/1 member in OVERRIDE/)).toBeInTheDocument();

    const noOverride = [makeDecision({ member_id: "M1", safety: { state: "CLEAR", context_completeness: "COMPLETE", context_source: "CALLER_SUPPLIED" } })];
    rerender(<PopulationSummary decisions={noOverride} filters={DEFAULT_FILTERS} onFilterChange={vi.fn()} />);
    expect(screen.queryByText(/member.*in OVERRIDE/)).not.toBeInTheDocument();
  });

  it("clicking a probability histogram bin sets probMin/probMax", async () => {
    const user = userEvent.setup();
    const decisions = [makeDecision({ member_id: "M1", risk: { probability: 0.35 } })];
    const { onFilterChange } = renderSummary(decisions);

    await user.click(screen.getByRole("button", { name: /Probability 30–39%: 1 members/ }));
    expect(onFilterChange).toHaveBeenCalledWith({ ...DEFAULT_FILTERS, probMin: "30", probMax: "40" });
  });
});
