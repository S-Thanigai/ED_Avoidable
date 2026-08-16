import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Uc07ResultsTable } from "../components/Uc07ResultsTable";
import { DEFAULT_SORT } from "../tableState";
import { makeDecision } from "./fixtures";

describe("Uc07ResultsTable", () => {
  it("renders each decision's own tier/probability/navigation, not a recomputed value", () => {
    const decisions = [
      makeDecision({ member_id: "M1", risk: { tier: "HIGH", probability: 0.271 }, navigation: { destination: "CARE_MANAGEMENT", reason_codes: [], explanation: "x" } }),
    ];
    render(
      <Uc07ResultsTable decisions={decisions} onSelect={() => {}} selectedMemberId={null} sort={DEFAULT_SORT} onSortChange={() => {}} />,
    );
    expect(screen.getByText("HIGH")).toBeInTheDocument();
    expect(screen.getByText("27.1%")).toBeInTheDocument();
    expect(screen.getByText("Care Management")).toBeInTheDocument();
  });

  it("renders a Safety badge reflecting each decision's own safety state", () => {
    const decisions = [
      makeDecision({ member_id: "M1", safety: { state: "CLEAR", context_completeness: "COMPLETE", context_source: "CALLER_SUPPLIED" } }),
      makeDecision({ member_id: "M2", safety: { state: "OVERRIDE", override: true }, navigation: { destination: null, reason_codes: [], explanation: "x" } }),
    ];
    render(
      <Uc07ResultsTable decisions={decisions} onSelect={() => {}} selectedMemberId={null} sort={DEFAULT_SORT} onSortChange={() => {}} />,
    );
    expect(screen.getByText("Clear")).toBeInTheDocument();
    expect(screen.getByText("Override")).toBeInTheDocument();
  });

  it("shows 'Suppressed by override' for a null destination rather than a fabricated navigation label", () => {
    const decisions = [
      makeDecision({ member_id: "M1", navigation: { destination: null, reason_codes: [], explanation: "x" }, safety: { state: "OVERRIDE", override: true } }),
    ];
    render(
      <Uc07ResultsTable decisions={decisions} onSelect={() => {}} selectedMemberId={null} sort={DEFAULT_SORT} onSortChange={() => {}} />,
    );
    expect(screen.getByText("Suppressed by override")).toBeInTheDocument();
  });

  it("calls onSelect with the clicked decision from both the member link and the details button", async () => {
    const onSelect = vi.fn();
    const user = userEvent.setup();
    const decision = makeDecision({ member_id: "M042" });
    render(
      <Uc07ResultsTable decisions={[decision]} onSelect={onSelect} selectedMemberId={null} sort={DEFAULT_SORT} onSortChange={onSelect} />,
    );
    await user.click(screen.getByText("M042"));
    expect(onSelect).toHaveBeenCalledWith(decision);

    onSelect.mockClear();
    await user.click(screen.getByRole("button", { name: /view details for member m042/i }));
    expect(onSelect).toHaveBeenCalledWith(decision);
  });

  it("toggles sort direction on repeated header clicks, ascending first then descending", async () => {
    const onSortChange = vi.fn();
    const user = userEvent.setup();
    const decisions = [makeDecision({ member_id: "M1" })];
    const { rerender } = render(
      <Uc07ResultsTable decisions={decisions} onSelect={() => {}} selectedMemberId={null} sort={DEFAULT_SORT} onSortChange={onSortChange} />,
    );
    await user.click(screen.getByRole("button", { name: /risk tier/i }));
    expect(onSortChange).toHaveBeenCalledWith({ key: "tier", direction: "asc" });

    rerender(
      <Uc07ResultsTable
        decisions={decisions}
        onSelect={() => {}}
        selectedMemberId={null}
        sort={{ key: "tier", direction: "asc" }}
        onSortChange={onSortChange}
      />,
    );
    await user.click(screen.getByRole("button", { name: /risk tier/i }));
    expect(onSortChange).toHaveBeenCalledWith({ key: "tier", direction: "desc" });
  });

  it("marks the selected row via aria-selected", () => {
    const decisions = [makeDecision({ member_id: "M1" }), makeDecision({ member_id: "M2" })];
    render(
      <Uc07ResultsTable decisions={decisions} onSelect={() => {}} selectedMemberId="M2" sort={DEFAULT_SORT} onSortChange={() => {}} />,
    );
    const rows = screen.getAllByRole("row").slice(1); // skip header row
    expect(rows[0]).toHaveAttribute("aria-selected", "false");
    expect(rows[1]).toHaveAttribute("aria-selected", "true");
  });
});
