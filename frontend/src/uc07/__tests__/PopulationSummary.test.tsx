import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { PopulationSummary } from "../components/PopulationSummary";
import { makeDecision } from "./fixtures";

describe("PopulationSummary", () => {
  it("tallies tier/navigation/safety counts from the decisions array, not a hardcoded/stale value", () => {
    const decisions = [
      makeDecision({ member_id: "M1", risk: { tier: "LOW" }, navigation: { destination: "PRIMARY_CARE", reason_codes: [], explanation: "x" }, safety: { state: "CLEAR", context_completeness: "COMPLETE", context_source: "CALLER_SUPPLIED" } }),
      makeDecision({ member_id: "M2", risk: { tier: "HIGH" }, navigation: { destination: "CARE_MANAGEMENT", reason_codes: [], explanation: "x" }, safety: { state: "OVERRIDE", override: true } }),
      makeDecision({ member_id: "M3", risk: { tier: "HIGH" }, navigation: { destination: "CARE_MANAGEMENT", reason_codes: [], explanation: "x" }, safety: { state: "CAUTION" } }),
    ];
    render(<PopulationSummary decisions={decisions} totalCount={3} filtersActive={false} />);

    const riskCard = screen.getByText("Risk distribution").closest(".population-summary__card")!;
    expect(Array.from(riskCard.querySelectorAll("dd")).map((el) => el.textContent)).toEqual(["1", "0", "2"]); // Low/Moderate/High

    const navCard = screen.getByText("Navigation").closest(".population-summary__card")!;
    // NO_PROACTIVE_NAVIGATION, Primary Care, Urgent Care, Telehealth, Care Management
    expect(Array.from(navCard.querySelectorAll("dd")).map((el) => el.textContent)).toEqual(["0", "1", "0", "0", "2"]);

    const safetyCard = screen.getByText("Safety").closest(".population-summary__card")!;
    expect(Array.from(safetyCard.querySelectorAll("dd")).map((el) => el.textContent)).toEqual(["1", "1", "1"]); // Clear/Caution/Override
  });

  it("shows total-vs-filtered wording only when filters are active", () => {
    const decisions = [makeDecision({ member_id: "M1" })];
    const { rerender } = render(<PopulationSummary decisions={decisions} totalCount={5} filtersActive={false} />);
    expect(screen.getByText("5")).toBeInTheDocument();
    expect(screen.queryByText(/filters active/i)).not.toBeInTheDocument();

    rerender(<PopulationSummary decisions={decisions} totalCount={5} filtersActive={true} />);
    const caveat = screen.getByText(/filters active/i).closest(".population-summary__caveat") as HTMLElement;
    expect(within(caveat).getByText("1")).toBeInTheDocument(); // filtered count
    expect(within(caveat).getByText("5")).toBeInTheDocument(); // total count
  });

  it("never claims a clinical outcome or diagnosis", () => {
    render(<PopulationSummary decisions={[]} totalCount={0} filtersActive={false} />);
    expect(screen.getByText(/not a clinical outcome or/i)).toBeInTheDocument();
  });
});
