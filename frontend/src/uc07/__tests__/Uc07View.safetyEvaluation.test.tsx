import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Uc07View } from "../Uc07View";
import { makeDecision } from "./fixtures";
import type { UC07DecideResponse } from "../types";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, decideUC07: vi.fn() };
});

function batchResponse(): UC07DecideResponse {
  return {
    model_version: "uc07-risk-synthetic-v1", dataset_id: "synthetic_uc07_v1", synthetic_model: true,
    index_date: "2026-07-03", count: 3,
    decisions: [
      makeDecision({ member_id: "M00001", safety: { state: "CLEAR", context_completeness: "COMPLETE", context_source: "CALLER_SUPPLIED" } }),
      makeDecision({ member_id: "M00002", safety: { state: "CAUTION" } }),
      makeDecision({
        member_id: "M00003", navigation: { destination: null, reason_codes: [], explanation: "x" },
        safety: { state: "OVERRIDE", override: true },
      }),
    ],
  };
}

async function uploadThreeFiles(container: HTMLElement) {
  const user = userEvent.setup();
  const inputs = container.querySelectorAll('.upload-panel input[type="file"]');
  const csv = new File(["member_id\nM1\n"], "data.csv", { type: "text/csv" });
  for (const input of Array.from(inputs)) {
    await user.upload(input as HTMLInputElement, csv);
  }
}

function safetyCardCounts() {
  // Scoped to the Population Overview's Safety Distribution chart card
  // specifically -- "Safety" also appears as a standalone glance label
  // in the member workspace header when a member's details are open,
  // which would otherwise make this query ambiguous. Order matches
  // SAFETY_META in AnalyticsCharts.tsx: CLEAR, CAUTION, OVERRIDE.
  const summary = screen.getByRole("region", { name: "Population overview and analytics" });
  const card = within(summary).getByText("Safety Distribution").closest(".analytics-card")!;
  return Array.from(card.querySelectorAll(".analytics-legend__count")).map((el) => el.textContent);
}

describe("Uc07View -- batch mixed safety states + Evaluate Current Safety integration", () => {
  it("renders a genuine mixture of CLEAR/CAUTION/OVERRIDE counts, not forced equal", async () => {
    const { decideUC07 } = await import("../api");
    vi.mocked(decideUC07).mockResolvedValue(batchResponse());

    const user = userEvent.setup();
    const { container } = render(<Uc07View />);
    await uploadThreeFiles(container);
    await user.click(screen.getByRole("button", { name: /run risk analysis/i }));

    await screen.findByText("Showing 1–3 of 3 members");
    // 1 CLEAR, 1 CAUTION, 1 OVERRIDE (in that row order: see PopulationSummary's SAFETY_ROWS)
    expect(safetyCardCounts()).toEqual(["1", "1", "1"]);
  });

  it("updates counts and the reopened drawer after Evaluate Current Safety, without disturbing other members", async () => {
    const { decideUC07 } = await import("../api");
    vi.mocked(decideUC07).mockResolvedValueOnce(batchResponse());

    const user = userEvent.setup();
    const { container } = render(<Uc07View />);
    await uploadThreeFiles(container);
    await user.click(screen.getByRole("button", { name: /run risk analysis/i }));
    await screen.findByText("Showing 1–3 of 3 members");
    expect(safetyCardCounts()).toEqual(["1", "1", "1"]); // CLEAR / CAUTION / OVERRIDE

    // Open M00002 (currently CAUTION) and evaluate it to CLEAR.
    await user.click(screen.getByText("M00002"));
    const dialog = await screen.findByRole("dialog", { name: /details for member m00002/i });
    await user.click(within(dialog).getByRole("tab", { name: "Current Safety" }));

    const updated = makeDecision({ member_id: "M00002", safety: { state: "CLEAR", context_completeness: "COMPLETE", context_source: "CALLER_SUPPLIED" } });
    vi.mocked(decideUC07).mockResolvedValueOnce({
      model_version: "uc07-risk-synthetic-v1", dataset_id: "synthetic_uc07_v1", synthetic_model: true,
      index_date: "2026-07-03", count: 1, decisions: [updated],
    });

    await user.selectOptions(within(dialog).getByLabelText(/red-flag symptom present/i), "0");
    await user.selectOptions(within(dialog).getByLabelText(/^icu$/i), "0");
    await user.selectOptions(within(dialog).getByLabelText(/admitted/i), "0");
    await user.selectOptions(within(dialog).getByLabelText(/major procedure/i), "0");
    await user.selectOptions(within(dialog).getByLabelText(/triage level/i), "4");
    await user.click(within(dialog).getByRole("button", { name: /evaluate current safety/i }));

    // decideUC07 called scoped to exactly M00002, using the ORIGINAL batch's own index_date
    expect(decideUC07).toHaveBeenLastCalledWith(expect.objectContaining({ memberId: "M00002", indexDate: "2026-07-03" }));

    // the drawer's Current Safety Context section reflects the new state immediately (still open)
    const currentSafetySection = within(dialog).getByRole("region", { name: "Current safety context" });
    await waitFor(() => {
      expect(currentSafetySection.querySelector(".current-safety-context__state")).toHaveTextContent("Clear");
    });

    await user.click(screen.getByLabelText("Close member details"));

    // 2 CLEAR (M00001 + M00002), 0 CAUTION, 1 OVERRIDE (M00003 untouched)
    expect(safetyCardCounts()).toEqual(["2", "0", "1"]);
  });

  it("closes the details drawer on Escape", async () => {
    const { decideUC07 } = await import("../api");
    vi.mocked(decideUC07).mockResolvedValue(batchResponse());

    const user = userEvent.setup();
    const { container } = render(<Uc07View />);
    await uploadThreeFiles(container);
    await user.click(screen.getByRole("button", { name: /run risk analysis/i }));
    await screen.findByText("Showing 1–3 of 3 members");

    await user.click(screen.getByText("M00001"));
    await screen.findByRole("dialog");
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
