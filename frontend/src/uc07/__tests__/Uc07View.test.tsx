import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Uc07View } from "../Uc07View";
import { makeDecision } from "./fixtures";
import type { UC07DecideResponse } from "../types";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, decideUC07: vi.fn() };
});

// 30 decisions: enough to exercise 15-per-page pagination, with a
// distinct predictable tier/probability pattern for filter/sort assertions.
function buildResponse(): UC07DecideResponse {
  const decisions = Array.from({ length: 30 }, (_, i) => {
    const id = `M${String(i + 1).padStart(5, "0")}`;
    const tier = i < 10 ? "LOW" : i < 20 ? "MODERATE" : "HIGH";
    return makeDecision({
      member_id: id,
      risk: { tier, probability: (i + 1) / 100 },
      navigation: { destination: i % 2 === 0 ? "PRIMARY_CARE" : "NO_PROACTIVE_NAVIGATION" },
    });
  });
  return {
    model_version: "uc07-risk-synthetic-v1",
    dataset_id: "synthetic_uc07_v1",
    synthetic_model: true,
    index_date: "2026-07-03",
    count: decisions.length,
    decisions,
  };
}

async function uploadThreeFiles(container: HTMLElement) {
  const user = userEvent.setup();
  // Scoped to the required-files UploadPanel only -- the optional
  // current-safety-context CSV input lives outside it.
  const inputs = container.querySelectorAll('.upload-panel input[type="file"]');
  expect(inputs.length).toBe(3);
  const csv = new File(["member_id\nM1\n"], "data.csv", { type: "text/csv" });
  for (const input of Array.from(inputs)) {
    await user.upload(input as HTMLInputElement, csv);
  }
}

describe("Uc07View -- filters + sort + pagination + details state management", () => {
  it("paginates exactly 15 per page, filters, sorts, and preserves state when a details drawer opens/closes", async () => {
    const { decideUC07 } = await import("../api");
    vi.mocked(decideUC07).mockResolvedValue(buildResponse());

    const user = userEvent.setup();
    const { container } = render(<Uc07View />);

    await uploadThreeFiles(container);
    await user.click(screen.getByRole("button", { name: /run risk analysis/i }));

    // population summary + table appear -- exactly 15 rows on page 1
    expect(await screen.findByText("Showing 1–15 of 30 members")).toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(16); // header + 15 rows

    // page 2 starts at the 16th member (M00016), not the 26th
    await user.click(screen.getByRole("button", { name: "2" }));
    expect(await screen.findByText("Showing 16–30 of 30 members")).toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(16); // header + remaining 15 rows
    expect(within(screen.getAllByRole("row")[1]).getByText("M00016")).toBeInTheDocument();

    // filter to HIGH risk only (10 members) -> resets to page 1 automatically
    const riskSelect = screen.getByDisplayValue("Risk: All");
    await user.selectOptions(riskSelect, "HIGH");
    expect(await screen.findByText("Showing 1–10 of 10 members")).toBeInTheDocument();

    // sort by probability descending (exact name -- the Probability
    // Distribution chart's histogram-bin buttons also start with
    // "Probability", so an unanchored match would be ambiguous)
    await user.click(screen.getByRole("button", { name: "Probability" }));
    await user.click(screen.getByRole("button", { name: "Probability" }));
    const firstDataRow = screen.getAllByRole("row")[1];
    expect(within(firstDataRow).getByText("M00030")).toBeInTheDocument();

    // open member details for the top row
    await user.click(within(firstDataRow).getByText("M00030"));
    expect(await screen.findByRole("dialog", { name: /details for member m00030/i })).toBeInTheDocument();

    // close it -- filter/sort/page must be unchanged underneath
    await user.click(screen.getByLabelText("Close member details"));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.getByText("Showing 1–10 of 10 members")).toBeInTheDocument();
    expect(screen.getByDisplayValue("High")).toBeInTheDocument();
  });

  it("shows the empty-filter state and a working clear-filters action", async () => {
    const { decideUC07 } = await import("../api");
    vi.mocked(decideUC07).mockResolvedValue(buildResponse());

    const user = userEvent.setup();
    const { container } = render(<Uc07View />);
    await uploadThreeFiles(container);
    await user.click(screen.getByRole("button", { name: /run risk analysis/i }));
    await screen.findByText("Showing 1–15 of 30 members");

    await user.type(screen.getByPlaceholderText("Search member ID…"), "NOT_A_REAL_ID");
    expect(await screen.findByText("No members match the selected filters.")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(await screen.findByText("Showing 1–15 of 30 members")).toBeInTheDocument();
  });

  it("keeps chart selections and the dropdown filter in sync in both directions (Part 27 test 7)", async () => {
    const { decideUC07 } = await import("../api");
    vi.mocked(decideUC07).mockResolvedValue(buildResponse());

    const user = userEvent.setup();
    const { container } = render(<Uc07View />);
    await uploadThreeFiles(container);
    await user.click(screen.getByRole("button", { name: /run risk analysis/i }));
    await screen.findByText("Showing 1–15 of 30 members");

    // Chart click -> the dropdown reflects it, the table filters, and a
    // chip appears (there is only one filters state -- chart clicks and
    // the dropdown both write to it).
    await user.click(screen.getByRole("button", { name: /^High: 10 members/ }));
    expect(await screen.findByDisplayValue("High")).toBeInTheDocument();
    expect(await screen.findByText("Showing 1–10 of 10 members")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Remove filter HIGH/i })).toBeInTheDocument();

    // Dropdown change -> the chart's legend button reflects the new
    // selection as active (aria-pressed / "filter active" label).
    await user.selectOptions(screen.getByDisplayValue("High"), "LOW");
    expect(await screen.findByRole("button", { name: /^Low: 10 members.*filter active/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^High: 10 members/ })).toHaveAttribute("aria-pressed", "false");
  });

  it("Clear Filters resets chart selections along with the dropdowns (Part 27 test 8)", async () => {
    const { decideUC07 } = await import("../api");
    vi.mocked(decideUC07).mockResolvedValue(buildResponse());

    const user = userEvent.setup();
    const { container } = render(<Uc07View />);
    await uploadThreeFiles(container);
    await user.click(screen.getByRole("button", { name: /run risk analysis/i }));
    await screen.findByText("Showing 1–15 of 30 members");

    await user.click(screen.getByRole("button", { name: /^High: 10 members/ }));
    expect(await screen.findByText("Showing 1–10 of 10 members")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Clear all filters" }));
    expect(await screen.findByText("Showing 1–15 of 30 members")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^High: 10 members/ })).toHaveAttribute("aria-pressed", "false");
  });
});
