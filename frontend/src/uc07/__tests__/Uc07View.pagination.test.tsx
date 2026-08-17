import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Uc07View } from "../Uc07View";
import { makeDecision } from "./fixtures";
import type { UC07DecideResponse } from "../types";

// Phase 8D Part 8/15 (tests 17-18) -- pagination desync regression tests.
// Bug: evaluating a member's Current Safety Context can change their
// safety.state such that they no longer match an ACTIVE filter, shrinking
// the filtered/sorted set while `page` stays wherever the user was. Before
// the fix, this could leave the user on a page beyond the new valid range
// -- the table silently rendered zero rows (sorted.length was still > 0,
// so the "no members match" empty state never fired either), while
// Pagination's own internal display-only clamp made the controls LOOK
// fine. The fix (Uc07View.tsx's `clampedPage`) must make the table
// re-render a valid, non-blank page automatically.

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, decideUC07: vi.fn() };
});

// 16 members with PAGE_SIZE=15 reproduces the same "last page holds
// exactly 1 item" shape the previous 26-member/PAGE_SIZE=25 fixture had.
function batchOf16CautionMembers(): UC07DecideResponse {
  const decisions = Array.from({ length: 16 }, (_, i) =>
    makeDecision({
      member_id: `M${String(i + 1).padStart(5, "0")}`,
      safety: { state: "CAUTION", context_completeness: "ABSENT", context_source: "NOT_AVAILABLE" },
    }),
  );
  return {
    model_version: "uc07-risk-synthetic-v1", dataset_id: "synthetic_uc07_v1", synthetic_model: true,
    index_date: "2026-07-03", count: 16, decisions,
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

describe("Uc07View -- pagination clamp after safety re-evaluation", () => {
  it("clamps to a valid page and never renders a blank table when the filtered set shrinks below the current page", async () => {
    const { decideUC07 } = await import("../api");
    vi.mocked(decideUC07).mockResolvedValueOnce(batchOf16CautionMembers());

    const user = userEvent.setup();
    const { container } = render(<Uc07View />);
    await uploadThreeFiles(container);
    await user.click(screen.getByRole("button", { name: /run risk analysis/i }));
    await screen.findByText("Showing 1–15 of 16 members");

    // Filter to safety=CAUTION (matches all 16) so the filtered set is
    // exactly what pagination operates on.
    const filtersSection = screen.getByRole("region", { name: "Member filters" });
    await user.selectOptions(within(filtersSection).getByDisplayValue("Safety: All"), "CAUTION");
    await screen.findByText("Showing 1–15 of 16 members");

    // Go to page 2 -- shows the 16th (last) member only.
    await user.click(screen.getByRole("button", { name: "2" }));
    await screen.findByText("Showing 16–16 of 16 members");
    expect(screen.getByText("M00016")).toBeInTheDocument();

    // Open M00016 and evaluate it to CLEAR -- it now drops out of the
    // active CAUTION filter, shrinking the filtered set from 16 to 15.
    // Page 2 (which needed a 16th item) is no longer valid.
    await user.click(screen.getByText("M00016"));
    const dialog = await screen.findByRole("dialog", { name: /details for member m00016/i });
    await user.click(within(dialog).getByRole("tab", { name: "Current Safety" }));

    const updated = makeDecision({
      member_id: "M00016",
      safety: { state: "CLEAR", context_completeness: "COMPLETE", context_source: "CALLER_SUPPLIED" },
    });
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
    const currentSafetySection = within(dialog).getByRole("region", { name: "Current safety context" });
    await waitFor(() => {
      expect(currentSafetySection.querySelector(".current-safety-context__state")).toHaveTextContent("Clear");
    });

    await user.click(screen.getByLabelText("Close member details"));

    // The table must show the clamped, valid page 1 (15 CAUTION members
    // remain) -- never a blank table, and never the "no members match"
    // empty state (15 members DO still match).
    expect(await screen.findByText("Showing 1–15 of 15 members")).toBeInTheDocument();
    expect(screen.queryByText("No members match the selected filters.")).not.toBeInTheDocument();
    // a real member row is rendered, not a blank table
    expect(screen.getByText("M00001")).toBeInTheDocument();
  });

  it("shows the proper empty state (not a blank table) when a filter matches zero members", async () => {
    const { decideUC07 } = await import("../api");
    vi.mocked(decideUC07).mockResolvedValueOnce(batchOf16CautionMembers());

    const user = userEvent.setup();
    const { container } = render(<Uc07View />);
    await uploadThreeFiles(container);
    await user.click(screen.getByRole("button", { name: /run risk analysis/i }));
    await screen.findByText("Showing 1–15 of 16 members");

    const filtersSection = screen.getByRole("region", { name: "Member filters" });
    await user.selectOptions(within(filtersSection).getByDisplayValue("Safety: All"), "OVERRIDE");

    expect(await screen.findByText("No members match the selected filters.")).toBeInTheDocument();
    expect(screen.queryByText(/Showing \d/)).not.toBeInTheDocument();
  });
});
