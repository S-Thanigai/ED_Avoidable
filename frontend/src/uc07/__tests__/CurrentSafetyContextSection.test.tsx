import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { CurrentSafetyContextSection } from "../components/CurrentSafetyContextSection";
import { makeDecision } from "./fixtures";
import type { UploadFiles } from "../../types";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, decideUC07: vi.fn() };
});

const FILES: UploadFiles = {
  members: new File(["x"], "m.csv"),
  edVisits: new File(["x"], "e.csv"),
  care: new File(["x"], "c.csv"),
};

describe("CurrentSafetyContextSection", () => {
  it("renders the decision's current CLEAR status/completeness/source", () => {
    const decision = makeDecision({
      safety: { state: "CLEAR", context_completeness: "COMPLETE", context_source: "CALLER_SUPPLIED" },
    });
    render(<CurrentSafetyContextSection decision={decision} files={FILES} indexDate="2026-07-03" onEvaluated={() => {}} />);
    expect(screen.getByText("Clear")).toBeInTheDocument();
    expect(screen.getByText("Complete")).toBeInTheDocument();
    expect(screen.getByText("User/caller supplied")).toBeInTheDocument();
  });

  it("renders CAUTION status without implying safety", () => {
    const decision = makeDecision({ safety: { state: "CAUTION", context_completeness: "ABSENT", context_source: "NOT_AVAILABLE" } });
    render(<CurrentSafetyContextSection decision={decision} files={FILES} indexDate="2026-07-03" onEvaluated={() => {}} />);
    expect(screen.getByText("Caution")).toBeInTheDocument();
    expect(screen.queryByText(/no emergency detected/i)).not.toBeInTheDocument();
  });

  it("renders OVERRIDE status prominently", () => {
    const decision = makeDecision({ safety: { state: "OVERRIDE", override: true, context_completeness: "PARTIAL", context_source: "CALLER_SUPPLIED" } });
    render(<CurrentSafetyContextSection decision={decision} files={FILES} indexDate="2026-07-03" onEvaluated={() => {}} />);
    expect(screen.getByText("Override")).toBeInTheDocument();
  });

  it("calls decideUC07 for this member only and reports the updated decision, echoing submitted fields", async () => {
    const { decideUC07 } = await import("../api");
    const updated = makeDecision({
      member_id: "M00001",
      safety: { state: "CLEAR", context_completeness: "COMPLETE", context_source: "CALLER_SUPPLIED" },
    });
    vi.mocked(decideUC07).mockResolvedValue({
      model_version: "uc07-risk-synthetic-v1", dataset_id: "synthetic_uc07_v1", synthetic_model: true,
      index_date: "2026-07-03", count: 1, decisions: [updated],
    });

    const onEvaluated = vi.fn();
    const decision = makeDecision({ member_id: "M00001", safety: { state: "CAUTION" } });
    const user = userEvent.setup();
    render(<CurrentSafetyContextSection decision={decision} files={FILES} indexDate="2026-07-03" onEvaluated={onEvaluated} />);

    await user.selectOptions(screen.getByLabelText(/red-flag symptom present/i), "0");
    await user.selectOptions(screen.getByLabelText(/^icu$/i), "0");
    await user.selectOptions(screen.getByLabelText(/admitted/i), "0");
    await user.selectOptions(screen.getByLabelText(/major procedure/i), "0");
    await user.selectOptions(screen.getByLabelText(/triage level/i), "4");
    await user.click(screen.getByRole("button", { name: /evaluate current safety/i }));

    expect(decideUC07).toHaveBeenCalledWith(
      expect.objectContaining({
        memberId: "M00001",
        indexDate: "2026-07-03",
        currentSafetyContext: { M00001: { red_flag: 0, icu: 0, admitted: 0, major_procedure: 0, triage_level: 4 } },
      }),
    );
    expect(onEvaluated).toHaveBeenCalledWith(updated);
    expect(await screen.findByText("Red flag")).toBeInTheDocument();
    expect(screen.getAllByText("No").length).toBeGreaterThan(0);
  });

  it("surfaces a clean error message when the backend evaluation call fails", async () => {
    const { decideUC07, UC07ApiError } = await import("../api");
    vi.mocked(decideUC07).mockRejectedValue(new UC07ApiError("members_file failed validation", 422));

    const user = userEvent.setup();
    const decision = makeDecision({ member_id: "M00001" });
    render(<CurrentSafetyContextSection decision={decision} files={FILES} indexDate="2026-07-03" onEvaluated={() => {}} />);
    await user.click(screen.getByRole("button", { name: /evaluate current safety/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("members_file failed validation");
  });
});
