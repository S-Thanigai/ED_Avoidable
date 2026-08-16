import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemberDetailsDrawer } from "../components/MemberDetailsDrawer";
import { makeDecision } from "./fixtures";
import type { UploadFiles } from "../../types";
import type { MemberExplanationResponse } from "../types";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, explainMember: vi.fn(), decideUC07: vi.fn() };
});

const FILES: UploadFiles = {
  members: new File(["x"], "m.csv"),
  edVisits: new File(["x"], "e.csv"),
  care: new File(["x"], "c.csv"),
};

const FALLBACK_EXPLANATION: MemberExplanationResponse = {
  summary: "Low modeled risk.",
  risk_explanation: "ok",
  navigation_explanation: "ok",
  safety_explanation: "ok",
  disclaimer: "disclaimer text",
  explanation_source: "DETERMINISTIC_FALLBACK",
  model_used: null,
  generation_time_ms: null,
};

describe("MemberDetailsDrawer", () => {
  it("composes the decision panel, data sections, safety context, and AI explanation for the given member", async () => {
    const { explainMember } = await import("../api");
    vi.mocked(explainMember).mockResolvedValue(FALLBACK_EXPLANATION);

    const decision = makeDecision({ member_id: "M00123" });
    render(
      <MemberDetailsDrawer
        decision={decision}
        lookups={null}
        lookupsLoading={false}
        files={FILES}
        indexDate="2026-07-03"
        onSafetyEvaluated={() => {}}
        onClose={() => {}}
      />,
    );

    expect(screen.getByRole("dialog", { name: /details for member m00123/i })).toBeInTheDocument();
    expect(screen.getByText("M00123")).toBeInTheDocument();
    // AiExplanationSection fetched an explanation scoped to THIS decision
    expect(explainMember).toHaveBeenCalledWith(decision);
    expect(screen.getByText("AI Explanation")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Current safety context" })).toBeInTheDocument();
  });

  it("calls onClose on Escape and on overlay click, but not on a click inside the drawer body", async () => {
    const { explainMember } = await import("../api");
    vi.mocked(explainMember).mockResolvedValue(FALLBACK_EXPLANATION);

    const onClose = vi.fn();
    const user = userEvent.setup();
    const decision = makeDecision({ member_id: "M00001" });
    const { container } = render(
      <MemberDetailsDrawer
        decision={decision}
        lookups={null}
        lookupsLoading={false}
        files={FILES}
        indexDate="2026-07-03"
        onSafetyEvaluated={() => {}}
        onClose={onClose}
      />,
    );

    // click inside the dialog body must NOT close it
    await user.click(screen.getByRole("dialog"));
    expect(onClose).not.toHaveBeenCalled();

    // click on the overlay itself (outside the dialog) DOES close it
    const overlay = container.querySelector(".member-details-drawer__overlay")!;
    await user.click(overlay);
    expect(onClose).toHaveBeenCalledTimes(1);

    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("passes onSafetyEvaluated through to the Current Safety Context section", async () => {
    const { explainMember, decideUC07, UC07ApiError } = await import("../api");
    vi.mocked(explainMember).mockResolvedValue(FALLBACK_EXPLANATION);
    const updated = makeDecision({ member_id: "M00001", safety: { state: "CLEAR", context_completeness: "COMPLETE", context_source: "CALLER_SUPPLIED" } });
    vi.mocked(decideUC07).mockResolvedValue({
      model_version: "v1", dataset_id: "d1", synthetic_model: true, index_date: "2026-07-03", count: 1, decisions: [updated],
    });

    const onSafetyEvaluated = vi.fn();
    const user = userEvent.setup();
    const decision = makeDecision({ member_id: "M00001", safety: { state: "CAUTION" } });
    render(
      <MemberDetailsDrawer
        decision={decision}
        lookups={null}
        lookupsLoading={false}
        files={FILES}
        indexDate="2026-07-03"
        onSafetyEvaluated={onSafetyEvaluated}
        onClose={() => {}}
      />,
    );

    await user.selectOptions(screen.getByLabelText(/red-flag symptom present/i), "0");
    await user.selectOptions(screen.getByLabelText(/^icu$/i), "0");
    await user.selectOptions(screen.getByLabelText(/admitted/i), "0");
    await user.selectOptions(screen.getByLabelText(/major procedure/i), "0");
    await user.selectOptions(screen.getByLabelText(/triage level/i), "4");
    await user.click(screen.getByRole("button", { name: /evaluate current safety/i }));

    expect(onSafetyEvaluated).toHaveBeenCalledWith(updated);
    void UC07ApiError; // referenced to satisfy the import above
  });
});
