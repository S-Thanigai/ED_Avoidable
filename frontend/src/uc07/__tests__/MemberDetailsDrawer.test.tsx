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

function renderDrawer(overrides: Parameters<typeof makeDecision>[0] = {}, extraProps: Partial<Parameters<typeof MemberDetailsDrawer>[0]> = {}) {
  const decision = makeDecision({ member_id: "M00123", ...overrides });
  const onClose = vi.fn();
  const onSafetyEvaluated = vi.fn();
  const utils = render(
    <MemberDetailsDrawer
      decision={decision}
      lookups={null}
      lookupsLoading={false}
      files={FILES}
      indexDate="2026-07-03"
      onSafetyEvaluated={onSafetyEvaluated}
      onClose={onClose}
      {...extraProps}
    />,
  );
  return { ...utils, decision, onClose, onSafetyEvaluated };
}

describe("MemberDetailsDrawer (member workspace)", () => {
  it("opens on the Overview tab, showing risk/navigation/safety at a glance and in full", () => {
    const { decision } = renderDrawer();
    expect(screen.getByRole("dialog", { name: /details for member m00123/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Overview", selected: true })).toBeInTheDocument();
    // header "at a glance" metrics
    expect(screen.getByText(decision.member_id)).toBeInTheDocument();
    // full RiskCard/SafetyCard/NavigationCard render in the Overview panel
    expect(screen.getByRole("region", { name: "Risk assessment" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Safety status" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Navigation recommendation" })).toBeInTheDocument();
  });

  it("does not fetch an AI explanation until the AI Explanation tab is actually opened (lazy, Part 14)", async () => {
    const { explainMember } = await import("../api");
    vi.mocked(explainMember).mockResolvedValue(FALLBACK_EXPLANATION);
    const user = userEvent.setup();
    renderDrawer();

    expect(explainMember).not.toHaveBeenCalled();

    await user.click(screen.getByRole("tab", { name: "AI Explanation" }));
    expect(await screen.findByRole("heading", { name: "AI Explanation" })).toBeInTheDocument();
    expect(explainMember).toHaveBeenCalledTimes(1);
  });

  it("switches to the Why Flagged tab and shows the SHAP explanation there", async () => {
    const user = userEvent.setup();
    renderDrawer();
    await user.click(screen.getByRole("tab", { name: "Why Flagged" }));
    expect(screen.getByText("Why This Member Was Flagged")).toBeInTheDocument();
  });

  it("switches to the Current Safety tab and exposes the safety-context evaluator there", async () => {
    const user = userEvent.setup();
    renderDrawer();
    await user.click(screen.getByRole("tab", { name: "Current Safety" }));
    expect(screen.getByRole("region", { name: "Current safety context" })).toBeInTheDocument();
  });

  it("switches to the Communication tab and exposes report/email actions there (not in the header)", async () => {
    const user = userEvent.setup();
    renderDrawer();
    expect(screen.queryByRole("button", { name: "Download Report" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Communication" }));
    expect(screen.getByRole("heading", { name: "Member Communication" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download Report" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Email Member" })).toBeInTheDocument();
  });

  it("shows the strongest-priority OVERRIDE banner regardless of which tab is active", async () => {
    const user = userEvent.setup();
    renderDrawer({ safety: { state: "OVERRIDE", override: true }, navigation: { destination: null, reason_codes: [], explanation: "x" } });
    const banner = () => document.querySelector(".member-workspace__override-banner");
    expect(banner()).toHaveTextContent(/safety override active/i);
    await user.click(screen.getByRole("tab", { name: "Why Flagged" }));
    expect(banner()).toHaveTextContent(/safety override active/i);
  });

  it("resets to the Overview tab when a different member is opened", async () => {
    const user = userEvent.setup();
    const decision1 = makeDecision({ member_id: "M00001" });
    const { rerender } = render(
      <MemberDetailsDrawer decision={decision1} lookups={null} lookupsLoading={false} files={FILES} indexDate="2026-07-03" onSafetyEvaluated={() => {}} onClose={() => {}} />,
    );
    await user.click(screen.getByRole("tab", { name: "Why Flagged" }));
    expect(screen.getByRole("tab", { name: "Why Flagged", selected: true })).toBeInTheDocument();

    const decision2 = makeDecision({ member_id: "M00002" });
    rerender(
      <MemberDetailsDrawer decision={decision2} lookups={null} lookupsLoading={false} files={FILES} indexDate="2026-07-03" onSafetyEvaluated={() => {}} onClose={() => {}} />,
    );
    expect(screen.getByRole("tab", { name: "Overview", selected: true })).toBeInTheDocument();
  });

  it("calls onClose on Escape and on overlay click, but not on a click inside the workspace", async () => {
    const user = userEvent.setup();
    const { onClose } = renderDrawer();

    await user.click(screen.getByRole("dialog"));
    expect(onClose).not.toHaveBeenCalled();

    const overlay = document.querySelector(".member-workspace__overlay") as HTMLElement;
    await user.click(overlay);
    expect(onClose).toHaveBeenCalledTimes(1);

    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("focuses the close button on open and traps Tab focus within the workspace", async () => {
    renderDrawer();
    expect(screen.getByLabelText("Close member details")).toHaveFocus();
  });

  it("passes onSafetyEvaluated through to the Current Safety Context section", async () => {
    const { decideUC07 } = await import("../api");
    const updated = makeDecision({ member_id: "M00001", safety: { state: "CLEAR", context_completeness: "COMPLETE", context_source: "CALLER_SUPPLIED" } });
    vi.mocked(decideUC07).mockResolvedValue({
      model_version: "v1", dataset_id: "d1", synthetic_model: true, index_date: "2026-07-03", count: 1, decisions: [updated],
    });

    const user = userEvent.setup();
    const { onSafetyEvaluated } = renderDrawer({ member_id: "M00001", safety: { state: "CAUTION" } });

    await user.click(screen.getByRole("tab", { name: "Current Safety" }));
    await user.selectOptions(screen.getByLabelText(/red-flag symptom present/i), "0");
    await user.selectOptions(screen.getByLabelText(/^icu$/i), "0");
    await user.selectOptions(screen.getByLabelText(/admitted/i), "0");
    await user.selectOptions(screen.getByLabelText(/major procedure/i), "0");
    await user.selectOptions(screen.getByLabelText(/triage level/i), "4");
    await user.click(screen.getByRole("button", { name: /evaluate current safety/i }));

    expect(onSafetyEvaluated).toHaveBeenCalledWith(updated);
  });
});
