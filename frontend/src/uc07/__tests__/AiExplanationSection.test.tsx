import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { AiExplanationSection } from "../components/AiExplanationSection";
import { makeDecision } from "./fixtures";
import type { MemberExplanationResponse } from "../types";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, explainMember: vi.fn() };
});

const GENAI_RESPONSE: MemberExplanationResponse = {
  summary: "Low modeled risk; no proactive navigation suggested.",
  risk_explanation: "This member's modeled risk tier is low.",
  navigation_explanation: "No proactive navigation destination applies.",
  safety_explanation: "Current safety information is absent or incomplete.",
  disclaimer: "For care navigation only -- never a reason to delay care.",
  explanation_source: "GENAI",
  model_used: "qwen3:8b",
  generation_time_ms: 8123.4,
};

const FALLBACK_RESPONSE: MemberExplanationResponse = {
  ...GENAI_RESPONSE,
  explanation_source: "DETERMINISTIC_FALLBACK",
  model_used: null,
  generation_time_ms: null,
};

describe("AiExplanationSection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows a loading state while the explanation is being generated", async () => {
    const { explainMember } = await import("../api");
    vi.mocked(explainMember).mockReturnValue(new Promise(() => {})); // never resolves
    const decision = makeDecision({});
    render(<AiExplanationSection decision={decision} />);
    expect(screen.getByText(/generating explanation/i)).toBeInTheDocument();
  });

  it("renders the AI-generated explanation and labels its source as AI-generated", async () => {
    const { explainMember } = await import("../api");
    vi.mocked(explainMember).mockResolvedValue(GENAI_RESPONSE);
    const decision = makeDecision({});
    render(<AiExplanationSection decision={decision} />);

    expect(await screen.findByText(GENAI_RESPONSE.summary)).toBeInTheDocument();
    expect(screen.getByText(/Source: AI-generated explanation using Qwen3 8B/)).toBeInTheDocument();
  });

  it("renders the deterministic fallback and labels its source distinctly, not implying AI generation", async () => {
    const { explainMember } = await import("../api");
    vi.mocked(explainMember).mockResolvedValue(FALLBACK_RESPONSE);
    const decision = makeDecision({});
    render(<AiExplanationSection decision={decision} />);

    expect(await screen.findByText(FALLBACK_RESPONSE.summary)).toBeInTheDocument();
    expect(screen.getByText("Explanation source: Deterministic system explanation.")).toBeInTheDocument();
    expect(screen.queryByText(/AI-generated/)).not.toBeInTheDocument();
  });

  it("shows a clean error and does not crash when the backend call fails (e.g. Ollama unreachable)", async () => {
    const { explainMember, UC07ApiError } = await import("../api");
    vi.mocked(explainMember).mockRejectedValue(new UC07ApiError("Could not reach the UC07 decision service", null));
    const decision = makeDecision({});
    render(<AiExplanationSection decision={decision} />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Could not reach the UC07 decision service");
    // the section heading itself still renders -- one failed lazy call
    // does not take down the rest of the member details view
    expect(screen.getByText("AI Explanation")).toBeInTheDocument();
  });

  it("calls explainMember exactly once per member on mount (lazy, on-demand)", async () => {
    const { explainMember } = await import("../api");
    vi.mocked(explainMember).mockResolvedValue(GENAI_RESPONSE);
    const decision = makeDecision({ member_id: "M00042" });
    render(<AiExplanationSection decision={decision} />);
    await waitFor(() => expect(explainMember).toHaveBeenCalledTimes(1));
    expect(explainMember).toHaveBeenCalledWith(decision);
  });

  it("re-fetches when the member's safety state changes (e.g. after Evaluate Current Safety)", async () => {
    const { explainMember } = await import("../api");
    vi.mocked(explainMember).mockResolvedValue(GENAI_RESPONSE);
    const decision = makeDecision({ member_id: "M00001", safety: { state: "CAUTION" } });
    const { rerender } = render(<AiExplanationSection decision={decision} />);
    await waitFor(() => expect(explainMember).toHaveBeenCalledTimes(1));

    const updated = makeDecision({ member_id: "M00001", safety: { state: "CLEAR", context_completeness: "COMPLETE" } });
    rerender(<AiExplanationSection decision={updated} />);
    await waitFor(() => expect(explainMember).toHaveBeenCalledTimes(2));
  });

  it("never renders a raw JSON blob or unexpected field -- only the known text fields", async () => {
    const { explainMember } = await import("../api");
    vi.mocked(explainMember).mockResolvedValue(GENAI_RESPONSE);
    const decision = makeDecision({});
    render(<AiExplanationSection decision={decision} />);
    await screen.findByText(GENAI_RESPONSE.summary);
    const text = document.body.textContent ?? "";
    expect(text).not.toContain("{");
    expect(text).not.toContain("generation_time_ms");
  });
});
