import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { WhyFlaggedSection } from "../components/WhyFlaggedSection";
import { makeDecision } from "./fixtures";

describe("WhyFlaggedSection", () => {
  it("renders each explanation factor's display name", () => {
    const decision = makeDecision({
      risk: {
        explanation_factors: [
          { feature: "recent_ed_count", display_name: "Recent overall ED utilization", direction: "INCREASES_RISK", contribution: 0.2, explanation_method: "SHAP_LINEAR" },
          { feature: "telehealth_available", display_name: "Telehealth availability", direction: "DECREASES_RISK", contribution: -0.1, explanation_method: "SHAP_LINEAR" },
        ],
      },
    });
    render(<WhyFlaggedSection risk={decision.risk} />);
    expect(screen.getByText("Recent overall ED utilization")).toBeInTheDocument();
    expect(screen.getByText("Telehealth availability")).toBeInTheDocument();
  });

  it("groups factors into clearly labeled Increased/Decreased sections with a diverging bar per factor", () => {
    const decision = makeDecision({
      risk: {
        explanation_factors: [
          { feature: "a", display_name: "Factor A", direction: "INCREASES_RISK", contribution: 0.2, explanation_method: "SHAP_LINEAR" },
          { feature: "b", display_name: "Factor B", direction: "DECREASES_RISK", contribution: -0.1, explanation_method: "SHAP_LINEAR" },
        ],
      },
    });
    render(<WhyFlaggedSection risk={decision.risk} />);
    expect(screen.getByText("Increased estimate")).toBeInTheDocument();
    expect(screen.getByText("Decreased estimate")).toBeInTheDocument();
    expect(document.querySelectorAll(".why-flagged__row-bar--up")).toHaveLength(1);
    expect(document.querySelectorAll(".why-flagged__row-bar--down")).toHaveLength(1);
    expect(screen.getByText("+0.200")).toBeInTheDocument();
    expect(screen.getByText("−0.100")).toBeInTheDocument();
  });

  it("shows the explanation method, correctly reflecting SHAP vs linear contribution", () => {
    const shapDecision = makeDecision({ risk: { explanation_method: "SHAP_LINEAR" } });
    const { rerender, container } = render(<WhyFlaggedSection risk={shapDecision.risk} />);
    expect(container.querySelector(".why-flagged__method")?.textContent).toMatch(/Explanation method: SHAP/);

    const linearDecision = makeDecision({ risk: { explanation_method: "LINEAR_CONTRIBUTION" } });
    rerender(<WhyFlaggedSection risk={linearDecision.risk} />);
    expect(container.querySelector(".why-flagged__method")?.textContent).toMatch(/Explanation method: Logistic regression/);
  });

  it("never uses causal language", () => {
    const decision = makeDecision({});
    render(<WhyFlaggedSection risk={decision.risk} />);
    const text = document.body.textContent ?? "";
    expect(text.toLowerCase()).not.toMatch(/\bcause[sd]?\b/);
    expect(text.toLowerCase()).not.toContain("diagnos");
  });

  it("shows a caveat that attribution values are not causal (Phase 8D Part 10)", () => {
    const decision = makeDecision({});
    render(<WhyFlaggedSection risk={decision.risk} />);
    expect(
      screen.getByText(/attribution signals and may reflect correlated features/i),
    ).toBeInTheDocument();
  });

  it("renders nothing when there are no explanation factors", () => {
    const decision = makeDecision({ risk: { explanation_factors: [] } });
    const { container } = render(<WhyFlaggedSection risk={decision.risk} />);
    expect(container).toBeEmptyDOMElement();
  });
});
