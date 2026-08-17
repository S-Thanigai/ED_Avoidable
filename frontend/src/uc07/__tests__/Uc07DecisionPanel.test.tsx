import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Uc07DecisionPanel } from "../components/Uc07DecisionPanel";
import { makeDecision } from "./fixtures";

describe("Uc07DecisionPanel -- risk tier rendering", () => {
  it("renders LOW", () => {
    render(<Uc07DecisionPanel decision={makeDecision({ risk: { tier: "LOW", probability: 0.05 } })} />);
    expect(screen.getByText("Low")).toBeInTheDocument();
    expect(screen.getByText("5.0%")).toBeInTheDocument();
  });

  it("renders MODERATE", () => {
    render(<Uc07DecisionPanel decision={makeDecision({ risk: { tier: "MODERATE", probability: 0.15 } })} />);
    expect(screen.getByText("Moderate")).toBeInTheDocument();
    expect(screen.getByText("15.0%")).toBeInTheDocument();
  });

  it("renders HIGH", () => {
    render(<Uc07DecisionPanel decision={makeDecision({ risk: { tier: "HIGH", probability: 0.42 } })} />);
    expect(screen.getByText("High")).toBeInTheDocument();
    expect(screen.getByText("42.0%")).toBeInTheDocument();
  });
});

describe("Uc07DecisionPanel -- navigation destination rendering", () => {
  it("renders PRIMARY_CARE", () => {
    render(<Uc07DecisionPanel decision={makeDecision({ navigation: { destination: "PRIMARY_CARE" } })} />);
    expect(screen.getByText("Primary Care")).toBeInTheDocument();
  });

  it("renders URGENT_CARE", () => {
    render(<Uc07DecisionPanel decision={makeDecision({ navigation: { destination: "URGENT_CARE" } })} />);
    expect(screen.getByText("Urgent Care")).toBeInTheDocument();
  });

  it("renders TELEHEALTH", () => {
    render(<Uc07DecisionPanel decision={makeDecision({ navigation: { destination: "TELEHEALTH" } })} />);
    expect(screen.getByText("Telehealth")).toBeInTheDocument();
  });

  it("renders CARE_MANAGEMENT", () => {
    render(<Uc07DecisionPanel decision={makeDecision({ navigation: { destination: "CARE_MANAGEMENT" } })} />);
    expect(screen.getByText("Care Management")).toBeInTheDocument();
  });

  it("renders NO_PROACTIVE_NAVIGATION", () => {
    render(<Uc07DecisionPanel decision={makeDecision({ navigation: { destination: "NO_PROACTIVE_NAVIGATION" } })} />);
    expect(screen.getByText("No proactive navigation")).toBeInTheDocument();
  });
});

describe("Uc07DecisionPanel -- safety state rendering", () => {
  it("renders CLEAR", () => {
    render(
      <Uc07DecisionPanel
        decision={makeDecision({ safety: { state: "CLEAR", context_completeness: "COMPLETE", context_source: "CALLER_SUPPLIED" } })}
      />,
    );
    expect(screen.getByText("Clear")).toBeInTheDocument();
  });

  it("renders CAUTION and never implies safety", () => {
    render(<Uc07DecisionPanel decision={makeDecision({ safety: { state: "CAUTION" } })} />);
    expect(screen.getByText("Caution")).toBeInTheDocument();
    expect(screen.queryByText(/no emergency detected/i)).not.toBeInTheDocument();
  });

  it("renders OVERRIDE with navigation suppressed, not 'use this instead'", () => {
    render(
      <Uc07DecisionPanel
        decision={makeDecision({
          navigation: { destination: null, reason_codes: [], explanation: "x" },
          safety: { state: "OVERRIDE", override: true },
        })}
      />,
    );
    expect(screen.getByText("Emergency safety override")).toBeInTheDocument();
    expect(screen.getByText("Navigation logic suppressed by safety override")).toBeInTheDocument();
    expect(screen.queryByText(/use this instead/i)).not.toBeInTheDocument();
  });
});

describe("Uc07DecisionPanel -- context completeness rendering", () => {
  it("renders COMPLETE", () => {
    render(
      <Uc07DecisionPanel
        decision={makeDecision({ safety: { context_completeness: "COMPLETE", context_source: "CALLER_SUPPLIED" } })}
      />,
    );
    expect(screen.getByText("Complete")).toBeInTheDocument();
  });

  it("renders PARTIAL", () => {
    render(
      <Uc07DecisionPanel
        decision={makeDecision({ safety: { context_completeness: "PARTIAL", context_source: "CALLER_SUPPLIED" } })}
      />,
    );
    expect(screen.getByText("Partial")).toBeInTheDocument();
  });

  it("renders ABSENT", () => {
    render(
      <Uc07DecisionPanel
        decision={makeDecision({ safety: { context_completeness: "ABSENT", context_source: "NOT_AVAILABLE" } })}
      />,
    );
    // "Not available" legitimately appears twice here: completeness badge
    // AND source label both read ABSENT/NOT_AVAILABLE -- assert both occurrences exist.
    expect(screen.getAllByText("Not available")).toHaveLength(2);
  });

  it("never labels CALLER_SUPPLIED as verified", () => {
    render(
      <Uc07DecisionPanel
        decision={makeDecision({ safety: { context_completeness: "COMPLETE", context_source: "CALLER_SUPPLIED" } })}
      />,
    );
    expect(screen.getByText("User/caller supplied")).toBeInTheDocument();
    expect(screen.queryByText(/verified/i)).not.toBeInTheDocument();
  });
});

describe("Uc07DecisionPanel -- disclosure", () => {
  it("shows synthetic disclosure and model version", () => {
    render(<Uc07DecisionPanel decision={makeDecision({ risk: { model_version: "uc07-risk-synthetic-v1" } })} />);
    expect(screen.getByText(/trained on synthetic data/i)).toBeInTheDocument();
    // model version legitimately appears twice (RiskCard footnote + synthetic disclosure) -- assert at least one, not exactly one
    expect(screen.getAllByText(/uc07-risk-synthetic-v1/).length).toBeGreaterThan(0);
  });
});

describe("Uc07DecisionPanel -- no client-side fabricated recommendation", () => {
  it("never renders a destination label the backend didn't return", () => {
    render(
      <Uc07DecisionPanel
        decision={makeDecision({
          navigation: { destination: null, reason_codes: [], explanation: "x" },
          safety: { state: "OVERRIDE", override: true },
        })}
      />,
    );
    for (const label of ["Primary Care", "Urgent Care", "Telehealth", "Care Management"]) {
      expect(screen.queryByText(label)).not.toBeInTheDocument();
    }
  });
});
