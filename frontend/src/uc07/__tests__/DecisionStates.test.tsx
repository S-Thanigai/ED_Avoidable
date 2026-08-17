import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { DecisionLoading } from "../components/DecisionLoading";
import { DecisionError } from "../components/DecisionError";
import { UC07ApiError } from "../api";

describe("DecisionLoading", () => {
  it("shows an accessible loading indicator", () => {
    render(<DecisionLoading />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });
});

describe("DecisionError -- validation error (422)", () => {
  it("shows a concise message, no fabricated recommendation, no stack trace", () => {
    render(<DecisionError error={new UC07ApiError("members_file failed validation: [...]", 422)} />);
    expect(screen.getByText("Invalid request data")).toBeInTheDocument();
    expect(screen.getByText(/decision unavailable/i)).toBeInTheDocument();
    expect(screen.queryByText(/Traceback/)).not.toBeInTheDocument();
  });
});

describe("DecisionError -- backend unavailable (network failure)", () => {
  it("shows a backend-unavailable message and does not fabricate a decision", () => {
    render(<DecisionError error={new UC07ApiError("Could not reach the UC07 decision service.", null)} />);
    expect(screen.getByText("Backend unavailable")).toBeInTheDocument();
    expect(screen.getByText(/decision unavailable/i)).toBeInTheDocument();
    // never shows any risk tier / navigation / safety text -- there is no decision to show
    for (const forbidden of ["LOW", "MODERATE", "HIGH", "CLEAR", "CAUTION", "OVERRIDE"]) {
      expect(screen.queryByText(forbidden)).not.toBeInTheDocument();
    }
  });
});

describe("DecisionError -- unknown member (404)", () => {
  it("shows member-not-found", () => {
    render(<DecisionError error={new UC07ApiError("member_id 'X' not found", 404)} />);
    expect(screen.getByText("Member not found")).toBeInTheDocument();
  });
});
