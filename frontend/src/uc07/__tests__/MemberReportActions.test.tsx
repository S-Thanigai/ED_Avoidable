import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemberReportActions } from "../components/MemberReportActions";
import { makeDecision } from "./fixtures";
import { setMemberContact } from "../memberContacts";
import { UC07ApiError } from "../api";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, fetchMemberReportPdf: vi.fn() };
});

describe("MemberReportActions (Member Communication tab content)", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.clearAllMocks();
  });

  it("renders both communication actions", () => {
    const decision = makeDecision({ member_id: "M00123" });
    render(<MemberReportActions decision={decision} lookups={null} />);
    expect(screen.getByRole("heading", { name: "Member Communication" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download Report" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Email Member" })).toBeInTheDocument();
  });

  it("downloads the PDF via fetchMemberReportPdf when clicked", async () => {
    const { fetchMemberReportPdf } = await import("../api");
    vi.mocked(fetchMemberReportPdf).mockResolvedValue({
      blob: new Blob(["%PDF-1.4"], { type: "application/pdf" }),
      filename: "Member_Care_Navigation_Report_M00123.pdf",
    });
    const decision = makeDecision({ member_id: "M00123" });
    const user = userEvent.setup();
    render(<MemberReportActions decision={decision} lookups={null} />);

    await user.click(screen.getByRole("button", { name: "Download Report" }));

    await waitFor(() => expect(fetchMemberReportPdf).toHaveBeenCalledTimes(1));
    const request = vi.mocked(fetchMemberReportPdf).mock.calls[0][0];
    expect(request.member.member_id).toBe("M00123");
    expect(request.risk.tier).toBe(decision.risk.tier);
    expect(await screen.findByText("Report downloaded.")).toBeInTheDocument();
  });

  it("shows a clean error message if report generation fails", async () => {
    const { fetchMemberReportPdf } = await import("../api");
    vi.mocked(fetchMemberReportPdf).mockRejectedValue(new UC07ApiError("Report generation failed.", 500));
    const decision = makeDecision({ member_id: "M00123" });
    const user = userEvent.setup();
    render(<MemberReportActions decision={decision} lookups={null} />);

    await user.click(screen.getByRole("button", { name: "Download Report" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Report generation failed.");
  });

  it("opens the email composer with the member's saved contact email prefilled", async () => {
    setMemberContact("M00123", { email: "saved.contact@example.com", name: "Jordan Lee" });
    const decision = makeDecision({ member_id: "M00123" });
    const user = userEvent.setup();
    render(<MemberReportActions decision={decision} lookups={null} />);

    await user.click(screen.getByRole("button", { name: "Email Member" }));

    expect(screen.getByRole("dialog", { name: /send care navigation report/i })).toBeInTheDocument();
    expect(screen.getByLabelText("Recipient")).toHaveValue("saved.contact@example.com");
  });

  it("closing the composer removes it from the DOM", async () => {
    const decision = makeDecision({ member_id: "M00123" });
    const user = userEvent.setup();
    render(<MemberReportActions decision={decision} lookups={null} />);

    await user.click(screen.getByRole("button", { name: "Email Member" }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    await user.click(screen.getByLabelText("Close email composer"));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
