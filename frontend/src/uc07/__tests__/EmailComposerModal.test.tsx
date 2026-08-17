import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { EmailComposerModal } from "../components/EmailComposerModal";
import { makeDecision } from "./fixtures";
import { UC07ApiError } from "../api";
import { getMemberContact } from "../memberContacts";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, fetchMemberReportPdf: vi.fn(), sendMemberReportEmail: vi.fn() };
});

const MEMBER = { name: "Jordan Lee", email: "jordan.lee@example.com", age: 57, gender: "F" };

function renderComposer(overrides: Parameters<typeof makeDecision>[0] = {}, member = MEMBER) {
  const decision = makeDecision({ member_id: "M00123", ...overrides });
  const onClose = vi.fn();
  const utils = render(<EmailComposerModal decision={decision} member={member} onClose={onClose} />);
  return { ...utils, decision, onClose };
}

describe("EmailComposerModal", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.clearAllMocks();
  });

  it("prefills recipient from the member contact, a safe default subject, and an editable body", () => {
    renderComposer();
    expect(screen.getByLabelText("Recipient")).toHaveValue("jordan.lee@example.com");
    expect(screen.getByLabelText("Subject")).toHaveValue("Your Care Navigation Summary");
    const body = screen.getByLabelText("Message") as HTMLTextAreaElement;
    expect(body.value).toContain("Hello Jordan Lee,");
    expect(body.value).toContain("does not replace medical evaluation");
    expect(body.value).toContain("do not delay seeking appropriate emergency evaluation");
  });

  it("recipient, subject, and body are all editable", async () => {
    const user = userEvent.setup();
    renderComposer();

    await user.clear(screen.getByLabelText("Recipient"));
    await user.type(screen.getByLabelText("Recipient"), "new.recipient@example.com");
    await user.clear(screen.getByLabelText("Subject"));
    await user.type(screen.getByLabelText("Subject"), "Edited subject");
    await user.clear(screen.getByLabelText("Message"));
    await user.type(screen.getByLabelText("Message"), "Edited body text.");

    expect(screen.getByLabelText("Recipient")).toHaveValue("new.recipient@example.com");
    expect(screen.getByLabelText("Subject")).toHaveValue("Edited subject");
    expect(screen.getByLabelText("Message")).toHaveValue("Edited body text.");
  });

  it("shows a validation error and disables Send for a malformed recipient address", async () => {
    const user = userEvent.setup();
    renderComposer();

    await user.clear(screen.getByLabelText("Recipient"));
    await user.type(screen.getByLabelText("Recipient"), "not-an-email");

    expect(screen.getByText("Enter a valid email address.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /review & send/i })).toBeDisabled();
  });

  it("requires an explicit two-step confirmation before actually sending", async () => {
    const { sendMemberReportEmail } = await import("../api");
    vi.mocked(sendMemberReportEmail).mockResolvedValue({
      sent: true, provider: "smtp", message: "Report sent to j***@example.com.", error_code: null, report_id: "UC07-RPT-1",
    });
    const user = userEvent.setup();
    renderComposer();

    await user.click(screen.getByRole("button", { name: /review & send/i }));
    expect(sendMemberReportEmail).not.toHaveBeenCalled();
    expect(screen.getByText("Confirm Send")).toBeInTheDocument();
    expect(screen.getByText("jordan.lee@example.com", { exact: false })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Send Email" }));
    await waitFor(() => expect(sendMemberReportEmail).toHaveBeenCalledTimes(1));
  });

  it("cancelling the confirmation prompt never sends", async () => {
    const { sendMemberReportEmail } = await import("../api");
    const user = userEvent.setup();
    renderComposer();

    await user.click(screen.getByRole("button", { name: /review & send/i }));
    await user.click(screen.getByRole("button", { name: "Back" }));

    expect(sendMemberReportEmail).not.toHaveBeenCalled();
    expect(screen.queryByText("Confirm Send")).not.toBeInTheDocument();
  });

  it("shows a success state and remembers the recipient contact after a successful send", async () => {
    const { sendMemberReportEmail } = await import("../api");
    vi.mocked(sendMemberReportEmail).mockResolvedValue({
      sent: true, provider: "smtp", message: "Report sent to j***@example.com.", error_code: null, report_id: "UC07-RPT-1",
    });
    const user = userEvent.setup();
    renderComposer();

    await user.click(screen.getByRole("button", { name: /review & send/i }));
    await user.click(screen.getByRole("button", { name: "Send Email" }));

    expect(await screen.findByText("Sent successfully.")).toBeInTheDocument();
    expect(screen.getByText(/report sent to j\*\*\*@example\.com/i)).toBeInTheDocument();
    expect(getMemberContact("M00123").email).toBe("jordan.lee@example.com");
  });

  it("shows a clean failure message on send failure, never a stack trace", async () => {
    const { sendMemberReportEmail } = await import("../api");
    vi.mocked(sendMemberReportEmail).mockResolvedValue({
      sent: false, provider: "smtp", message: "Could not connect to the email provider.", error_code: "NETWORK_ERROR", report_id: "UC07-RPT-2",
    });
    const user = userEvent.setup();
    renderComposer();

    await user.click(screen.getByRole("button", { name: /review & send/i }));
    await user.click(screen.getByRole("button", { name: "Send Email" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Failed to send: Could not connect to the email provider.");
    expect(alert.textContent).not.toMatch(/traceback/i);
  });

  it("generates and opens a PDF preview via fetchMemberReportPdf", async () => {
    const { fetchMemberReportPdf } = await import("../api");
    vi.mocked(fetchMemberReportPdf).mockResolvedValue({
      blob: new Blob(["%PDF-1.4"], { type: "application/pdf" }),
      filename: "Member_Care_Navigation_Report_M00123.pdf",
    });
    const user = userEvent.setup();
    renderComposer();

    await user.click(screen.getByRole("button", { name: "Preview PDF" }));

    await waitFor(() => expect(fetchMemberReportPdf).toHaveBeenCalledTimes(1));
    expect(window.open).toHaveBeenCalled();
    expect(await screen.findByText(/application\/pdf/)).toBeInTheDocument();
  });

  it("shows a clean preview error message on failure", async () => {
    const { fetchMemberReportPdf } = await import("../api");
    vi.mocked(fetchMemberReportPdf).mockRejectedValue(new UC07ApiError("Report generation failed.", 500));
    const user = userEvent.setup();
    renderComposer();

    await user.click(screen.getByRole("button", { name: "Preview PDF" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Report generation failed.");
  });

  it("Cancel closes the composer without sending", async () => {
    const { sendMemberReportEmail } = await import("../api");
    const user = userEvent.setup();
    const { onClose } = renderComposer();

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(sendMemberReportEmail).not.toHaveBeenCalled();
  });

  it("Escape closes the composer (keyboard accessibility)", async () => {
    const user = userEvent.setup();
    const { onClose } = renderComposer();
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("focuses the close button on open", () => {
    renderComposer();
    expect(screen.getByLabelText("Close email composer")).toHaveFocus();
  });
});
