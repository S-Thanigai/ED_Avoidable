import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SafetyContextCsvUpload } from "../components/SafetyContextCsvUpload";

describe("SafetyContextCsvUpload", () => {
  it("is clearly labeled Optional and explains it does not affect ML risk prediction", () => {
    render(<SafetyContextCsvUpload file={null} onChange={() => {}} disabled={false} />);
    expect(screen.getByText("Optional")).toBeInTheDocument();
    expect(screen.getByText(/does not affect ml risk prediction/i)).toBeInTheDocument();
  });

  it("reports a selected file via onChange", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const { container } = render(<SafetyContextCsvUpload file={null} onChange={onChange} disabled={false} />);
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const csv = new File(["member_id,red_flag\nM1,1\n"], "current_safety_context.csv", { type: "text/csv" });
    await user.upload(input, csv);
    expect(onChange).toHaveBeenCalledWith(csv);
  });

  it("shows the selected filename and allows removal", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const file = new File(["x"], "current_safety_context.csv");
    render(<SafetyContextCsvUpload file={file} onChange={onChange} disabled={false} />);
    expect(screen.getByText("current_safety_context.csv")).toBeInTheDocument();
    await user.click(screen.getByLabelText("Remove current safety context file"));
    expect(onChange).toHaveBeenCalledWith(null);
  });
});
