import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Pagination } from "../components/Pagination";

describe("Pagination", () => {
  it("shows the correct range for the first page", () => {
    render(<Pagination page={1} totalItems={137} onPageChange={() => {}} />);
    expect(screen.getByText("Showing 1–25 of 137 members")).toBeInTheDocument();
  });

  it("shows the correct range for an interior page", () => {
    render(<Pagination page={2} totalItems={137} onPageChange={() => {}} />);
    expect(screen.getByText("Showing 26–50 of 137 members")).toBeInTheDocument();
  });

  it("clamps the last page's range to the true item count", () => {
    render(<Pagination page={6} totalItems={137} onPageChange={() => {}} />);
    expect(screen.getByText("Showing 126–137 of 137 members")).toBeInTheDocument();
  });

  it("disables Previous on the first page and Next on the last page", () => {
    const { rerender } = render(<Pagination page={1} totalItems={137} onPageChange={() => {}} />);
    expect(screen.getByText("Previous")).toBeDisabled();
    expect(screen.getByText("Next")).not.toBeDisabled();

    rerender(<Pagination page={6} totalItems={137} onPageChange={() => {}} />);
    expect(screen.getByText("Previous")).not.toBeDisabled();
    expect(screen.getByText("Next")).toBeDisabled();
  });

  it("highlights the current page", () => {
    render(<Pagination page={3} totalItems={137} onPageChange={() => {}} />);
    const current = screen.getByRole("button", { name: "3" });
    expect(current).toHaveAttribute("aria-current", "page");
  });

  it("calls onPageChange with the target page when Next is clicked", async () => {
    const user = userEvent.setup();
    const onPageChange = vi.fn();
    render(<Pagination page={1} totalItems={137} onPageChange={onPageChange} />);
    await user.click(screen.getByText("Next"));
    expect(onPageChange).toHaveBeenCalledWith(2);
  });

  it("renders nothing when there are no items", () => {
    const { container } = render(<Pagination page={1} totalItems={0} onPageChange={() => {}} />);
    expect(container).toBeEmptyDOMElement();
  });
});
