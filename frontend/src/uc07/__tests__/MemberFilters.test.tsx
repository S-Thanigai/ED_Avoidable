import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemberFilters } from "../components/MemberFilters";
import { DEFAULT_FILTERS, type MemberFiltersState } from "../tableState";

describe("MemberFilters", () => {
  it("reports a changed filter via onChange without mutating the caller's state object", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<MemberFilters filters={DEFAULT_FILTERS} onChange={onChange} totalCount={10} filteredCount={10} />);

    await user.selectOptions(screen.getByText("Risk: All").closest("select")!, "HIGH");
    expect(onChange).toHaveBeenCalledWith({ ...DEFAULT_FILTERS, tier: "HIGH" });
    expect(DEFAULT_FILTERS.tier).toBe("ALL"); // original object untouched
  });

  it("shows active-filter chips and clears a single field when its chip is clicked", async () => {
    const filters: MemberFiltersState = { ...DEFAULT_FILTERS, tier: "HIGH", safety: "OVERRIDE" };
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<MemberFilters filters={filters} onChange={onChange} totalCount={10} filteredCount={2} />);

    expect(screen.getByText("HIGH")).toBeInTheDocument();
    expect(screen.getByText("OVERRIDE")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /remove filter HIGH/i }));
    expect(onChange).toHaveBeenCalledWith({ ...filters, tier: "ALL" });
  });

  it("Clear all filters resets to DEFAULT_FILTERS and is disabled when nothing is active", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    const { rerender } = render(
      <MemberFilters filters={DEFAULT_FILTERS} onChange={onChange} totalCount={10} filteredCount={10} />,
    );
    expect(screen.getByRole("button", { name: /clear all filters/i })).toBeDisabled();

    const active: MemberFiltersState = { ...DEFAULT_FILTERS, search: "M001" };
    rerender(<MemberFilters filters={active} onChange={onChange} totalCount={10} filteredCount={1} />);
    const clearButton = screen.getByRole("button", { name: /clear all filters/i });
    expect(clearButton).toBeEnabled();
    await user.click(clearButton);
    expect(onChange).toHaveBeenCalledWith(DEFAULT_FILTERS);
  });

  it("applies a probability preset to both min and max fields at once", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<MemberFilters filters={DEFAULT_FILTERS} onChange={onChange} totalCount={10} filteredCount={10} />);
    await user.click(screen.getByRole("button", { name: "20–30%" }));
    expect(onChange).toHaveBeenCalledWith({ ...DEFAULT_FILTERS, probMin: "20", probMax: "30" });
  });
});
