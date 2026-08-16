import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SafetyContextForm } from "../components/SafetyContextForm";

describe("SafetyContextForm -- unknown stays unknown, never becomes false", () => {
  it("starts every field on Unknown by default", () => {
    render(<SafetyContextForm value={{}} onChange={() => {}} />);
    const selects = screen.getAllByRole("combobox") as HTMLSelectElement[];
    for (const select of selects) {
      expect(select.value).toBe("");
    }
  });

  it("emits an explicit 0/1, not undefined, once a value is chosen", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<SafetyContextForm value={{}} onChange={onChange} />);
    const redFlagSelect = screen.getByLabelText(/red-flag symptom present/i);
    await user.selectOptions(redFlagSelect, "1");
    expect(onChange).toHaveBeenCalledWith({ red_flag: 1 });
  });

  it("removes the key entirely when reset back to Unknown", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<SafetyContextForm value={{ red_flag: 1 }} onChange={onChange} />);
    const redFlagSelect = screen.getByLabelText(/red-flag symptom present/i);
    await user.selectOptions(redFlagSelect, "");
    expect(onChange).toHaveBeenCalledWith({});
  });
});
