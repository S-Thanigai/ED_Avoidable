import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderHook, act } from "@testing-library/react";
import { Header } from "../components/Header";
import { useTheme } from "../useTheme";

// Part 27 test 16: light/dark theme both work -- the toggle actually
// flips document.documentElement's data-theme attribute (what tokens.css
// keys off of) and every themed surface re-renders without crashing.
describe("theme (Part 27 test 16)", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute("data-theme");
  });
  afterEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute("data-theme");
  });

  it("defaults to dark and stamps data-theme on <html>", () => {
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("toggling flips between light and dark, updating the DOM attribute each time", () => {
    const { result } = renderHook(() => useTheme());
    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe("light");
    expect(document.documentElement.dataset.theme).toBe("light");

    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("Header renders correctly and calls back in both theme states", async () => {
    const user = userEvent.setup();
    let theme: "light" | "dark" = "dark";
    let toggled = false;
    const { rerender } = render(<Header theme={theme} onToggleTheme={() => (toggled = true)} />);
    expect(screen.getByRole("button", { name: /switch to light mode/i })).toBeInTheDocument();

    theme = "light";
    rerender(<Header theme={theme} onToggleTheme={() => (toggled = true)} />);
    const toggle = screen.getByRole("button", { name: /switch to dark mode/i });
    await user.click(toggle);
    expect(toggled).toBe(true);
  });
});
