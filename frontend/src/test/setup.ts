import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

afterEach(() => {
  cleanup();
});

// jsdom does not implement Blob URL creation or window.open -- Phase 9's
// report download / PDF preview (MemberReportActions.tsx,
// EmailComposerModal.tsx) use both. Stubbed globally (not per-test) so
// any test that renders the member workspace, not just the report/email
// tests themselves, doesn't hit a "not implemented" jsdom error.
URL.createObjectURL = vi.fn(() => "blob:mock-url");
URL.revokeObjectURL = vi.fn();
window.open = vi.fn();
