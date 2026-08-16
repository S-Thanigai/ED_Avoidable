import { describe, expect, it } from "vitest";
import { buildSafetyContextPayload } from "../api";

describe("buildSafetyContextPayload -- missing must never become false", () => {
  it("omits unset fields entirely rather than sending 0", () => {
    const payload = buildSafetyContextPayload("M1", { red_flag: 1 });
    expect(payload).toEqual({ M1: { red_flag: 1 } });
    expect(payload?.M1).not.toHaveProperty("icu");
    expect(payload?.M1).not.toHaveProperty("admitted");
    expect(payload?.M1).not.toHaveProperty("major_procedure");
    expect(payload?.M1).not.toHaveProperty("triage_level");
  });

  it("returns undefined (no context supplied) when every field is unknown", () => {
    const payload = buildSafetyContextPayload("M1", {});
    expect(payload).toBeUndefined();
  });

  it("preserves an explicit 0 as a known false value distinct from omission", () => {
    const payload = buildSafetyContextPayload("M1", { red_flag: 0 });
    expect(payload).toEqual({ M1: { red_flag: 0 } });
  });

  it("supports a fully complete context", () => {
    const payload = buildSafetyContextPayload("M1", {
      red_flag: 0,
      icu: 0,
      admitted: 0,
      major_procedure: 0,
      triage_level: 4,
    });
    expect(payload).toEqual({ M1: { red_flag: 0, icu: 0, admitted: 0, major_procedure: 0, triage_level: 4 } });
  });
});
