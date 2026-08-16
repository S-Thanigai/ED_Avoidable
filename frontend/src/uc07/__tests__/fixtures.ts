import type { FinalUC07Decision } from "../types";

/** Base decision fixture, deep-overridable per test. Values are shaped
 * exactly like a real backend response (see
 * artifacts/phase8_frontend/frontend_api_contract.json, captured from a
 * live decode of orchestrator.decision_to_dict()). */
export function makeDecision(overrides: {
  risk?: Partial<FinalUC07Decision["risk"]>;
  navigation?: Partial<FinalUC07Decision["navigation"]>;
  safety?: Partial<FinalUC07Decision["safety"]>;
  member_id?: string;
}): FinalUC07Decision {
  return {
    member_id: overrides.member_id ?? "M00001",
    risk: {
      member_id: overrides.member_id ?? "M00001",
      probability: 0.05,
      tier: "LOW",
      contributing_factors: ["A historical utilization or access pattern contributed to lower risk."],
      model_version: "uc07-risk-synthetic-v1",
      dataset_id: "synthetic_uc07_v1",
      synthetic_model: true,
      index_date: "2026-07-03",
      moderate_threshold: 0.105986,
      high_threshold: 0.213252,
      explanation_factors: [
        { feature: "access_burden", display_name: "Access burden", direction: "DECREASES_RISK", contribution: -0.12, explanation_method: "SHAP_LINEAR" },
      ],
      explanation_method: "SHAP_LINEAR",
      explanation_causal: false,
      ...overrides.risk,
    },
    navigation: {
      destination: "NO_PROACTIVE_NAVIGATION",
      reason_codes: ["NO_OPPORTUNITY_IDENTIFIED"],
      explanation: "Based on this member's current pattern, no meaningful navigation opportunity was identified at this time.",
      ...overrides.navigation,
    },
    safety: {
      state: "CAUTION",
      override: false,
      message:
        "Current clinical/safety context for this encounter was not provided, so this system cannot confirm the current situation is non-emergency. For non-emergency situations only: if you are experiencing a medical emergency, call 911 or go to the nearest emergency department immediately.",
      blocked_phrases: [],
      context_completeness: "ABSENT",
      context_source: "NOT_AVAILABLE",
      ...overrides.safety,
    },
    disclaimer:
      "For care navigation only -- never a reason to delay care. This tool flags patterns of potentially avoidable ED use and suggests a lower-acuity next step for FUTURE, non-emergency needs. It does not diagnose, triage, or determine medical necessity, and no output here should ever be read as discouraging emergency care. If you or a member may be experiencing a medical emergency, call 911 or go to the nearest emergency department immediately.",
  };
}
