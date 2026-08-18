import type { UploadFiles } from "../types";

// Type contracts for the authoritative UC07 multi-agent decision system.
// Mirrors backend/agents/contracts.py + backend/agents/orchestrator.py's
// decision_to_dict() EXACTLY (verified against a live decode, see
// artifacts/phase8_frontend/frontend_api_contract.json) -- no field here
// is invented; every field the backend does not return is absent here.

export type RiskTier = "LOW" | "MODERATE" | "HIGH";

export type NavigationDestination =
  | "PRIMARY_CARE"
  | "URGENT_CARE"
  | "TELEHEALTH"
  | "CARE_MANAGEMENT"
  | "NO_PROACTIVE_NAVIGATION";

export type SafetyState = "CLEAR" | "CAUTION" | "OVERRIDE";

export type ContextCompleteness = "COMPLETE" | "PARTIAL" | "ABSENT";

export type ContextSource = "CALLER_SUPPLIED" | "SYSTEM_DERIVED" | "NOT_AVAILABLE";

export type ReasonCode =
  | "ELEVATED_FUTURE_RISK"
  | "REPEATED_LOWER_ACUITY_HISTORY"
  | "TRANSPORTATION_BARRIER"
  | "LIMITED_PCP_ACCESS"
  | "TELEHEALTH_AVAILABLE"
  | "CHRONIC_COMPLEXITY"
  | "PRIOR_CM_ENGAGEMENT"
  | "OUTPATIENT_CONTINUITY_OPPORTUNITY"
  | "URGENT_CARE_ACCESS_ADVANTAGE"
  | "NO_OPPORTUNITY_IDENTIFIED";

// Phase 8C -- authoritative, structured model explainability. NEVER a
// causal claim: `direction` describes the sign of a feature's
// contribution to the model's OWN estimate, not a real-world cause.
export type FactorDirection = "INCREASES_RISK" | "DECREASES_RISK";

export type ExplanationMethod = "SHAP_LINEAR" | "LINEAR_CONTRIBUTION";

export interface ExplanationFactor {
  feature: string;
  display_name: string;
  direction: FactorDirection;
  contribution: number;
  explanation_method: ExplanationMethod;
}

export interface RiskAssessment {
  member_id: string;
  probability: number;
  tier: RiskTier;
  contributing_factors: string[];
  model_version: string;
  dataset_id: string;
  synthetic_model: boolean;
  index_date: string; // ISO date
  moderate_threshold: number;
  high_threshold: number;
  // Phase 8C
  explanation_factors: ExplanationFactor[];
  explanation_method: ExplanationMethod;
  explanation_causal: boolean;
}

export interface FinalNavigationView {
  destination: NavigationDestination | null; // null only when safety.state === "OVERRIDE"
  reason_codes: ReasonCode[];
  explanation: string;
}

export interface SafetyDecision {
  state: SafetyState;
  override: boolean;
  message: string;
  blocked_phrases: string[];
  context_completeness: ContextCompleteness;
  context_source: ContextSource;
}

export interface FinalUC07Decision {
  member_id: string;
  risk: RiskAssessment;
  navigation: FinalNavigationView;
  safety: SafetyDecision;
  disclaimer: string;
}

export interface UC07DecideResponse {
  model_version: string;
  dataset_id: string;
  synthetic_model: boolean;
  index_date: string;
  count: number;
  decisions: FinalUC07Decision[];
}

// ---- Request-side types ----

// Distinguishes "caller knows this is 0/false" from "caller does not
// know" -- undefined/omitted means UNKNOWN, never coerced to false.
// Mirrors backend/agents/safety_context_schema.py's SafetyContextEntry:
// an omitted key stays absent, never defaults to 0.
export interface CurrentSafetyContextInput {
  red_flag?: 0 | 1;
  icu?: 0 | 1;
  admitted?: 0 | 1;
  major_procedure?: 0 | 1;
  triage_level?: 1 | 2 | 3 | 4 | 5;
}

export type CurrentSafetyContextPayload = Record<string, CurrentSafetyContextInput>;

export interface DecideUC07Params {
  files: UploadFiles;
  memberId?: string;
  indexDate?: string; // ISO date, optional -- backend defaults to today
  currentSafetyContext?: CurrentSafetyContextPayload;
  /** Optional current_safety_context.csv (member_id, red_flag, icu,
   * admitted, major_procedure, triage_level) -- an alternative, batch-
   * friendly way to supply the same CurrentSafetyContextPayload data.
   * Used only by the Safety Agent; never a model feature. */
  safetyContextFile?: File;
}

// ---- GET /model-info ----

export interface ModelInfoResponse {
  model_version: string;
  dataset_id: string;
  synthetic_model: boolean;
  target: string | null;
  target_definition: string | null;
  prediction_horizon_days: number | null;
  observation_window_days: number | null;
  moderate_threshold: number;
  high_threshold: number;
  feature_count: number;
  algorithm: string | null;
  intended_use: string | null;
  disclaimer: string | null;
}

// ---- GET /health ----

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  uc07_model_loaded: boolean;
  uc07_model_version: string | null;
  uc07_model_error: string | null;
}

// ---- POST /uc07/explain (Phase 8C -- GenAI Explanation Agent) ----
//
// Strict input boundary: the request body is ONLY the minimal,
// already-computed decision summary below -- never a CSV, never a raw
// member history. Built entirely FROM a FinalUC07Decision this app
// already has (see buildExplainRequest in api.ts); nothing here is
// invented client-side.

export interface ExplainRequest {
  risk: {
    probability: number;
    tier: RiskTier;
    model_version: string;
    factors: Array<{ display_name: string; direction: FactorDirection }>;
  };
  navigation: {
    destination: NavigationDestination | null;
    reason_codes: ReasonCode[];
  };
  safety: {
    state: SafetyState;
    context_completeness: ContextCompleteness;
    context_source: ContextSource;
  };
  synthetic_model: boolean;
}

export type ExplanationSource = "GENAI" | "DETERMINISTIC_FALLBACK";

export interface MemberExplanationResponse {
  summary: string;
  risk_explanation: string;
  navigation_explanation: string;
  safety_explanation: string;
  disclaimer: string;
  explanation_source: ExplanationSource;
  model_used: string | null;
  generation_time_ms: number | null;
}

// ---- Phase 9 -- POST /uc07/report + POST /uc07/email (Member
// Communication & Reporting). Same one-way principle as ExplainRequest
// above: request bodies are built entirely FROM an already-computed
// FinalUC07Decision (+ an already-approved MemberExplanationResponse,
// if one has been fetched) plus member CONTACT fields the care manager
// supplies -- never a new decision input. See
// frontend/src/uc07/memberContacts.ts for where member/email comes
// from (the member dataset itself has no email column).

export interface ReportMemberInput {
  member_id: string;
  name?: string | null;
  email?: string | null;
  age?: number | null;
  gender?: string | null;
}

export interface ReportRiskInput {
  probability: number;
  tier: RiskTier;
  model_version: string;
  factors: Array<{ display_name: string; direction: FactorDirection }>;
}

export interface ReportNavigationInput {
  destination: NavigationDestination | null;
  reason_codes: ReasonCode[];
}

export interface ReportSafetyInput {
  state: SafetyState;
  context_completeness: ContextCompleteness;
  context_source: ContextSource;
  message: string;
}

export interface ReportExplanationInput {
  summary: string;
  risk_explanation: string;
  navigation_explanation: string;
  safety_explanation: string;
  disclaimer: string;
  explanation_source: ExplanationSource;
  model_used: string | null;
}

export interface ReportRequestPayload {
  member: ReportMemberInput;
  risk: ReportRiskInput;
  navigation: ReportNavigationInput;
  safety: ReportSafetyInput;
  synthetic_model: boolean;
  dataset_id?: string | null;
  /** If omitted, the backend renders the report using its own
   * deterministic fallback explanation -- it never calls a fresh LLM
   * just to build a PDF/email (see backend/main.py's
   * `_resolve_report_explanation`). */
  explanation?: ReportExplanationInput | null;
}

export interface EmailSendRequestPayload {
  report: ReportRequestPayload;
  to_email: string;
  subject: string;
  body: string;
}

export interface EmailSendResponse {
  sent: boolean;
  provider: string;
  message: string;
  error_code: string | null;
  report_id: string;
}
