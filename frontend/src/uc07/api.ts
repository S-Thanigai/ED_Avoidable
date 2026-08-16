// Authoritative UC07 API client -- the ONLY place this app is allowed to
// call POST /uc07/decide, GET /model-info, GET /health. Every risk tier,
// navigation destination, and safety state a component ever renders
// comes from a response returned by one of these functions -- nothing
// here computes, infers, or overrides a decision; it only transports and
// types the backend's own response.
import { API_BASE_URL } from "../apiConfig";
import type {
  CurrentSafetyContextPayload,
  DecideUC07Params,
  ExplainRequest,
  FinalUC07Decision,
  HealthResponse,
  MemberExplanationResponse,
  ModelInfoResponse,
  UC07DecideResponse,
} from "./types";

export class UC07ApiError extends Error {
  readonly status: number | null;
  constructor(message: string, status: number | null) {
    super(message);
    this.name = "UC07ApiError";
    this.status = status;
  }
}

async function parseErrorDetail(response: Response, fallback: string): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
      if (detail !== undefined) return JSON.stringify(detail);
    }
  } catch {
    /* body wasn't JSON -- keep the fallback, never surface a raw parse error */
  }
  return fallback;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new UC07ApiError(
      `Could not reach the UC07 decision service at ${API_BASE_URL}. Is the backend running?`,
      null,
    );
  }
  if (!response.ok) {
    const detail = await parseErrorDetail(response, `UC07 request failed (HTTP ${response.status}).`);
    throw new UC07ApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

export async function decideUC07(params: DecideUC07Params): Promise<UC07DecideResponse> {
  const { files, memberId, indexDate, currentSafetyContext, safetyContextFile } = params;
  if (!files.members || !files.edVisits || !files.care) {
    throw new UC07ApiError("Members, ED visits, and care history CSVs are all required.", null);
  }

  const formData = new FormData();
  formData.append("members_file", files.members);
  formData.append("ed_visits_file", files.edVisits);
  formData.append("care_file", files.care);
  if (memberId) formData.append("member_id", memberId);
  if (indexDate) formData.append("index_date", indexDate);
  if (currentSafetyContext && Object.keys(currentSafetyContext).length > 0) {
    formData.append("current_safety_context", JSON.stringify(currentSafetyContext));
  }
  // Optional batch current_safety_context.csv (Phase 8B Part B). Used
  // only by the Safety Agent -- never a model feature, never joined into
  // the risk-scoring path. A member also present in `currentSafetyContext`
  // above (e.g. a single-member ad-hoc override) takes precedence over
  // the CSV for that member, matching the backend's documented merge rule.
  if (safetyContextFile) {
    formData.append("safety_context_file", safetyContextFile);
  }

  return requestJson<UC07DecideResponse>("/uc07/decide", { method: "POST", body: formData });
}

/** Builds the minimal, allow-listed POST /uc07/explain request body FROM
 * an already-fetched FinalUC07Decision -- transports only what the
 * backend's ExplainRequest schema accepts (risk probability/tier/
 * model_version/factors, navigation destination/reason_codes, safety
 * state/context_completeness/context_source, synthetic_model). Never
 * sends member_id, raw member data, or anything not already visible in
 * this decision. Purely a data-shaping function -- computes nothing. */
export function buildExplainRequest(decision: FinalUC07Decision): ExplainRequest {
  return {
    risk: {
      probability: decision.risk.probability,
      tier: decision.risk.tier,
      model_version: decision.risk.model_version,
      factors: decision.risk.explanation_factors.map((f) => ({
        display_name: f.display_name,
        direction: f.direction,
      })),
    },
    navigation: {
      destination: decision.navigation.destination,
      reason_codes: decision.navigation.reason_codes,
    },
    safety: {
      state: decision.safety.state,
      context_completeness: decision.safety.context_completeness,
      context_source: decision.safety.context_source,
    },
    synthetic_model: decision.risk.synthetic_model,
  };
}

/** Lazy, on-demand, single-member only (Phase 8C Part 14) -- callers
 * must invoke this at most once per member the user actually opens,
 * never for a whole uploaded population. Always resolves with a
 * MemberExplanationResponse (GenAI or DETERMINISTIC_FALLBACK); this
 * function itself never decides which -- the backend does. */
export async function explainMember(decision: FinalUC07Decision): Promise<MemberExplanationResponse> {
  return requestJson<MemberExplanationResponse>("/uc07/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildExplainRequest(decision)),
  });
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  return requestJson<ModelInfoResponse>("/model-info", { method: "GET" });
}

export async function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health", { method: "GET" });
}

/** Builds the current_safety_context payload for a single member from
 * discrete tri-state form values -- undefined stays UNKNOWN (omitted),
 * never coerced to a known 0/false. Returns undefined (no key emitted)
 * if every field is unknown, matching "no context supplied for this
 * member" at the API layer. */
export function buildSafetyContextPayload(
  memberId: string,
  entry: {
    red_flag?: 0 | 1;
    icu?: 0 | 1;
    admitted?: 0 | 1;
    major_procedure?: 0 | 1;
    triage_level?: 1 | 2 | 3 | 4 | 5;
  },
): CurrentSafetyContextPayload | undefined {
  const cleaned = Object.fromEntries(
    Object.entries(entry).filter(([, value]) => value !== undefined),
  );
  if (Object.keys(cleaned).length === 0) return undefined;
  return { [memberId]: cleaned };
}
