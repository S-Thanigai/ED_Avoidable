// Saved-population API client. Every call sends credentials: "include"
// (session cookie) -- ownership itself is enforced server-side
// (backend/routers/populations.py + db/repositories/populations.py),
// this client never sends or trusts a user id.
import { API_BASE_URL } from "../apiConfig";
import type { UploadFiles } from "../types";
import type { CurrentSafetyContextPayload } from "../uc07/types";
import type { MemberDetail, MemberListParams, PaginatedMembers, PopulationDetail, PopulationSummary } from "./types";

export class PopulationsApiError extends Error {
  readonly status: number | null;
  constructor(message: string, status: number | null) {
    super(message);
    this.name = "PopulationsApiError";
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
    /* not JSON -- keep the fallback */
  }
  return fallback;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...init, credentials: "include" });
  } catch {
    throw new PopulationsApiError(`Could not reach the backend at ${API_BASE_URL}. Is it running?`, null);
  }
  if (!response.ok) {
    const detail = await parseErrorDetail(response, `Request failed (HTTP ${response.status}).`);
    throw new PopulationsApiError(detail, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export function listPopulations(): Promise<PopulationSummary[]> {
  return requestJson<PopulationSummary[]>("/populations", { method: "GET" });
}

export function getPopulation(populationId: number): Promise<PopulationDetail> {
  return requestJson<PopulationDetail>(`/populations/${populationId}`, { method: "GET" });
}

export function deletePopulation(populationId: number): Promise<void> {
  return requestJson<void>(`/populations/${populationId}`, { method: "DELETE" });
}

export function listMembers(populationId: number, params: MemberListParams): Promise<PaginatedMembers> {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") query.set(key, String(value));
  }
  const qs = query.toString();
  return requestJson<PaginatedMembers>(`/populations/${populationId}/members${qs ? `?${qs}` : ""}`, {
    method: "GET",
  });
}

export function getMember(populationId: number, memberId: string): Promise<MemberDetail> {
  return requestJson<MemberDetail>(
    `/populations/${populationId}/members/${encodeURIComponent(memberId)}`,
    { method: "GET" },
  );
}

export interface SaveAnalysisParams {
  name: string;
  files: UploadFiles;
  indexDate: string;
  currentSafetyContext?: CurrentSafetyContextPayload;
  safetyContextFile?: File | null;
}

/** Recomputes the decision server-side from the SAME files/index_date/
 * safety context already used for the on-screen analysis (see
 * backend/uc07_pipeline.py) -- never sends the already-rendered
 * FinalUC07Decision[] as if it were authoritative. `indexDate` MUST be
 * the index_date the original decideUC07() response returned, so the
 * saved snapshot matches what's on screen rather than silently
 * re-dating to "today". */
export function saveAnalysis(params: SaveAnalysisParams): Promise<PopulationSummary> {
  const { name, files, indexDate, currentSafetyContext, safetyContextFile } = params;
  if (!files.members || !files.edVisits || !files.care) {
    throw new PopulationsApiError("Members, ED visits, and care history CSVs are all required.", null);
  }
  const formData = new FormData();
  formData.append("name", name);
  formData.append("index_date", indexDate);
  formData.append("members_file", files.members);
  formData.append("ed_visits_file", files.edVisits);
  formData.append("care_file", files.care);
  if (currentSafetyContext && Object.keys(currentSafetyContext).length > 0) {
    formData.append("current_safety_context", JSON.stringify(currentSafetyContext));
  }
  if (safetyContextFile) {
    formData.append("safety_context_file", safetyContextFile);
  }
  return requestJson<PopulationSummary>("/populations/save-analysis", { method: "POST", body: formData });
}
