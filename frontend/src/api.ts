import type { PredictResponse, ShapExplanation, UploadFiles } from "./types";

const API_BASE_URL =
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://127.0.0.1:8001";

export async function runPrediction(files: UploadFiles): Promise<PredictResponse> {
  if (!files.members || !files.edVisits || !files.care) {
    throw new Error("All three CSV files are required.");
  }

  const formData = new FormData();
  formData.append("members_file", files.members);
  formData.append("ed_visits_file", files.edVisits);
  formData.append("care_file", files.care);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/predict-json`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error(
      `Could not reach the prediction API at ${API_BASE_URL}. Is the backend running?`,
    );
  }

  if (!response.ok) {
    let detail = `Prediction request failed (HTTP ${response.status}).`;
    try {
      const err = await response.json();
      if (err?.detail) detail = err.detail;
    } catch {
      /* body wasn't JSON, keep default message */
    }
    throw new Error(detail);
  }

  return response.json() as Promise<PredictResponse>;
}

export async function explainMember(
  files: UploadFiles,
  memberId: string,
): Promise<ShapExplanation> {
  if (!files.members || !files.edVisits || !files.care) {
    throw new Error("Original CSV files are no longer available for this session.");
  }

  const formData = new FormData();
  formData.append("member_id", memberId);
  formData.append("members_file", files.members);
  formData.append("ed_visits_file", files.edVisits);
  formData.append("care_file", files.care);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/explain-member`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error(`Could not reach the prediction API at ${API_BASE_URL}.`);
  }

  if (!response.ok) {
    let detail = `Explanation request failed (HTTP ${response.status}).`;
    try {
      const err = await response.json();
      if (err?.detail) detail = err.detail;
    } catch {
      /* body wasn't JSON, keep default message */
    }
    throw new Error(detail);
  }

  return response.json() as Promise<ShapExplanation>;
}
