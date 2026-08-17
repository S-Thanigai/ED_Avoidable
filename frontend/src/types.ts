export type RiskCategory = "Low" | "Medium" | "High";

export type RecommendedCare = "PCP" | "Urgent Care" | "Telehealth";

export interface PatientRow {
  member_id: string;
  risk_probability: number;
  risk_score: number;
  predicted_frequent_ED: number;
  risk_category: RiskCategory;
  age: number;
  gender: string;
  diabetes: number;
  copd: number;
  hypertension: number;
  chf: number;
  asthma: number;
  ckd: number;
  transportation_barrier: number;
  telehealth_available: number;
  pcp_distance_miles: number;
  urgent_care_distance_miles: number;
  days_since_last_ED: number | null;
  days_since_last_care: number | null;
  alternative_care_visits: number;
  total_non_ED_care: number;
  access_burden: number;
  clinical_burden: number;
  recommended_alternative_care: RecommendedCare | string;
  alternative_care_reason: string;
}

export interface ShapExplanation {
  shap_feature_1: string;
  shap_value_1: number;
  shap_feature_2: string;
  shap_value_2: number;
  shap_feature_3: string;
  shap_value_3: number;
  shap_explanation_summary: string;
  shap_explanation_json: string;
}

export interface PredictResponse {
  columns: string[];
  rows: PatientRow[];
  count: number;
}

export interface UploadFiles {
  members: File | null;
  edVisits: File | null;
  care: File | null;
}
