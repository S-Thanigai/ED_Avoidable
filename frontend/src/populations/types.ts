import type { FinalUC07Decision } from "../uc07/types";

export interface PopulationSummary {
  id: number;
  name: string;
  member_count: number;
  index_date: string;
  model_version: string;
  dataset_id: string;
  synthetic_model: boolean;
  created_at: string;
  updated_at: string;
}

export interface PopulationDetail extends PopulationSummary {
  tier_counts: Record<string, number>;
  safety_counts: Record<string, number>;
  navigation_counts: Record<string, number>;
  probability_bins: number[];
  moderate_threshold: number | null;
  high_threshold: number | null;
}

export interface PaginatedMembers {
  items: FinalUC07Decision[];
  page: number;
  page_size: number;
  total_items: number;
  total_pages: number;
}

export interface MemberProfile {
  member_id: string;
  age: number;
  gender: string;
  diabetes: number;
  copd: number;
  hypertension: number;
  chf: number;
  asthma: number;
  ckd: number;
  num_chronic_conditions: number;
  transportation_barrier: number;
  telehealth_available: number;
  pcp_distance_miles: number;
  urgent_care_distance_miles: number;
}

export interface MemberEdVisit {
  visit_id: string | null;
  member_id: string;
  visit_date: string;
  diagnosis: string | null;
  triage_level: number;
  admitted: number;
  icu: number;
  major_procedure: number;
  cost: number;
  red_flag: number;
}

export interface MemberCareVisit {
  care_id: string | null;
  member_id: string;
  visit_date: string;
  care_type: string;
}

export interface MemberDetail {
  decision: FinalUC07Decision;
  profile: MemberProfile | null;
  ed_visits: MemberEdVisit[];
  care_visits: MemberCareVisit[];
  safety_context_captured_at: string | null;
}

export interface MemberListParams {
  page?: number;
  page_size?: number;
  search?: string;
  tier?: string;
  navigation?: string;
  safety?: string;
  prob_min?: number;
  prob_max?: number;
  sort_key?: string;
  sort_dir?: string;
}
