import type { CareVisitRow, EdVisitRow, MemberProfileRow, Uc07MemberDataLookups } from "../uc07/csvUtils";
import type { MemberDetail } from "./types";

/** Converts one DB-backed MemberDetail (backend/db/repositories/
 * populations.py's get_member_detail) into the SAME
 * Uc07MemberDataLookups shape frontend/src/uc07/csvUtils.ts produces by
 * parsing the raw CSVs client-side -- field-for-field, all string-typed
 * to match. This is what lets MemberDataSections / MemberReportActions /
 * MemberDetailsDrawer render a saved population's member identically to
 * a freshly-uploaded one's, with zero changes to those components. */
export function memberDetailToLookups(detail: MemberDetail): Uc07MemberDataLookups {
  const members = new Map<string, MemberProfileRow>();
  if (detail.profile) {
    const p = detail.profile;
    members.set(p.member_id, {
      member_id: p.member_id,
      age: String(p.age),
      gender: p.gender,
      diabetes: String(p.diabetes),
      copd: String(p.copd),
      hypertension: String(p.hypertension),
      chf: String(p.chf),
      asthma: String(p.asthma),
      ckd: String(p.ckd),
      num_chronic_conditions: String(p.num_chronic_conditions),
      transportation_barrier: String(p.transportation_barrier),
      telehealth_available: String(p.telehealth_available),
      pcp_distance_miles: String(p.pcp_distance_miles),
      urgent_care_distance_miles: String(p.urgent_care_distance_miles),
    });
  }

  const edVisitsByMember = new Map<string, EdVisitRow[]>();
  edVisitsByMember.set(
    detail.decision.member_id,
    detail.ed_visits.map((v) => ({
      visit_id: v.visit_id ?? "",
      member_id: v.member_id,
      visit_date: v.visit_date,
      diagnosis: v.diagnosis ?? "",
      triage_level: String(v.triage_level),
      admitted: String(v.admitted),
      icu: String(v.icu),
      major_procedure: String(v.major_procedure),
      cost: String(v.cost),
      red_flag: String(v.red_flag),
    })),
  );

  const careVisitsByMember = new Map<string, CareVisitRow[]>();
  careVisitsByMember.set(
    detail.decision.member_id,
    detail.care_visits.map((v) => ({
      care_id: v.care_id ?? "",
      member_id: v.member_id,
      visit_date: v.visit_date,
      care_type: v.care_type,
    })),
  );

  return { members, edVisitsByMember, careVisitsByMember };
}
