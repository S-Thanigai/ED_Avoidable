// Client-side parsing of the same three CSV files the user already
// uploaded for POST /uc07/decide -- purely presentational (populates the
// member details drawer's profile/utilization/access/care-history
// sections). Never sent anywhere, never influences a decision, and
// never invents a value the file doesn't contain.
//
// This exists because /uc07/decide's response (FinalUC07Decision) only
// carries the risk/navigation/safety decision, not the underlying raw
// member/ED/care rows -- and per the Phase 7/8 "reuse data already
// produced" principle, those rows are already sitting in memory as the
// File objects the user just uploaded, so re-reading them here avoids
// any backend/API change.

export interface MemberProfileRow {
  member_id: string;
  age: string;
  gender: string;
  diabetes: string;
  copd: string;
  hypertension: string;
  chf: string;
  asthma: string;
  ckd: string;
  num_chronic_conditions: string;
  transportation_barrier: string;
  telehealth_available: string;
  pcp_distance_miles: string;
  urgent_care_distance_miles: string;
}

export interface EdVisitRow {
  visit_id: string;
  member_id: string;
  visit_date: string;
  diagnosis: string;
  triage_level: string;
  admitted: string;
  icu: string;
  major_procedure: string;
  cost: string;
  red_flag: string;
}

export interface CareVisitRow {
  care_id: string;
  member_id: string;
  visit_date: string;
  care_type: string;
}

export interface Uc07MemberDataLookups {
  members: Map<string, MemberProfileRow>;
  edVisitsByMember: Map<string, EdVisitRow[]>;
  careVisitsByMember: Map<string, CareVisitRow[]>;
}

/** Minimal RFC4180-ish CSV line splitter -- handles quoted fields with
 * embedded commas/quotes, which the simple synthetic datasets don't
 * currently use, but a real upload might. */
function parseCsvLine(line: string): string[] {
  const fields: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (inQuotes) {
      if (char === '"') {
        if (line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        current += char;
      }
    } else if (char === '"') {
      inQuotes = true;
    } else if (char === ",") {
      fields.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  fields.push(current);
  return fields;
}

function parseCsvText(text: string): Record<string, string>[] {
  const lines = text.split(/\r\n|\n/).filter((line) => line.length > 0);
  if (lines.length === 0) return [];
  const headers = parseCsvLine(lines[0]);
  const rows: Record<string, string>[] = [];
  for (let i = 1; i < lines.length; i++) {
    const fields = parseCsvLine(lines[i]);
    const row: Record<string, string> = {};
    headers.forEach((header, idx) => {
      row[header] = fields[idx] ?? "";
    });
    rows.push(row);
  }
  return rows;
}

export async function readAndParseUc07Files(files: {
  members: File | null;
  edVisits: File | null;
  care: File | null;
}): Promise<Uc07MemberDataLookups> {
  const [membersText, edText, careText] = await Promise.all([
    files.members?.text() ?? Promise.resolve(""),
    files.edVisits?.text() ?? Promise.resolve(""),
    files.care?.text() ?? Promise.resolve(""),
  ]);

  const members = new Map<string, MemberProfileRow>();
  for (const row of parseCsvText(membersText) as unknown as MemberProfileRow[]) {
    if (row.member_id) members.set(row.member_id, row);
  }

  const edVisitsByMember = new Map<string, EdVisitRow[]>();
  for (const row of parseCsvText(edText) as unknown as EdVisitRow[]) {
    if (!row.member_id) continue;
    const list = edVisitsByMember.get(row.member_id) ?? [];
    list.push(row);
    edVisitsByMember.set(row.member_id, list);
  }
  for (const list of edVisitsByMember.values()) {
    list.sort((a, b) => (a.visit_date < b.visit_date ? 1 : -1)); // most recent first
  }

  const careVisitsByMember = new Map<string, CareVisitRow[]>();
  for (const row of parseCsvText(careText) as unknown as CareVisitRow[]) {
    if (!row.member_id) continue;
    const list = careVisitsByMember.get(row.member_id) ?? [];
    list.push(row);
    careVisitsByMember.set(row.member_id, list);
  }
  for (const list of careVisitsByMember.values()) {
    list.sort((a, b) => (a.visit_date < b.visit_date ? 1 : -1));
  }

  return { members, edVisitsByMember, careVisitsByMember };
}
