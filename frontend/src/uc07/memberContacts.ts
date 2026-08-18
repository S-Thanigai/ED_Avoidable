// Member contact/communication store -- Section 1's answer to "does
// member email already exist?": it does NOT. raw_members.csv (and every
// UC07 API response) has no email or name column at all (verified by
// inspection, never invented). Email/name are communication-layer-only
// fields a care manager types in here; they are NEVER sent to
// POST /uc07/decide, never read by any agent, and never become an ML
// feature. This is a plain localStorage map (member_id -> contact),
// scoped entirely to the browser -- there is no backend member/contact
// database in this prototype (see docs/09_MEMBER_COMMUNICATION_REPORTING.md
// "Limitations").
export interface MemberContact {
  name?: string;
  email?: string;
}

const STORAGE_KEY = "uc07.memberContacts.v1";

function readAll(): Record<string, MemberContact> {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    return parsed && typeof parsed === "object" ? (parsed as Record<string, MemberContact>) : {};
  } catch {
    // localStorage unavailable (private browsing, quota, non-browser
    // test environment) -- contacts just won't persist; never throw.
    return {};
  }
}

function writeAll(all: Record<string, MemberContact>): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
  } catch {
    /* see readAll() -- silently a no-op if storage isn't available */
  }
}

export function getMemberContact(memberId: string): MemberContact {
  return readAll()[memberId] ?? {};
}

export function setMemberContact(memberId: string, contact: MemberContact): void {
  const all = readAll();
  all[memberId] = { ...all[memberId], ...contact };
  writeAll(all);
}
