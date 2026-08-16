import type { Uc07MemberDataLookups } from "../csvUtils";
import "./MemberDataSections.css";

const CONDITION_LABELS: { key: "diabetes" | "copd" | "hypertension" | "chf" | "asthma" | "ckd"; label: string }[] = [
  { key: "diabetes", label: "Diabetes" },
  { key: "copd", label: "COPD" },
  { key: "hypertension", label: "Hypertension" },
  { key: "chf", label: "CHF" },
  { key: "asthma", label: "Asthma" },
  { key: "ckd", label: "CKD" },
];

const CARE_TYPES = ["PCP", "Telehealth", "Urgent Care", "Care Management"] as const;

function yesNo(raw: string | undefined): string {
  if (raw === undefined || raw === "") return "—";
  return raw === "1" ? "Yes" : "No";
}

/** Renders the member profile / chronic conditions / ED utilization /
 * access barriers / care history sections from the same three CSV files
 * already uploaded for this decision -- read client-side (see
 * csvUtils.ts), never fabricated, never sent to or re-derived by any
 * model. If a member's row isn't present in the uploaded data, each
 * section says so plainly rather than guessing. */
export function MemberDataSections({
  memberId,
  lookups,
  loading,
}: {
  memberId: string;
  lookups: Uc07MemberDataLookups | null;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="member-data-sections">
        <h3 className="member-data-sections__heading">Why this member was flagged</h3>
        <p className="member-data-sections__status">Loading member data from the uploaded files…</p>
      </div>
    );
  }

  if (!lookups) {
    return (
      <div className="member-data-sections">
        <h3 className="member-data-sections__heading">Why this member was flagged</h3>
        <p className="member-data-sections__status">
          Additional member data is not available (original upload files are no longer in memory
          for this session).
        </p>
      </div>
    );
  }

  const profile = lookups.members.get(memberId);
  const edVisits = lookups.edVisitsByMember.get(memberId) ?? [];
  const careVisits = lookups.careVisitsByMember.get(memberId) ?? [];

  return (
    <div className="member-data-sections">
      <h3 className="member-data-sections__heading">Why this member was flagged</h3>

      <section className="member-data-sections__block">
        <h4>Member profile</h4>
        {profile ? (
          <dl className="member-data-sections__grid">
            <div>
              <dt>Age</dt>
              <dd>{profile.age}</dd>
            </div>
            <div>
              <dt>Gender</dt>
              <dd>{profile.gender}</dd>
            </div>
            <div>
              <dt>Chronic conditions</dt>
              <dd>{profile.num_chronic_conditions}</dd>
            </div>
          </dl>
        ) : (
          <p className="member-data-sections__status">No member profile row found in uploaded data.</p>
        )}
      </section>

      <section className="member-data-sections__block">
        <h4>Chronic conditions</h4>
        {profile ? (
          (() => {
            const present = CONDITION_LABELS.filter((c) => profile[c.key] === "1");
            return present.length > 0 ? (
              <div className="member-data-sections__chips">
                {present.map((c) => (
                  <span className="member-data-sections__chip" key={c.key}>
                    {c.label}
                  </span>
                ))}
              </div>
            ) : (
              <p className="member-data-sections__status">No chronic conditions on file.</p>
            );
          })()
        ) : (
          <p className="member-data-sections__status">No member profile row found in uploaded data.</p>
        )}
      </section>

      <section className="member-data-sections__block">
        <h4>
          ED visit history <span className="member-data-sections__count">({edVisits.length} from uploaded data)</span>
        </h4>
        {edVisits.length > 0 ? (
          <div className="member-data-sections__table-wrap">
            <table className="member-data-sections__table">
              <thead>
                <tr>
                  <th scope="col">Date</th>
                  <th scope="col">Diagnosis</th>
                  <th scope="col">Triage</th>
                  <th scope="col">Admitted</th>
                  <th scope="col">Red flag</th>
                </tr>
              </thead>
              <tbody>
                {edVisits.slice(0, 8).map((v, i) => (
                  <tr key={`${v.visit_id || v.visit_date}-${i}`}>
                    <td>{v.visit_date}</td>
                    <td>{v.diagnosis || "—"}</td>
                    <td>{v.triage_level || "—"}</td>
                    <td>{yesNo(v.admitted)}</td>
                    <td>{yesNo(v.red_flag)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {edVisits.length > 8 && (
              <p className="member-data-sections__more">+{edVisits.length - 8} earlier visit(s) not shown</p>
            )}
          </div>
        ) : (
          <p className="member-data-sections__status">No ED visit history in uploaded data.</p>
        )}
      </section>

      <section className="member-data-sections__block">
        <h4>Access barriers</h4>
        {profile ? (
          <dl className="member-data-sections__grid">
            <div>
              <dt>Transportation barrier</dt>
              <dd>{yesNo(profile.transportation_barrier)}</dd>
            </div>
            <div>
              <dt>Telehealth available</dt>
              <dd>{yesNo(profile.telehealth_available)}</dd>
            </div>
            <div>
              <dt>PCP distance</dt>
              <dd>{profile.pcp_distance_miles ? `${profile.pcp_distance_miles} mi` : "—"}</dd>
            </div>
            <div>
              <dt>Urgent care distance</dt>
              <dd>{profile.urgent_care_distance_miles ? `${profile.urgent_care_distance_miles} mi` : "—"}</dd>
            </div>
          </dl>
        ) : (
          <p className="member-data-sections__status">No member profile row found in uploaded data.</p>
        )}
      </section>

      <section className="member-data-sections__block">
        <h4>Care history</h4>
        <dl className="member-data-sections__grid">
          {CARE_TYPES.map((type) => {
            const visits = careVisits.filter((v) => v.care_type === type);
            return (
              <div key={type}>
                <dt>{type}</dt>
                <dd>
                  {visits.length > 0 ? (
                    <>
                      {visits.length} visit{visits.length === 1 ? "" : "s"}
                      <span className="member-data-sections__muted"> · most recent {visits[0].visit_date}</span>
                    </>
                  ) : (
                    "No visits on file"
                  )}
                </dd>
              </div>
            );
          })}
        </dl>
      </section>
    </div>
  );
}
