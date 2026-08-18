import { useState } from "react";
import type { FinalUC07Decision } from "../types";
import type { Uc07MemberDataLookups } from "../csvUtils";
import { buildReportRequest, fetchMemberReportPdf, UC07ApiError } from "../api";
import { getCachedExplanation } from "../explanationCache";
import { getMemberContact } from "../memberContacts";
import { EmailComposerModal } from "./EmailComposerModal";
import "./MemberReportActions.css";

function triggerBrowserDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * "Member Communication" tab content (Section 23-25) -- Download PDF
 * Report and Send to Member, presented as a proper care-management
 * workflow section (not two buttons crammed into the workspace
 * header). Both actions build the SAME report request (see api.ts's
 * buildReportRequest); the email composer modal reuses the identical
 * request-building logic for its own attachment, so the downloaded and
 * emailed reports are always the same document.
 */
export function MemberReportActions({
  decision,
  lookups,
}: {
  decision: FinalUC07Decision;
  lookups: Uc07MemberDataLookups | null;
}) {
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [downloadedOnce, setDownloadedOnce] = useState(false);
  const [composerOpen, setComposerOpen] = useState(false);

  const profile = lookups?.members.get(decision.member_id);
  const contact = getMemberContact(decision.member_id);
  const member = {
    name: contact.name ?? null,
    email: contact.email ?? null,
    age: profile?.age ? Number(profile.age) : null,
    gender: profile?.gender ?? null,
  };

  async function handleDownload() {
    setDownloading(true);
    setDownloadError(null);
    try {
      const explanation = getCachedExplanation(decision) ?? null;
      const request = buildReportRequest(decision, member, explanation);
      const { blob, filename } = await fetchMemberReportPdf(request);
      triggerBrowserDownload(blob, filename);
      setDownloadedOnce(true);
    } catch (err) {
      setDownloadError(
        err instanceof UC07ApiError || err instanceof Error ? err.message : "Could not generate the report.",
      );
    } finally {
      setDownloading(false);
    }
  }

  return (
    <section className="member-comm" aria-label="Member communication">
      <div className="member-comm__intro">
        <h3 className="member-comm__heading">Member Communication</h3>
        <p className="member-comm__subtitle">
          Generate the current care-navigation report for this member, or send it directly with an
          editable message.
        </p>
      </div>

      <div className="member-comm__actions">
        <div className="member-comm__action-card">
          <span className="member-comm__action-icon member-comm__action-icon--report" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z" />
              <path d="M14 3v5h5M9 13h6M9 17h6M9 9h1" />
            </svg>
          </span>
          <div className="member-comm__action-body">
            <span className="member-comm__action-title">Generate / Download Member Report</span>
            <p className="member-comm__action-desc">
              Creates the current ED Navigator care-navigation report using the member's existing decision
              information.
            </p>
            {downloadError && (
              <p className="member-comm__action-error" role="alert">
                {downloadError}
              </p>
            )}
            {downloadedOnce && !downloading && !downloadError && (
              <p className="member-comm__action-status">Report downloaded.</p>
            )}
          </div>
          <button
            type="button"
            className="member-comm__button"
            onClick={handleDownload}
            disabled={downloading}
          >
            {downloading ? "Generating…" : "Download Report"}
          </button>
        </div>

        <div className="member-comm__action-card">
          <span className="member-comm__action-icon member-comm__action-icon--email" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="5" width="18" height="14" rx="2" />
              <path d="m3.5 6.5 8 6.2a1.8 1.8 0 0 0 2 0l8-6.2" />
            </svg>
          </span>
          <div className="member-comm__action-body">
            <span className="member-comm__action-title">Send Member Report</span>
            <p className="member-comm__action-desc">
              Opens an editable email with the same report attached as a PDF. Nothing is sent until
              you confirm.
            </p>
          </div>
          <button
            type="button"
            className="member-comm__button member-comm__button--primary"
            onClick={() => setComposerOpen(true)}
          >
            Email Member
          </button>
        </div>
      </div>

      {composerOpen && (
        <EmailComposerModal decision={decision} member={member} onClose={() => setComposerOpen(false)} />
      )}
    </section>
  );
}
