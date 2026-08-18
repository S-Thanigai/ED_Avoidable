import { useEffect, useRef, useState } from "react";
import type { FinalUC07Decision, MemberExplanationResponse } from "../types";
import { buildReportRequest, fetchMemberReportPdf, sendMemberReportEmail, UC07ApiError } from "../api";
import { getCachedExplanation } from "../explanationCache";
import { setMemberContact } from "../memberContacts";
import "./EmailComposerModal.css";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const DEFAULT_SUBJECT = "Your Care Navigation Summary";

function defaultBody(memberLabel: string): string {
  return `Hello ${memberLabel},

Your care-management team has prepared a care-navigation summary based on available healthcare utilization information.

The attached report includes:
- your current care-navigation risk summary
- factors that contributed to the model estimate
- suggested care-navigation support
- applicable safety information

Please note that this report is intended for care-management support and does not replace medical evaluation.

If you are experiencing emergency symptoms or believe you need emergency care, do not delay seeking appropriate emergency evaluation.

Regards,
Care Management Team`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

type SendState = "idle" | "sending" | "sent" | "failed";

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Editable email composer -- Section 2/18/21. Recipient/subject/body are
 * always editable text inputs (no truly "trusted" contact source exists
 * in this prototype -- see memberContacts.ts); nothing is ever sent
 * automatically. Sending requires an explicit two-step confirmation
 * (Section 21): the primary button first shows an inline
 * "Send this report to <address>?" prompt, and only a second, distinct
 * click actually calls the backend.
 */
export function EmailComposerModal({
  decision,
  member,
  onClose,
}: {
  decision: FinalUC07Decision;
  member: { name?: string | null; email?: string | null; age?: number | null; gender?: string | null };
  onClose: () => void;
}) {
  const memberLabel = member.name?.trim() || decision.member_id;
  const [to, setTo] = useState(member.email ?? "");
  const [subject, setSubject] = useState(DEFAULT_SUBJECT);
  const [body, setBody] = useState(defaultBody(memberLabel));
  const [confirming, setConfirming] = useState(false);
  const [sendState, setSendState] = useState<SendState>("idle");
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [attachmentSize, setAttachmentSize] = useState<number | null>(null);

  const panelRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const previewUrlRef = useRef<string | null>(null);

  const sending = sendState === "sending";
  const trimmedTo = to.trim();
  const toIsValid = EMAIL_RE.test(trimmedTo);
  const subjectIsValid = subject.trim().length > 0;
  const bodyIsValid = body.trim().length > 0;
  const canSend = toIsValid && subjectIsValid && bodyIsValid && !sending;

  useEffect(() => {
    closeButtonRef.current?.focus();
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key !== "Tab" || !panelRef.current) return;
      const focusable = Array.from(panelRef.current.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
        (el) => el.offsetParent !== null,
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  function buildRequest() {
    const cachedExplanation: MemberExplanationResponse | null = getCachedExplanation(decision) ?? null;
    return buildReportRequest(decision, member, cachedExplanation);
  }

  async function handlePreview() {
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      const { blob } = await fetchMemberReportPdf(buildRequest());
      setAttachmentSize(blob.size);
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      const url = URL.createObjectURL(blob);
      previewUrlRef.current = url;
      window.open(url, "_blank", "noopener,noreferrer");
    } catch (err) {
      setPreviewError(
        err instanceof UC07ApiError || err instanceof Error ? err.message : "Could not generate a preview.",
      );
    } finally {
      setPreviewLoading(false);
    }
  }

  function handleSendClick() {
    if (!canSend) return;
    if (!confirming) {
      setConfirming(true);
      return;
    }
    void doSend();
  }

  async function doSend() {
    setSendState("sending");
    setStatusMessage(null);
    try {
      const result = await sendMemberReportEmail({
        report: buildRequest(),
        to_email: trimmedTo,
        subject: subject.trim(),
        body,
      });
      if (result.sent) {
        setSendState("sent");
        setStatusMessage(result.message);
        setMemberContact(decision.member_id, { email: trimmedTo, name: member.name ?? undefined });
      } else {
        setSendState("failed");
        setStatusMessage(result.message);
      }
    } catch (err) {
      setSendState("failed");
      setStatusMessage(
        err instanceof UC07ApiError || err instanceof Error ? err.message : "Failed to send the report.",
      );
    } finally {
      setConfirming(false);
    }
  }

  return (
    <div className="email-composer__overlay" onClick={onClose}>
      <div
        ref={panelRef}
        className="email-composer"
        role="dialog"
        aria-modal="true"
        aria-label={`Send care navigation report to ${decision.member_id}`}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="email-composer__header">
          <h2 className="email-composer__title">Send to Member</h2>
          <button
            ref={closeButtonRef}
            type="button"
            className="email-composer__close"
            onClick={onClose}
            aria-label="Close email composer"
          >
            ×
          </button>
        </header>

        <div className="email-composer__body">
          <label className="email-composer__field">
            <span className="email-composer__label">Recipient</span>
            <input
              type="email"
              className="email-composer__input"
              value={to}
              onChange={(e) => {
                setTo(e.target.value);
                setConfirming(false);
              }}
              placeholder="member@example.com"
              disabled={sending}
              aria-invalid={trimmedTo.length > 0 && !toIsValid}
            />
            {trimmedTo.length > 0 && !toIsValid && (
              <span className="email-composer__field-error">Enter a valid email address.</span>
            )}
          </label>

          <label className="email-composer__field">
            <span className="email-composer__label">Subject</span>
            <input
              type="text"
              className="email-composer__input"
              value={subject}
              onChange={(e) => {
                setSubject(e.target.value);
                setConfirming(false);
              }}
              disabled={sending}
            />
          </label>

          <label className="email-composer__field">
            <span className="email-composer__label">Message</span>
            <textarea
              className="email-composer__textarea"
              value={body}
              onChange={(e) => {
                setBody(e.target.value);
                setConfirming(false);
              }}
              rows={10}
              disabled={sending}
            />
          </label>

          <div className="email-composer__attachment">
            <span className="email-composer__attachment-icon" aria-hidden="true">
              📄
            </span>
            <div className="email-composer__attachment-info">
              <span className="email-composer__attachment-name">
                Member_Care_Navigation_Report_{decision.member_id}.pdf
              </span>
              <span className="email-composer__attachment-meta">
                application/pdf{attachmentSize !== null ? ` · ${formatBytes(attachmentSize)}` : ""}
              </span>
            </div>
            <button
              type="button"
              className="email-composer__preview-button"
              onClick={handlePreview}
              disabled={previewLoading || sending}
            >
              {previewLoading ? "Generating…" : "Preview PDF"}
            </button>
          </div>
          {previewError && (
            <p className="email-composer__error" role="alert">
              {previewError}
            </p>
          )}
        </div>

        {sendState === "sent" ? (
          <div className="email-composer__status email-composer__status--success" role="status">
            <strong>Sent successfully.</strong> {statusMessage}
            <div className="email-composer__actions">
              <button type="button" className="email-composer__button" onClick={onClose}>
                Close
              </button>
            </div>
          </div>
        ) : (
          <footer className="email-composer__footer">
            {sendState === "failed" && statusMessage && (
              <p className="email-composer__error" role="alert">
                Failed to send: {statusMessage}
              </p>
            )}

            {confirming ? (
              <div className="email-composer__confirm" role="alertdialog" aria-label="Confirm send">
                <p className="email-composer__confirm-heading">Confirm Send</p>
                <dl className="email-composer__confirm-summary">
                  <div>
                    <dt>Recipient</dt>
                    <dd>{trimmedTo}</dd>
                  </div>
                  <div>
                    <dt>Attachment</dt>
                    <dd>Member_Care_Navigation_Report_{decision.member_id}.pdf</dd>
                  </div>
                </dl>
                <div className="email-composer__actions">
                  <button
                    type="button"
                    className="email-composer__button"
                    onClick={() => setConfirming(false)}
                    disabled={sending}
                  >
                    Back
                  </button>
                  <button
                    type="button"
                    className="email-composer__button email-composer__button--primary"
                    onClick={handleSendClick}
                    disabled={sending}
                  >
                    {sending ? "Sending…" : "Send Email"}
                  </button>
                </div>
              </div>
            ) : (
              <div className="email-composer__actions">
                <button type="button" className="email-composer__button" onClick={onClose} disabled={sending}>
                  Cancel
                </button>
                <button
                  type="button"
                  className="email-composer__button email-composer__button--primary"
                  onClick={handleSendClick}
                  disabled={!canSend}
                >
                  Review &amp; Send
                </button>
              </div>
            )}
          </footer>
        )}
      </div>
    </div>
  );
}
