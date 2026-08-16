import { useEffect } from "react";
import type { UploadFiles } from "../../types";
import type { FinalUC07Decision } from "../types";
import type { Uc07MemberDataLookups } from "../csvUtils";
import { Uc07DecisionPanel } from "./Uc07DecisionPanel";
import { MemberDataSections } from "./MemberDataSections";
import { CurrentSafetyContextSection } from "./CurrentSafetyContextSection";
import { AiExplanationSection } from "./AiExplanationSection";
import "./MemberDetailsDrawer.css";

/** Right-side slide-over showing one member's full decision + the raw
 * profile/utilization/access/care-history data behind it, plus the
 * "Current Safety Context" evaluator. Purely a display composition --
 * closing it does not affect the caller's filter/sort/pagination state
 * (those all live in Uc07View, unaffected by which drawer is open). */
export function MemberDetailsDrawer({
  decision,
  lookups,
  lookupsLoading,
  files,
  indexDate,
  onSafetyEvaluated,
  onClose,
}: {
  decision: FinalUC07Decision;
  lookups: Uc07MemberDataLookups | null;
  lookupsLoading: boolean;
  files: UploadFiles;
  indexDate: string;
  onSafetyEvaluated: (updated: FinalUC07Decision) => void;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="member-details-drawer__overlay" onClick={onClose}>
      <aside
        className="member-details-drawer"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={`Details for member ${decision.member_id}`}
      >
        <div className="member-details-drawer__header">
          <div>
            <span className="member-details-drawer__eyebrow">Member</span>
            <h2 className="member-details-drawer__title">{decision.member_id}</h2>
          </div>
          <button type="button" className="member-details-drawer__close" onClick={onClose} aria-label="Close member details">
            ×
          </button>
        </div>

        <div className="member-details-drawer__body">
          <Uc07DecisionPanel decision={decision} />
          <MemberDataSections memberId={decision.member_id} lookups={lookups} loading={lookupsLoading} />
          <CurrentSafetyContextSection
            decision={decision}
            files={files}
            indexDate={indexDate}
            onEvaluated={onSafetyEvaluated}
          />
          <AiExplanationSection decision={decision} />
        </div>
      </aside>
    </div>
  );
}
