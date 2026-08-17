import { useEffect, useRef, useState } from "react";
import type { UploadFiles } from "../../types";
import type { FinalUC07Decision } from "../types";
import type { Uc07MemberDataLookups } from "../csvUtils";
import { RiskCard } from "./RiskCard";
import { NavigationCard } from "./NavigationCard";
import { SafetyCard } from "./SafetyCard";
import { WhyFlaggedSection } from "./WhyFlaggedSection";
import { MemberDataSections } from "./MemberDataSections";
import { CurrentSafetyContextSection } from "./CurrentSafetyContextSection";
import { AiExplanationSection } from "./AiExplanationSection";
import { SyntheticDisclosure } from "./SyntheticDisclosure";
import "./MemberDetailsDrawer.css";

const TIER_LABEL: Record<FinalUC07Decision["risk"]["tier"], string> = { LOW: "Low", MODERATE: "Moderate", HIGH: "High" };
const SAFETY_LABEL: Record<FinalUC07Decision["safety"]["state"], string> = { CLEAR: "Clear", CAUTION: "Caution", OVERRIDE: "Override" };
const DEST_LABEL: Record<string, string> = {
  PRIMARY_CARE: "Primary Care",
  URGENT_CARE: "Urgent Care",
  TELEHEALTH: "Telehealth",
  CARE_MANAGEMENT: "Care Management",
  NO_PROACTIVE_NAVIGATION: "No proactive navigation",
};

type TabKey = "overview" | "why-flagged" | "ai-explanation" | "current-safety";

const TABS: { key: TabKey; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "why-flagged", label: "Why Flagged" },
  { key: "ai-explanation", label: "AI Explanation" },
  { key: "current-safety", label: "Current Safety" },
];

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** Full analytical workspace for one member's decision -- a large
 * right-side panel on desktop (near-full-screen on mobile), organized
 * into OVERVIEW / WHY FLAGGED / AI EXPLANATION / CURRENT SAFETY tabs
 * rather than one long undifferentiated scroll. Purely a display
 * composition over already-decided data -- closing it does not affect
 * the caller's filter/sort/pagination state (those all live in
 * Uc07View), and switching tabs never recomputes or re-fetches a
 * decision. */
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
  const [activeTab, setActiveTab] = useState<TabKey>("overview");
  const panelRef = useRef<HTMLElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  // Reset to Overview whenever a DIFFERENT member is opened, so the
  // workspace never silently opens on a stale tab from a previous member.
  useEffect(() => {
    setActiveTab("overview");
  }, [decision.member_id]);

  // Escape closes; focus is trapped inside the workspace while open
  // (Tab/Shift+Tab cycle only among its own focusable elements) so
  // keyboard users can never tab out into the page behind it.
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

  const { risk, navigation, safety } = decision;
  const destinationLabel = navigation.destination ? DEST_LABEL[navigation.destination] ?? navigation.destination : "None (override)";

  return (
    <div className="member-workspace__overlay" onClick={onClose}>
      <aside
        ref={panelRef}
        className="member-workspace"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={`Details for member ${decision.member_id}`}
      >
        <header className="member-workspace__header">
          <div className="member-workspace__identity">
            <span className="member-workspace__eyebrow">Member</span>
            <h2 className="member-workspace__title">{decision.member_id}</h2>
          </div>

          <div className="member-workspace__glance" aria-label="Decision summary">
            <div className="member-workspace__glance-item">
              <span className="member-workspace__glance-label">Risk</span>
              <span className={`member-workspace__glance-badge member-workspace__glance-badge--${risk.tier.toLowerCase()}`}>
                {TIER_LABEL[risk.tier]}
              </span>
              <span className="member-workspace__glance-sub tabular">{(risk.probability * 100).toFixed(1)}%</span>
            </div>
            <div className="member-workspace__glance-item">
              <span className="member-workspace__glance-label">Navigation</span>
              <span className="member-workspace__glance-value">{destinationLabel}</span>
            </div>
            <div className="member-workspace__glance-item">
              <span className="member-workspace__glance-label">Safety</span>
              <span className={`member-workspace__glance-badge member-workspace__glance-badge--safety-${safety.state.toLowerCase()}`}>
                {SAFETY_LABEL[safety.state]}
              </span>
            </div>
          </div>

          <button
            ref={closeButtonRef}
            type="button"
            className="member-workspace__close"
            onClick={onClose}
            aria-label="Close member details"
          >
            ×
          </button>
        </header>

        {safety.state === "OVERRIDE" && (
          <div className="member-workspace__override-banner" role="alert">
            <strong>Safety override active.</strong> A configured high-acuity signal was present for this
            encounter — see the Current Safety tab. Emergency care should never be delayed.
          </div>
        )}

        <nav className="member-workspace__tabs" role="tablist" aria-label="Member detail sections">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              type="button"
              role="tab"
              id={`member-workspace-tab-${tab.key}`}
              aria-selected={activeTab === tab.key}
              aria-controls={`member-workspace-panel-${tab.key}`}
              className={`member-workspace__tab${activeTab === tab.key ? " member-workspace__tab--active" : ""}`}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <div className="member-workspace__body">
          <div
            role="tabpanel"
            id="member-workspace-panel-overview"
            aria-labelledby="member-workspace-tab-overview"
            hidden={activeTab !== "overview"}
            className="member-workspace__panel"
          >
            <div className="member-workspace__concept-grid">
              <RiskCard risk={risk} />
              <SafetyCard safety={safety} />
              <NavigationCard navigation={navigation} safetyState={safety.state} />
            </div>
            <MemberDataSections memberId={decision.member_id} lookups={lookups} loading={lookupsLoading} />
            <div className="member-workspace__footer">
              <SyntheticDisclosure modelVersion={risk.model_version} />
              <p className="member-workspace__disclaimer">{decision.disclaimer}</p>
            </div>
          </div>

          <div
            role="tabpanel"
            id="member-workspace-panel-why-flagged"
            aria-labelledby="member-workspace-tab-why-flagged"
            hidden={activeTab !== "why-flagged"}
            className="member-workspace__panel"
          >
            <WhyFlaggedSection risk={risk} />
          </div>

          <div
            role="tabpanel"
            id="member-workspace-panel-ai-explanation"
            aria-labelledby="member-workspace-tab-ai-explanation"
            hidden={activeTab !== "ai-explanation"}
            className="member-workspace__panel"
          >
            {activeTab === "ai-explanation" && <AiExplanationSection decision={decision} />}
          </div>

          <div
            role="tabpanel"
            id="member-workspace-panel-current-safety"
            aria-labelledby="member-workspace-tab-current-safety"
            hidden={activeTab !== "current-safety"}
            className="member-workspace__panel"
          >
            <CurrentSafetyContextSection
              decision={decision}
              files={files}
              indexDate={indexDate}
              onEvaluated={onSafetyEvaluated}
            />
          </div>
        </div>
      </aside>
    </div>
  );
}
