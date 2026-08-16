import type { FinalUC07Decision } from "../types";
import { RiskCard } from "./RiskCard";
import { SafetyCard } from "./SafetyCard";
import { NavigationCard } from "./NavigationCard";
import { SyntheticDisclosure } from "./SyntheticDisclosure";
import { WhyFlaggedSection } from "./WhyFlaggedSection";
import "./Uc07DecisionPanel.css";

/** Composes one member's FinalUC07Decision. Order is deliberate: Risk
 * (context) -> WHY THIS MEMBER WAS FLAGGED (Phase 8C structured model
 * explainability, directly tied to the risk score above it) -> Safety
 * (gates what navigation may say/show) -> Navigation -- so the Safety
 * Agent's output always sits visually above the Care Navigation Agent's,
 * giving it priority, per docs/08_FRONTEND_INTEGRATION.md section 10. */
export function Uc07DecisionPanel({ decision }: { decision: FinalUC07Decision }) {
  return (
    <div className="uc07-decision-panel">
      <RiskCard risk={decision.risk} />
      <WhyFlaggedSection risk={decision.risk} />
      <SafetyCard safety={decision.safety} />
      <NavigationCard navigation={decision.navigation} safetyState={decision.safety.state} />
      <div className="uc07-decision-panel__footer">
        <SyntheticDisclosure modelVersion={decision.risk.model_version} />
        <p className="uc07-decision-panel__disclaimer">{decision.disclaimer}</p>
      </div>
    </div>
  );
}
