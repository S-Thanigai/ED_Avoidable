import { useEffect, useState } from "react";
import type { FinalUC07Decision, MemberExplanationResponse } from "../types";
import { explainMember, UC07ApiError } from "../api";
import "./AiExplanationSection.css";

/** Cosmetic-only formatting of the raw Ollama model id ("qwen3:8b") into
 * a human-friendly label ("Qwen3 8B") for display -- never changes which
 * model was actually used, only how its name is shown. */
function formatModelName(modelId: string): string {
  const [name, size] = modelId.split(":");
  if (!name) return modelId;
  const prettyName = name.charAt(0).toUpperCase() + name.slice(1);
  return size ? `${prettyName} ${size.toUpperCase()}` : prettyName;
}

/**
 * "AI EXPLANATION" (Phase 8C Part 13). Lazy, on-demand, single-member
 * only (Part 14): fires exactly one POST /uc07/explain when THIS member
 * is opened (member_id changes) -- never for a whole uploaded
 * population, never repeated on unrelated re-renders. Purely renders
 * whatever backend/agents/genai_explanation.py already decided:
 *   - if explanation_source is "GENAI", labels the text as AI-generated
 *   - if "DETERMINISTIC_FALLBACK" (GenAI disabled, Ollama unreachable,
 *     timed out, or its output failed a policy/consistency check),
 *     labels it as a deterministic system explanation instead
 * This component has no decision logic of its own -- it never computes
 * a risk/navigation/safety value, and never displays anything the
 * backend did not return (no hidden reasoning/thinking is ever part of
 * the response to begin with).
 */
export function AiExplanationSection({ decision }: { decision: FinalUC07Decision }) {
  const [explanation, setExplanation] = useState<MemberExplanationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Re-fetches when the member changes OR when this member's safety
  // state changes (e.g. the user used "Evaluate Current Safety" in the
  // same drawer session) -- otherwise a stale AI explanation could keep
  // describing an old CLEAR/CAUTION/OVERRIDE state after the
  // authoritative SafetyCard above it has already updated to a new one.
  const refetchKey = `${decision.member_id}|${decision.safety.state}|${decision.safety.context_completeness}`;

  useEffect(() => {
    let cancelled = false;
    setExplanation(null);
    setError(null);
    setLoading(true);

    explainMember(decision)
      .then((result) => {
        if (!cancelled) setExplanation(result);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof UC07ApiError || err instanceof Error ? err.message : "Could not generate an explanation.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refetchKey]);

  return (
    <section className="ai-explanation" aria-label="AI explanation">
      <h3 className="ai-explanation__heading">AI Explanation</h3>

      {loading && (
        <p className="ai-explanation__loading" role="status">
          Generating explanation…
        </p>
      )}

      {!loading && error && (
        <p className="ai-explanation__error" role="alert">
          {error}
        </p>
      )}

      {!loading && !error && explanation && (
        <>
          <p className="ai-explanation__summary">{explanation.summary}</p>
          <dl className="ai-explanation__breakdown">
            <div>
              <dt>Risk</dt>
              <dd>{explanation.risk_explanation}</dd>
            </div>
            <div>
              <dt>Navigation</dt>
              <dd>{explanation.navigation_explanation}</dd>
            </div>
            <div>
              <dt>Safety</dt>
              <dd>{explanation.safety_explanation}</dd>
            </div>
          </dl>
          <p className="ai-explanation__disclaimer">{explanation.disclaimer}</p>
          <p className="ai-explanation__source">
            {explanation.explanation_source === "GENAI"
              ? `Source: AI-generated explanation using ${explanation.model_used ? formatModelName(explanation.model_used) : "a local language model"}.`
              : "Explanation source: Deterministic system explanation."}
          </p>
        </>
      )}
    </section>
  );
}
