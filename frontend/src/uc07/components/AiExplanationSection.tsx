import { useEffect, useState } from "react";
import type { FinalUC07Decision, MemberExplanationResponse } from "../types";
import { explainMember, UC07ApiError } from "../api";
import { buildExplanationCacheKey, getCachedExplanation, setCachedExplanation } from "../explanationCache";
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
 * "AI EXPLANATION" (Phase 8C Part 13, session-cached as of this phase).
 * Lazy, on-demand, single-member only: fires at most one
 * POST /uc07/explain per DISTINCT decision (see explanationCache.ts for
 * exactly what "distinct" means -- member_id alone is not enough, the
 * key also covers tier/probability/navigation/safety/model_version).
 * Reopening the same member with the same decision shows the previously
 * generated explanation immediately, with no loading state and no new
 * network call. If the underlying decision changes (e.g. a safety
 * re-evaluation moves CAUTION -> OVERRIDE, or a fresh /uc07/decide run
 * changes risk/navigation), the cache key changes and a fresh
 * explanation is fetched -- the old text is never shown as though it
 * describes the new decision.
 *
 * This component has no decision logic of its own -- it never computes
 * a risk/navigation/safety value, and never displays anything the
 * backend did not return (no hidden reasoning/thinking is ever part of
 * the response to begin with). Caching is purely a frontend performance
 * optimization over an unmodified backend contract; /uc07/explain's
 * request/response shape and the GenAI Agent's own logic are untouched.
 */
export function AiExplanationSection({ decision }: { decision: FinalUC07Decision }) {
  const cacheKey = buildExplanationCacheKey(decision);
  const cached = getCachedExplanation(decision);

  const [explanation, setExplanation] = useState<MemberExplanationResponse | null>(cached ?? null);
  const [loading, setLoading] = useState(!cached);
  const [error, setError] = useState<string | null>(null);
  const [fromCache, setFromCache] = useState(Boolean(cached));

  useEffect(() => {
    const alreadyCached = getCachedExplanation(decision);
    if (alreadyCached) {
      // Instant, no request, no loading flash -- Part 20's explicit
      // requirement for a cache hit.
      setExplanation(alreadyCached);
      setError(null);
      setLoading(false);
      setFromCache(true);
      return;
    }

    let cancelled = false;
    setExplanation(null);
    setError(null);
    setLoading(true);
    setFromCache(false);

    explainMember(decision)
      .then((result) => {
        if (cancelled) return;
        setCachedExplanation(decision, result);
        setExplanation(result);
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
    // cacheKey encodes every field that would change what /uc07/explain
    // returns (see explanationCache.ts) -- it, not `decision` itself
    // (a new object reference on every parent render), is the correct
    // effect dependency.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cacheKey]);

  return (
    <section className="ai-explanation" aria-label="AI explanation">
      <div className="ai-explanation__heading-row">
        <span className="ai-explanation__icon" aria-hidden="true">
          ✦
        </span>
        <div>
          <h3 className="ai-explanation__heading">AI Explanation</h3>
          <p className="ai-explanation__subtitle">
            Generated from the existing model/agent decision. AI does not make or change the decision.
          </p>
        </div>
      </div>

      {loading && (
        <div className="ai-explanation__loading" role="status">
          <span className="ai-explanation__skeleton" />
          <span className="ai-explanation__skeleton ai-explanation__skeleton--short" />
          <span className="ai-explanation__loading-text">Generating explanation…</span>
        </div>
      )}

      {!loading && error && (
        <p className="ai-explanation__error" role="alert">
          AI explanation unavailable: {error}
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
              ? `AI-generated explanation using ${explanation.model_used ? formatModelName(explanation.model_used) : "a local language model"}`
              : "Deterministic fallback explanation (AI generation unavailable or filtered)"}
            {fromCache && <span className="ai-explanation__cache-note"> · Previously generated this session</span>}
          </p>
        </>
      )}
    </section>
  );
}
