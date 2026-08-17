// In-memory, session-scoped cache for GenAI/deterministic explanations
// (POST /uc07/explain results) -- exists purely to avoid re-calling
// Ollama when the user closes and reopens the SAME member with the SAME
// underlying decision. No backend persistence, no localStorage: a plain
// module-level Map, which lives exactly as long as this browser tab's JS
// session (cleared on refresh -- explicitly acceptable, see Part 2 spec).
//
// The cache key is derived from the fields that actually determine what
// /uc07/explain would return -- NOT just member_id. If any of these
// change (e.g. "Evaluate Current Safety" moves CAUTION -> OVERRIDE, or a
// fresh /uc07/decide run changes risk/navigation), the key changes, the
// old cache entry is never matched, and a fresh explanation is fetched.
// This is what "cache invalidation" means here: there is no explicit
// eviction step, just a key that naturally stops matching once the
// decision it described no longer exists.
import type { FinalUC07Decision, MemberExplanationResponse } from "./types";

export function buildExplanationCacheKey(decision: FinalUC07Decision): string {
  const { risk, navigation, safety } = decision;
  return [
    decision.member_id,
    risk.model_version,
    risk.tier,
    risk.probability,
    navigation.destination ?? "NONE",
    safety.state,
    safety.context_completeness,
    safety.context_source,
  ].join("|");
}

const cache = new Map<string, MemberExplanationResponse>();

export function getCachedExplanation(decision: FinalUC07Decision): MemberExplanationResponse | undefined {
  return cache.get(buildExplanationCacheKey(decision));
}

export function setCachedExplanation(decision: FinalUC07Decision, explanation: MemberExplanationResponse): void {
  cache.set(buildExplanationCacheKey(decision), explanation);
}

/** Test-only reset -- a real user session never needs to call this; the
 * cache simply stops being consulted once the page is reloaded. */
export function clearExplanationCache(): void {
  cache.clear();
}
