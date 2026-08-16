"""
model_explainability.py
--------------------------
Phase 8C -- authoritative, member-level model explainability for the
Risk Detection Agent's frozen `uc07-risk-synthetic-v1` (Logistic
Regression) pipeline.

WHY SHAP LinearExplainer, not a bare coefficient x value shortcut
-------------------------------------------------------------------
Phase 8C explicitly requires NOT assuming a raw
`coefficient_i * standardized_value_i` decomposition is automatically
identical to SHAP. That assumption is only exact under a feature-
INDEPENDENCE assumption; UC07's 59-feature set has real, documented
correlations (e.g. `access_burden` is partly derived from
`transportation_barrier` and the distance features, r~0.70; the
`prior_ed_count_{30,90,180,270}d` windows are nested/cumulative and
strongly correlated with each other -- see docs/07_DISPARITY_INPUT_SAFETY_HARDENING.md
section 4). This was verified empirically (not assumed) before writing
this module: on 50 real TEST-snapshot rows, `shap.LinearExplainer` with
an INDEPENDENT masker matched the plain coefficient x value computation
closely (mean abs diff 0.0043), but with a CORRELATION-AWARE masker
(`shap.maskers.Impute`, which properly reallocates credit among
correlated features) the values diverge materially (mean abs diff
0.0185, max abs diff 0.377 -- roughly 4x/13x larger). That divergence IS
the point of using real SHAP here: it is a more statistically grounded
attribution than the independence shortcut, for a dataset that
genuinely has correlated features.

`shap.LinearExplainer` is technically appropriate for this exact case:
a saved sklearn ColumnTransformer + LogisticRegression pipeline is
precisely what LinearExplainer is designed for (unlike TreeExplainer,
used by the separate legacy `/explain-member` path for the old
RandomForest model -- see docs/08C section 4 for why that explainer
does not apply here).

Performance: building the SHAP explainer (estimating the masker's
imputation transform, nsamples=1000) takes ~5s -- a ONE-TIME cost, done
lazily on first use and cached at module level (keyed by model artifact
path) so it is never repeated per-request or per-agent-instance. Once
built, computing SHAP values for one row is sub-millisecond; for the
full 10,000-member synthetic population it is ~13ms total (measured).
This is why explanation_factors are computed EAGERLY for every member
in RiskDetectionAgent.assess()/assess_batch() -- unlike the GenAI
Explanation Agent (backend/agents/genai_explanation.py), there is no
"unusably slow population upload" concern for SHAP.

Fallback: if SHAP computation fails for ANY reason (import error,
explainer-build failure, an incompatible non-linear model in the
future, or any other exception), this module falls back to the exact,
deterministic `coefficient * standardized_value` decomposition
(mathematically exact for this model's log-odds; NOT a fake/invented
value) and marks every factor's `explanation_method` as
LINEAR_CONTRIBUTION so a caller always knows which method actually
produced what it is looking at (Phase 8C Part 3: "never fabricate SHAP
values").

Never causal: every factor exposed here describes a signed contribution
to the model's OWN score, not a real-world cause of a future ED visit.
"""
from __future__ import annotations

import contextlib
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from contracts import ExplanationFactor, ExplanationMethod, FactorDirection

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKGROUND_SNAPSHOT_PATH = REPO_ROOT / "data" / "derived" / "synthetic" / "train_snapshot.csv"
BACKGROUND_SAMPLE_SIZE = 1000
SHAP_NSAMPLES = 1000
RANDOM_STATE = 42

MAX_INCREASING_FACTORS = 3
MAX_DECREASING_FACTORS = 2

# Module-level cache: one built SHAP explainer per distinct model
# artifact, shared across every RiskDetectionAgent instance in this
# process (constructing a fresh agent -- as many tests do -- must not
# re-pay the ~5s one-time build cost every time).
_explainer_cache: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Feature name -> human-readable display name (generic humanizer, not a
# hand-maintained per-feature list -- stays correct automatically if the
# approved feature manifest ever changes).
# ---------------------------------------------------------------------------

_ACRONYMS = {"ed", "pcp", "ckd", "chf", "copd", "icu"}
_WINDOW_RE = re.compile(r"^(\d+)d$")


def _humanize_token(token: str) -> str:
    m = _WINDOW_RE.match(token)
    if m:
        return f"({m.group(1)} days)"
    if token in _ACRONYMS:
        return token.upper()
    return token


def humanize_feature_name(feature_name: str) -> str:
    """'prior_potentially_avoidable_ed_count_270d' ->
    'Prior potentially avoidable ED count (270 days)'. Deterministic,
    generic string formatting only -- never a claim about the feature's
    real-world meaning beyond its name."""
    tokens = [_humanize_token(t) for t in feature_name.split("_") if t]
    text = " ".join(tokens)
    return text[:1].upper() + text[1:] if text else feature_name


# ---------------------------------------------------------------------------
# Deterministic linear-contribution fallback (exact for this model's
# log-odds; moved here from risk_detection.py in Phase 8C so both the
# legacy sentence-based factors AND the new structured SHAP fallback
# share one implementation, never duplicated).
# ---------------------------------------------------------------------------

def linear_contributions(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """contribution_i = coefficient_i * standardized_value_i -- exact for
    logistic regression's log-odds (sum of contributions + intercept =
    logit). Returns (None, None) for any non-linear final model (never
    fabricates an explanation it cannot support). Vectorized over all
    rows of X at once."""
    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")
    if model is None or preprocessor is None or not hasattr(model, "coef_"):
        return None, None

    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
    contributions = transformed * model.coef_[0]
    return contributions, feature_names


# ---------------------------------------------------------------------------
# SHAP LinearExplainer (primary method)
# ---------------------------------------------------------------------------

def _build_shap_explainer(pipeline, feature_columns: list[str], cache_key: str):
    if cache_key in _explainer_cache:
        return _explainer_cache[cache_key]

    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocessor")
    if model is None or preprocessor is None or not hasattr(model, "coef_"):
        _explainer_cache[cache_key] = None
        return None

    try:
        import shap

        if not BACKGROUND_SNAPSHOT_PATH.exists():
            _explainer_cache[cache_key] = None
            return None
        background_df = pd.read_csv(BACKGROUND_SNAPSHOT_PATH)[feature_columns]
        n = min(BACKGROUND_SAMPLE_SIZE, len(background_df))
        background_df = background_df.sample(n=n, random_state=RANDOM_STATE)
        background_transformed = preprocessor.transform(background_df)
        if hasattr(background_transformed, "toarray"):
            background_transformed = background_transformed.toarray()

        # shap.LinearExplainer's `seed=` parameter reseeds the GLOBAL
        # numpy.random state (via np.random.seed) for its own nsamples
        # draws and does not restore it afterward -- left unguarded, this
        # silently changes global RNG state for the rest of the process,
        # which can perturb any LATER unrelated code that relies on
        # unseeded global `np.random` calls (empirically confirmed: it
        # was observed to change a non-semantic serialization detail of
        # backend/modeling/train.py's pickled artifact when both ran in
        # the same pytest process -- the model's actual learned
        # coefficients/thresholds were verified bit-identical either way,
        # but the byte-level artifact differed, which is exactly the kind
        # of process-global side effect a self-contained explainer-build
        # function must not leak). Save and restore the global state
        # around this call so building a SHAP explainer here can never
        # affect anything else running in the same process.
        rng_state = np.random.get_state()
        try:
            with warnings.catch_warnings(), open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
                # shap.maskers.Impute also prints a tqdm progress bar to
                # stderr while estimating its imputation transform --
                # harmless but noisy in server logs, so it is suppressed
                # for this one-time, ~5-10s build step only (never for
                # the fast per-row shap_values() calls below).
                warnings.simplefilter("ignore")
                explainer = shap.LinearExplainer(
                    model,
                    shap.maskers.Impute(background_transformed),
                    nsamples=SHAP_NSAMPLES,
                    seed=RANDOM_STATE,
                )
        finally:
            np.random.set_state(rng_state)
    except Exception:
        explainer = None

    _explainer_cache[cache_key] = explainer
    return explainer


def shap_contributions(pipeline, feature_columns: list[str], X: pd.DataFrame, cache_key: str) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """Returns (shap_values, feature_names) for every row of X, or
    (None, None) if SHAP is not applicable/available/buildable. Never
    raises -- any failure is treated as "SHAP unavailable" so the caller
    falls back to linear_contributions()."""
    explainer = _build_shap_explainer(pipeline, feature_columns, cache_key)
    if explainer is None:
        return None, None
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        transformed = preprocessor.transform(X)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        # LinearExplainer.shap_values() also draws from the masker's
        # nsamples-based estimate at call time, not only at build time --
        # same global-RNG-leak concern as _build_shap_explainer above, so
        # the same save/restore guard applies here too.
        rng_state = np.random.get_state()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values = explainer.shap_values(transformed)
        finally:
            np.random.set_state(rng_state)
        feature_names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
        return np.asarray(values), feature_names
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Structured factor ranking (shared by both methods)
# ---------------------------------------------------------------------------

def _rank_factors(contributions: np.ndarray, feature_names: list[str], method: ExplanationMethod) -> list[ExplanationFactor]:
    increasing: list[ExplanationFactor] = []
    decreasing: list[ExplanationFactor] = []
    order = np.argsort(np.abs(contributions))[::-1]
    for idx in order:
        value = float(contributions[idx])
        if abs(value) < 1e-9:
            continue
        feature = feature_names[idx]
        factor = ExplanationFactor(
            feature=feature,
            display_name=humanize_feature_name(feature),
            direction=FactorDirection.INCREASES_RISK if value > 0 else FactorDirection.DECREASES_RISK,
            contribution=round(value, 6),
            explanation_method=method,
        )
        if value > 0 and len(increasing) < MAX_INCREASING_FACTORS:
            increasing.append(factor)
        elif value < 0 and len(decreasing) < MAX_DECREASING_FACTORS:
            decreasing.append(factor)
        if len(increasing) >= MAX_INCREASING_FACTORS and len(decreasing) >= MAX_DECREASING_FACTORS:
            break
    return increasing + decreasing


def explain_row(pipeline, feature_columns: list[str], features_row: pd.DataFrame, cache_key: str) -> tuple[list[ExplanationFactor], ExplanationMethod]:
    """Authoritative single-member model explanation. Tries SHAP first;
    on ANY failure, falls back to the exact deterministic linear
    contribution -- never returns a fabricated/fake value, and never
    raises (a failure here must not break /uc07/decide)."""
    try:
        shap_values, feature_names = shap_contributions(pipeline, feature_columns, features_row, cache_key)
        if shap_values is not None:
            row_values = shap_values[0]
            factors = _rank_factors(row_values, feature_names, ExplanationMethod.SHAP_LINEAR)
            return factors, ExplanationMethod.SHAP_LINEAR
    except Exception:
        pass

    try:
        contributions, feature_names = linear_contributions(pipeline, features_row)
        if contributions is not None:
            factors = _rank_factors(contributions[0], feature_names, ExplanationMethod.LINEAR_CONTRIBUTION)
            return factors, ExplanationMethod.LINEAR_CONTRIBUTION
    except Exception:
        pass

    return [], ExplanationMethod.LINEAR_CONTRIBUTION


def explain_batch(pipeline, feature_columns: list[str], features: pd.DataFrame, cache_key: str) -> tuple[list[list[ExplanationFactor]], ExplanationMethod]:
    """Vectorized equivalent of calling explain_row() once per member --
    one SHAP/linear-contribution matrix for the whole batch. All rows in
    a batch share the SAME method (SHAP for the whole batch, or linear
    contribution for the whole batch, whichever succeeds) so a caller
    never sees a population half-explained by one method and half by
    another within a single request."""
    try:
        shap_values, feature_names = shap_contributions(pipeline, feature_columns, features, cache_key)
        if shap_values is not None:
            rows = [_rank_factors(shap_values[i], feature_names, ExplanationMethod.SHAP_LINEAR) for i in range(len(features))]
            return rows, ExplanationMethod.SHAP_LINEAR
    except Exception:
        pass

    try:
        contributions, feature_names = linear_contributions(pipeline, features)
        if contributions is not None:
            rows = [_rank_factors(contributions[i], feature_names, ExplanationMethod.LINEAR_CONTRIBUTION) for i in range(len(features))]
            return rows, ExplanationMethod.LINEAR_CONTRIBUTION
    except Exception:
        pass

    return [[] for _ in range(len(features))], ExplanationMethod.LINEAR_CONTRIBUTION
