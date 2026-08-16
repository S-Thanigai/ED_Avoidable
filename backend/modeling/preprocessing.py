"""
preprocessing.py
-----------------
Builds the sklearn ColumnTransformer used by every Phase 4 candidate
model. Fit ONLY on TRAIN by every caller in this package -- this module
itself is stateless (it returns an unfit transformer); the fitting
discipline lives in train.py's control flow, not here.

Missing-value strategy:
  - numeric features: median imputation. The only numeric features with
    real missingness are the Phase 3 `days_since_prior_*` recency columns
    (explicit NaN when no qualifying prior event exists in the 270-day
    observation window -- see docs/03_ML_DATA_PIPELINE.md section 8/17).
    Their companion `has_prior_*` 0/1 flags are already an explicit
    presence indicator, so median-imputing the recency value itself
    (rather than inventing a large sentinel) is a safe, simple choice for
    tree- and linear-model-friendly numeric input.
  - categorical features (`gender` only, in the current feature set):
    most-frequent imputation (there is no missingness in `gender` today,
    but this makes the pipeline robust to a future unseen file that has
    some), then one-hot encoding with unknown categories ignored at
    inference time.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_scaled_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Same as build_preprocessor but adds standard scaling on numeric
    features -- used only for the Logistic Regression candidate, where
    unscaled wide-range counts/distances would dominate the L2 penalty.
    Tree-based candidates (Random Forest, HistGradientBoosting) are scale
    invariant and use build_preprocessor() instead."""
    from sklearn.preprocessing import StandardScaler

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
