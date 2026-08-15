"""
train_model.py
--------------
Trains the Random Forest pipeline on the three training CSVs
and saves ed_risk_model.pkl in the same directory as this script.

Usage:
    python train_model.py
    python train_model.py --members ../raw_members.csv --ed ../raw_ed_visits.csv --care ../raw_care_history.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Re-use the shared feature-engineering constants
from feature_engineering import EXPECTED_CARE, DROP_BEFORE_MODEL

BASE = Path(__file__).parent.parent  # project root (one level up from backend/)


# ---------------------------------------------------------------------------
# Full feature engineering for TRAINING
# (identical to feature_engineering.py but also creates the target label)
# ---------------------------------------------------------------------------

def build_training_features(
    members: pd.DataFrame,
    ed: pd.DataFrame,
    care: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for model training."""

    ed   = ed.copy()
    care = care.copy()

    ed["visit_date"]   = pd.to_datetime(ed["visit_date"],   errors="coerce")
    care["visit_date"] = pd.to_datetime(care["visit_date"], errors="coerce")

    reference_date = ed["visit_date"].max()
    ed["days_since_reference"] = (reference_date - ed["visit_date"]).dt.days

    def _win(days: int, name: str) -> pd.DataFrame:
        return (
            ed[
                (ed["days_since_reference"] >= 0)
                & (ed["days_since_reference"] <= days)
            ]
            .groupby("member_id").size().reset_index(name=name)
        )

    ed_365 = _win(365, "ED_visits_365d")
    ed_180 = _win(180, "ED_visits_180d")
    ed_90  = _win(90,  "ED_visits_90d")
    ed_30  = _win(30,  "ED_visits_30d")

    last_ed = ed.groupby("member_id")["visit_date"].max().reset_index(name="last_ED_date")
    last_ed["days_since_last_ED"] = (reference_date - last_ed["last_ED_date"]).dt.days

    diag = pd.crosstab(ed["member_id"], ed["diagnosis"])
    diag.columns = [
        "diagnosis_" + str(c).replace(" ", "_").replace("-", "_").replace("/", "_")
        for c in diag.columns
    ]
    diag = diag.reset_index()

    ed_sum = (
        ed.groupby("member_id")
        .agg(
            ED_admissions=("admitted", "sum"),
            ED_ICU_visits=("icu", "sum"),
            ED_major_procedures=("major_procedure", "sum"),
            ED_red_flags=("red_flag", "sum"),
            ED_total_cost=("cost", "sum"),
            ED_avg_cost=("cost", "mean"),
        )
        .reset_index()
    )

    care_counts = pd.crosstab(care["member_id"], care["care_type"])
    care_counts.columns = [
        "care_" + str(c).replace(" ", "_") for c in care_counts.columns
    ]
    care_counts = care_counts.reset_index()
    for col in EXPECTED_CARE:
        if col not in care_counts.columns:
            care_counts[col] = 0

    last_care = care.groupby("member_id")["visit_date"].max().reset_index(name="last_care_date")
    last_care["days_since_last_care"] = (reference_date - last_care["last_care_date"]).dt.days

    features = members.copy()
    for tbl in [ed_365, ed_180, ed_90, ed_30, last_ed, diag, ed_sum, care_counts, last_care]:
        features = features.merge(tbl, on="member_id", how="left")

    for col in features.columns:
        if col.startswith("ED_") or col.startswith("care_") or col.startswith("diagnosis_"):
            features[col] = features[col].fillna(0)

    for col in ["pcp_distance_miles", "urgent_care_distance_miles"]:
        if col in features.columns:
            features[col] = features[col].fillna(features[col].median())

    features["alternative_care_visits"] = features["care_Urgent_Care"] + features["care_Telehealth"]
    features["total_non_ED_care"] = (
        features["care_PCP"] + features["care_Urgent_Care"]
        + features["care_Telehealth"] + features["care_Care_Management"]
    )
    features["ED_to_alternative_ratio"] = features["ED_visits_365d"] / (
        features["alternative_care_visits"] + 1
    )
    features["access_burden"] = (
        features["transportation_barrier"]
        + (features["pcp_distance_miles"] > 10).astype(int)
        + (features["urgent_care_distance_miles"] > 10).astype(int)
    )
    clinical_cols = [c for c in ["diabetes", "copd", "hypertension", "chf", "asthma", "ckd"]
                     if c in features.columns]
    features["clinical_burden"] = features[clinical_cols].sum(axis=1)

    # ---- Create target (BEFORE dropping ED utilisation columns) ----
    features["frequent_ED_user"] = (features["ED_visits_365d"] >= 2).astype(int)

    y = features["frequent_ED_user"]

    drop_cols = [c for c in DROP_BEFORE_MODEL if c in features.columns]
    X = features.drop(columns=drop_cols, errors="ignore")

    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(members_path: Path, ed_path: Path, care_path: Path) -> None:
    print("=" * 60)
    print("ED RISK MODEL – TRAINING")
    print("=" * 60)

    print("\nLoading training data ...")
    members = pd.read_csv(members_path)
    ed      = pd.read_csv(ed_path)
    care    = pd.read_csv(care_path)
    print(f"  Members      : {members.shape}")
    print(f"  ED Visits    : {ed.shape}")
    print(f"  Care History : {care.shape}")

    print("\nRunning feature engineering ...")
    X, y = build_training_features(members, ed, care)
    print(f"  Feature matrix : {X.shape}")

    print("\nTarget distribution:")
    print(y.value_counts())
    print(y.value_counts(normalize=True).round(3))

    # ---- Train / test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain : {X_train.shape}  |  Test : {X_test.shape}")

    # ---- Preprocessing ----
    cat_feats = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_feats = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric",     num_pipe, num_feats),
            ("categorical", cat_pipe, cat_feats),
        ]
    )

    # ---- Random Forest (as per AVOIDABLE.docx) ----
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", rf)])

    print("\nTraining Random Forest (400 trees) ...")
    pipeline.fit(X_train, y_train)
    print("✅ Training complete.")

    # ---- Evaluation ----
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n=== Model Performance ===")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1 Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ---- Save model package ----
    model_package = {
        "model": pipeline,
        "feature_columns": X.columns.tolist(),
        "target": "frequent_ED_user",
        "target_definition": "2 or more ED visits within 365 days",
    }

    out_path = Path(__file__).parent / "ed_risk_model.pkl"
    joblib.dump(model_package, out_path)
    print(f"\n✅ Saved model → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ED risk Random Forest model")
    parser.add_argument("--members", default=str(BASE / "raw_members.csv"),  help="Path to members CSV")
    parser.add_argument("--ed",      default=str(BASE / "raw_ed_visits.csv"), help="Path to ED visits CSV")
    parser.add_argument("--care",    default=str(BASE / "raw_care_history.csv"), help="Path to care history CSV")
    args = parser.parse_args()

    train(Path(args.members), Path(args.ed), Path(args.care))
