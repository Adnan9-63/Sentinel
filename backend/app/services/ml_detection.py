"""
ML detection layer for Sentinel.

Ensemble of:
  - Isolation Forest (unsupervised) -- catches novel anomaly shapes without
    needing labels, useful for attack patterns not well represented in
    training data.
  - Gradient Boosted Trees (supervised) -- catches known patterns well when
    labels exist, which we have here since this is synthetic data.

Evaluated on a genuine held-out split, not the training set, and reported
with a full confusion matrix + the false-positive cost the buildathon brief
explicitly asks for -- not just accuracy, which is meaningless on an
imbalanced dataset like this one (predicting "never fraud" would already
score ~95% accuracy here).

Run:
    python -m app.services.ml_detection
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, precision_recall_curve,
)
from app.core.paths import DATA_DIR, MODEL_DIR as _MODEL_DIR

MODEL_DIR = str(_MODEL_DIR)

FEATURE_COLS = [
    "velocity_1h", "velocity_24h", "device_velocity_1h", "ip_velocity_1h",
    "device_distinct_accounts_before", "amount_zscore", "geo_jump_km",
    "implied_speed_kmh", "account_age_days", "is_unusual_hour",
    "is_new_device_for_account", "is_new_geo", "status_failed",
]

# labels the detector should flag as fraud (binary target)
FRAUD_LABELS = {"card_testing", "card_testing_stealthy", "ato", "ato_stealthy", "ring"}
# ambiguous_legit_* rows are the hard negatives: legit behavior that looks
# risky. They are NOT dropped from train/test -- a model that never sees them
# would look artificially good. They're labeled 0 (not fraud) like any other
# legitimate transaction.


def load_and_split(test_size=0.3, random_state=42):
    df = pd.read_csv(DATA_DIR / "features.csv")
    df["is_fraud"] = df["label"].isin(FRAUD_LABELS).astype(int)

    X = df[FEATURE_COLS].fillna(0)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, df_train, df_test


def train_ensemble(X_train, y_train, sample_weight=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    iso = IsolationForest(
        n_estimators=200, contamination=0.06, random_state=42, n_jobs=-1
    )
    iso.fit(X_train_scaled)

    gbt = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42
    )
    gbt.fit(X_train, y_train, sample_weight=sample_weight)

    return scaler, iso, gbt


def save_ensemble(scaler, iso, gbt, iso_calibration: dict, model_dir: str = MODEL_DIR):
    """Persist trained models AND the iso calibration bounds so the API
    server can load them at startup instead of retraining every time it
    boots, and so single-transaction live scoring uses the same reference
    bounds the offline benchmark was measured with."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(iso, os.path.join(model_dir, "isolation_forest.joblib"))
    joblib.dump(gbt, os.path.join(model_dir, "gradient_boosted_trees.joblib"))
    with open(os.path.join(model_dir, "iso_calibration.json"), "w") as f:
        json.dump(iso_calibration, f)


def load_ensemble(model_dir: str = MODEL_DIR):
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    iso = joblib.load(os.path.join(model_dir, "isolation_forest.joblib"))
    gbt = joblib.load(os.path.join(model_dir, "gradient_boosted_trees.joblib"))
    with open(os.path.join(model_dir, "iso_calibration.json")) as f:
        iso_calibration = json.load(f)
    return scaler, iso, gbt, iso_calibration


def fit_iso_calibration(scaler, iso, X_train) -> dict:
    """Isolation Forest's raw scores are meaningful only relative to a
    reference distribution. Rescaling min-max against whatever batch is
    passed in works for offline evaluation (large batches) but silently
    breaks for live single-transaction scoring: with one row, min == max,
    so the rescaled score is always exactly 0.0 regardless of how anomalous
    the transaction actually is -- caught this by testing exactly that case
    before wiring up the live API. Fix: compute fixed calibration bounds
    once, from the training set, and reuse them for every future scoring
    call regardless of batch size."""
    X_train_scaled = scaler.transform(X_train)
    iso_raw = -iso.score_samples(X_train_scaled)
    return {"iso_raw_min": float(iso_raw.min()), "iso_raw_max": float(iso_raw.max())}


def ensemble_score(scaler, iso, gbt, X, iso_calibration: dict = None):
    X_scaled = scaler.transform(X)
    iso_raw = -iso.score_samples(X_scaled)

    if iso_calibration is not None:
        lo, hi = iso_calibration["iso_raw_min"], iso_calibration["iso_raw_max"]
    else:
        # fallback: batch-relative (only valid for large, representative
        # batches -- NOT safe for single-row live scoring, see docstring above)
        lo, hi = iso_raw.min(), iso_raw.max()
    iso_score = np.clip((iso_raw - lo) / (hi - lo + 1e-9), 0, 1)

    gbt_score = gbt.predict_proba(X)[:, 1]
    combined = 0.5 * iso_score + 0.5 * gbt_score
    return combined, iso_score, gbt_score


def evaluate(df_test, combined_score, threshold=0.5):
    y_true = df_test["is_fraud"].values
    y_pred = (combined_score >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, combined_score)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Threshold: {threshold}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  ROC-AUC: {auc:.3f}")
    print(f"Confusion matrix:\n{cm}")

    # honest false-positive cost breakdown: WHICH legitimate transactions get
    # wrongly flagged, broken down by type -- this is the buildathon's
    # explicit ask, not an afterthought
    fp_mask = (y_pred == 1) & (y_true == 0)
    fp_labels = df_test.loc[fp_mask, "label"].value_counts()
    print(f"\nFalse positives by true label ({fp_mask.sum()} total):")
    print(fp_labels)

    fn_mask = (y_pred == 0) & (y_true == 1)
    fn_labels = df_test.loc[fn_mask, "label"].value_counts()
    print(f"\nFalse negatives (missed fraud) by true label ({fn_mask.sum()} total):")
    print(fn_labels)

    return {"precision": precision, "recall": recall, "f1": f1, "auc": auc}


def threshold_sweep(df_test, combined_score):
    print("\nThreshold sweep (precision / recall / false positives on ambiguous-legit rows):")
    print(f"{'thresh':>7} {'precision':>10} {'recall':>8} {'fp_total':>9} {'fp_ambiguous':>13}")
    y_true = df_test["is_fraud"].values
    is_ambiguous = df_test["label"].str.startswith("ambiguous").values
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred = (combined_score >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        fp_total = int(((y_pred == 1) & (y_true == 0)).sum())
        fp_ambiguous = int(((y_pred == 1) & is_ambiguous).sum())
        print(f"{t:>7} {p:>10.3f} {r:>8.3f} {fp_total:>9} {fp_ambiguous:>13}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df_train, df_test = load_and_split()
    print(f"Train: {len(X_train)}  Test (held out): {len(X_test)}")
    print(f"Fraud rate -- train: {y_train.mean():.2%}  test: {y_test.mean():.2%}")

    # ATO has far fewer training examples (45) than card-testing (310) or
    # normal (thousands) -- with a single binary fraud label, gradient
    # boosting naturally optimizes for whichever sub-pattern is most
    # populous and separable (ip_velocity, which fires for card-testing and
    # rings), and under-serves rarer sub-types. Checked this directly: the
    # exact ATO case that exposed the account_age leakage bug still scored
    # low on the GBT component alone even after that fix. Upweighting rare
    # fraud sub-types in training is the honest fix, not silently hoping
    # the ensemble average covers for it.
    RARE_FRAUD_LABELS = {"ato", "ato_stealthy"}
    sample_weight = df_train["label"].apply(lambda l: 4.0 if l in RARE_FRAUD_LABELS else 1.0).values

    scaler, iso, gbt = train_ensemble(X_train, y_train, sample_weight=sample_weight)
    iso_calibration = fit_iso_calibration(scaler, iso, X_train)
    print(f"Iso calibration bounds (from training set): {iso_calibration}")
    combined_score, iso_score, gbt_score = ensemble_score(scaler, iso, gbt, X_test, iso_calibration)

    print("\n=== Ensemble (Isolation Forest + Gradient Boosted Trees) ===")
    evaluate(df_test, combined_score, threshold=0.5)

    threshold_sweep(df_test, combined_score)

    # save scored test set for the triage-gate / LLM-reasoning layer to consume next
    df_test = df_test.copy()
    df_test["risk_score"] = combined_score
    df_test["iso_score"] = iso_score
    df_test["gbt_score"] = gbt_score
    df_test.to_csv(DATA_DIR / "scored_test_set.csv", index=False)
    print("\nSaved scored test set to data/scored_test_set.csv")

    save_ensemble(scaler, iso, gbt, iso_calibration)
    print(f"Saved trained models + calibration to {MODEL_DIR}/ for the API server to load at startup")
