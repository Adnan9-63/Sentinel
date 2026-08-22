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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, precision_recall_curve,
)

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
    df = pd.read_csv("/home/claude/sentinel/data/features.csv")
    df["is_fraud"] = df["label"].isin(FRAUD_LABELS).astype(int)

    X = df[FEATURE_COLS].fillna(0)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, df_train, df_test


def train_ensemble(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    iso = IsolationForest(
        n_estimators=200, contamination=0.06, random_state=42, n_jobs=-1
    )
    iso.fit(X_train_scaled)

    gbt = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42
    )
    gbt.fit(X_train, y_train)

    return scaler, iso, gbt


def ensemble_score(scaler, iso, gbt, X):
    X_scaled = scaler.transform(X)
    # Isolation Forest: more negative = more anomalous. Rescale to 0-1.
    iso_raw = -iso.score_samples(X_scaled)
    iso_score = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-9)

    gbt_score = gbt.predict_proba(X)[:, 1]

    # simple average ensemble -- not tuned/weighted beyond this, kept
    # interpretable rather than squeezing out a marginal AUC gain
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

    scaler, iso, gbt = train_ensemble(X_train, y_train)
    combined_score, iso_score, gbt_score = ensemble_score(scaler, iso, gbt, X_test)

    print("\n=== Ensemble (Isolation Forest + Gradient Boosted Trees) ===")
    evaluate(df_test, combined_score, threshold=0.5)

    threshold_sweep(df_test, combined_score)

    # save scored test set for the triage-gate / LLM-reasoning layer to consume next
    df_test = df_test.copy()
    df_test["risk_score"] = combined_score
    df_test["iso_score"] = iso_score
    df_test["gbt_score"] = gbt_score
    df_test.to_csv("/home/claude/sentinel/data/scored_test_set.csv", index=False)
    print("\nSaved scored test set to data/scored_test_set.csv")
