"""
Base-rate precision projection for Sentinel.

The held-out test set has a 5.16% fraud rate (deliberately oversampled so
the model has enough positive examples to learn from -- see
data_generator.py). Real-world payment fraud rates are under 1%, often
closer to 0.1-0.5%. Precision is NOT base-rate-invariant -- the same
detector, with the same recall, produces very different precision at
different fraud prevalences. Reporting "1.000 precision" without this
context is technically true and practically misleading.

Naive approach: FPR was measured as exactly 0.0 (0 false positives among
2,407 negative test examples) and precision at any prevalence would be
100% if that's trusted literally. That's overconfident -- 0 observed
events in a finite sample does NOT mean the true rate is 0. This module
uses the Clopper-Pearson exact one-sided 95% confidence bound instead: the
statistically correct answer to "how bad could the true false-positive
rate plausibly be, given what we've actually observed."

Run:
    python -m app.services.base_rate_analysis
"""

import pandas as pd
from scipy.stats import beta
from app.core.paths import DATA_DIR


def confusion_counts(scored_df: pd.DataFrame, threshold: float = 0.5):
    y_true = scored_df["is_fraud"] if "is_fraud" in scored_df.columns else (
        scored_df["label"].isin(["card_testing", "card_testing_stealthy", "ato", "ato_stealthy", "ring"]).astype(int)
    )
    y_pred = (scored_df["risk_score"] >= threshold).astype(int)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    return TP, FN, FP, TN


def fpr_upper_bound(FP: int, n_negatives: int, confidence: float = 0.95) -> float:
    """Clopper-Pearson exact one-sided upper confidence bound on the true
    false-positive rate, given FP observed events in n_negatives trials."""
    return beta.ppf(confidence, FP + 1, n_negatives - FP)


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    """Bayes' rule: precision as a function of recall (TPR), false-positive
    rate (FPR), and the TRUE base rate of fraud in the population -- not
    the base rate of the test set, which is a design choice, not reality."""
    numerator = tpr * prevalence
    denominator = numerator + fpr * (1 - prevalence)
    return numerator / denominator if denominator > 0 else float("nan")


def run(threshold: float = 0.5, confidence: float = 0.95):
    df = pd.read_csv(DATA_DIR / "scored_test_set.csv")
    TP, FN, FP, TN = confusion_counts(df, threshold)
    n_negatives = FP + TN
    tpr = TP / (TP + FN)
    fpr_naive = FP / n_negatives
    fpr_conservative = fpr_upper_bound(FP, n_negatives, confidence)

    print(f"Confusion matrix at threshold {threshold}: TP={TP} FN={FN} FP={FP} TN={TN}")
    print(f"Recall (TPR): {tpr:.4f}")
    print(f"FPR, naive point estimate: {fpr_naive:.6f}")
    print(f"FPR, {confidence:.0%} upper confidence bound (Clopper-Pearson, n={n_negatives}): {fpr_conservative:.6f}")
    print()

    prevalences = [0.0516, 0.01, 0.005, 0.001, 0.0005]
    rows = []
    for p in prevalences:
        rows.append({
            "prevalence": p,
            "precision_naive": precision_at_prevalence(tpr, fpr_naive, p),
            "precision_conservative": precision_at_prevalence(tpr, fpr_conservative, p),
        })

    print(f"{'prevalence':>12} {'precision (naive)':>20} {'precision (95% CI bound)':>26}")
    for r in rows:
        label = f"{r['prevalence']*100:.3f}%"
        print(f"{label:>12} {r['precision_naive']*100:>19.1f}% {r['precision_conservative']*100:>25.1f}%")

    return rows, tpr, fpr_naive, fpr_conservative


if __name__ == "__main__":
    run()
