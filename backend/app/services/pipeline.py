"""
Full pipeline integration test for Sentinel.

Runs actual scored transactions (from ml_detection.py's output) through the
triage gate and audit ledger end to end. This proves the WIRING works --
thresholds route correctly, decisions get logged, the chain stays intact.

IMPORTANT HONESTY NOTE: this build environment has no ANTHROPIC_API_KEY, so
this script uses `mock_reasonable_llm` below instead of a real model call.
It's a simple rule-based stand-in -- NOT a claim about real LLM reasoning
quality. It exists only to prove the pipeline plumbing (routing, logging,
tamper-evidence) is correct. Real reasoning-layer quality can only be
measured against the live API, which requires your own ANTHROPIC_API_KEY.
Run with a real key by setting the environment variable and passing
call_llm_live (the default) instead of this mock.

Run:
    python -m app.services.pipeline
"""

import os
import json
import pandas as pd

from app.core.triage_gate import triage, LOW_THRESHOLD, HIGH_THRESHOLD
from app.services.audit_ledger import log_decision, verify_chain, LEDGER_PATH

FEATURE_COLS = [
    "velocity_1h", "velocity_24h", "device_velocity_1h", "ip_velocity_1h",
    "device_distinct_accounts_before", "amount_zscore", "geo_jump_km",
    "implied_speed_kmh", "account_age_days", "is_unusual_hour",
    "is_new_device_for_account", "is_new_geo", "status_failed",
]


def mock_reasonable_llm(system_prompt: str, user_prompt: str) -> str:
    """Rule-based stand-in for the real model, used ONLY because this build
    environment has no API key. Derives its answer from the feature values
    in the prompt itself (not from any ground-truth label), so it's an
    honest test of the pipeline plumbing even though it's not a real LLM."""
    data = json.loads(user_prompt.split("<untrusted_data>\n", 1)[1].split("\n</untrusted_data>")[0])
    f = data.get("transaction_features", {})

    signals = []
    if f.get("amount_zscore", 0) > 3:
        signals.append(f"amount is {f['amount_zscore']:.1f} std devs above account's normal spend")
    if f.get("is_new_device_for_account") and f.get("is_new_geo"):
        signals.append("new device and new location together")
    if f.get("device_velocity_1h", 0) > 5:
        signals.append(f"elevated device velocity ({f['device_velocity_1h']:.0f}/hr)")
    if data.get("ring_context"):
        signals.append("account is a member of a flagged coordinated cluster")

    if len(signals) >= 2:
        action, confidence = "review", 0.7
    elif len(signals) == 1:
        action, confidence = "review", 0.55
    else:
        action, confidence = "allow", 0.8
        signals = ["no strong individual risk signals in the mid-range score band"]

    return json.dumps({
        "action": action,
        "confidence": confidence,
        "evidence": signals[:6],
        "rationale": f"Based on {len(signals)} observed signal(s) in the feature data, "
                     f"recommending '{action}' with confidence {confidence}.",
    })


def load_ring_context_map(threshold: float = 0.3) -> dict:
    """account_id -> ring context dict, for accounts in a flagged cluster."""
    path = "/home/claude/sentinel/data/ring_clusters_summary.csv"
    if not os.path.exists(path):
        return {}
    clusters = pd.read_csv(path)
    flagged = clusters[clusters["ring_risk_score"] >= threshold]
    # ring_clusters_summary.csv doesn't carry account_ids (dropped before
    # saving in ring_detection.py to keep the CSV flat) -- rebuild from the
    # graph directly instead for this integration run.
    from app.services.ring_detection import build_account_graph, score_clusters
    txns = pd.read_csv("/home/claude/sentinel/data/transactions.csv")
    G = build_account_graph(txns)
    full_clusters = score_clusters(G, txns)
    full_flagged = full_clusters[full_clusters["ring_risk_score"] >= threshold]

    mapping = {}
    for _, row in full_flagged.iterrows():
        for acc in row["account_ids"]:
            mapping[acc] = {
                "cluster_type": row["cluster_type"],
                "ring_risk_score": row["ring_risk_score"],
                "cluster_size": row["n_accounts"],
            }
    return mapping


def run(sample_size: int = 300, seed: int = 42):
    if os.path.exists(LEDGER_PATH):
        os.remove(LEDGER_PATH)

    df = pd.read_csv("/home/claude/sentinel/data/scored_test_set.csv")
    ring_map = load_ring_context_map()

    # deliberately oversample the ambiguous middle band so this run actually
    # exercises the LLM-reasoning path, not just the easy auto-allow cases
    low_band = df[df["risk_score"] < LOW_THRESHOLD].sample(
        n=min(150, len(df[df["risk_score"] < LOW_THRESHOLD])), random_state=seed)
    mid_band = df[(df["risk_score"] >= LOW_THRESHOLD) & (df["risk_score"] < HIGH_THRESHOLD)]
    high_band = df[df["risk_score"] >= HIGH_THRESHOLD].sample(
        n=min(80, len(df[df["risk_score"] >= HIGH_THRESHOLD])), random_state=seed)
    sample = pd.concat([low_band, mid_band, high_band]).reset_index(drop=True)

    print(f"Running {len(sample)} transactions through the full pipeline "
          f"({len(low_band)} low, {len(mid_band)} mid-band, {len(high_band)} high)")

    path_counts = {"auto_allow": 0, "llm_reasoned": 0, "auto_flag_obvious": 0}
    status_counts = {"allowed": 0, "flagged_for_review": 0}

    for _, row in sample.iterrows():
        feature_context = {c: row[c] for c in FEATURE_COLS}
        ring_context = ring_map.get(row["account_id"])

        result = triage(
            risk_score=row["risk_score"],
            feature_context=feature_context,
            ring_context=ring_context,
            llm_caller=mock_reasonable_llm,
        )

        path_counts[result.path] += 1
        status_counts[result.final_status] += 1

        log_decision({
            "transaction_id": row["transaction_id"],
            "account_id": row["account_id"],
            "true_label": row["label"],  # kept for our own eval only, not shown to the LLM
            "path": result.path,
            "risk_score": float(row["risk_score"]),
            "final_status": result.final_status,
            "evidence": result.evidence_summary,
            "llm_confidence": result.decision.confidence if result.decision else None,
        })

    print(f"\nTriage path breakdown: {path_counts}")
    print(f"Final status breakdown: {status_counts}")

    chain = verify_chain()
    print(f"\nAudit chain verification: {chain}")
    assert chain["intact"], "audit chain broke during a clean run -- real bug if this fails"

    # sanity: cross-check flagged_for_review against true labels for an
    # honest read on how the FULL pipeline (not just the ML layer alone)
    # performs, now that some review-band cases get overturned to "allow"
    # by the (mock) reasoning layer
    ledger_df = pd.read_json(LEDGER_PATH, lines=True)
    fraud_labels = {"card_testing", "card_testing_stealthy", "ato", "ato_stealthy", "ring"}
    ledger_df["is_fraud"] = ledger_df["true_label"].isin(fraud_labels)
    print("\nFinal status vs. true label (full pipeline, mock reasoning layer):")
    print(pd.crosstab(ledger_df["final_status"], ledger_df["is_fraud"]))


if __name__ == "__main__":
    run()
