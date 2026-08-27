"""
Sentinel API routes.

Read endpoints (recent decisions, stats, clusters) serve the dashboard.
Simulate endpoints run a brand-new transaction through the REAL pipeline --
same FeatureState, same trained ML ensemble, same triage gate, same
reasoning agent (real API if ANTHROPIC_API_KEY is set on your machine,
graceful fallback if not) -- and log the result to the audit ledger, so
the dashboard's "live feed" is showing genuine pipeline output, not
canned/scripted responses.
"""

import json
import os
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.services.ml_detection import ensemble_score, FEATURE_COLS
from app.core.triage_gate import triage
from app.services.audit_ledger import log_decision, verify_chain, LEDGER_PATH
from app.services.live_simulator import (
    simulate_normal_transaction, simulate_ato_transaction, simulate_card_testing_burst,
)

router = APIRouter()


def _score_and_triage(txn: dict, feature_state, ring_context_override=None) -> dict:
    """Shared logic for both the simulate endpoints: compute features, score
    with the ML ensemble, check ring membership, run the triage gate, log
    the decision. Imported lazily from app.api.main to avoid a circular
    import at module load time.

    ring_context_override lets a caller supply a ring context computed
    LIVE (e.g. from a just-generated burst) instead of relying solely on
    state.ring_map, which is only computed once at startup from historical
    data. Without this, a coordinated attack pattern formed entirely during
    a live demo would score correctly on its own transaction-level features
    (velocity, device reuse) but would never get annotated as part of a
    ring -- the individual risk score still catches it, but the richer
    "this is a coordinated attack" explanation would silently never appear.
    Found by testing exactly that scenario, not by inspection alone."""
    from app.api.main import state

    feats = feature_state.compute_and_update(
        account_id=txn["account_id"], ts=txn["timestamp"], device_id=txn["device_id"],
        ip_address=txn["ip_address"], amount=txn["amount"], lat=txn["geo_lat"], lon=txn["geo_lon"],
        is_new_geo_flag=txn["is_new_geo"], status=txn["status"],
    )

    X = pd.DataFrame([{c: feats[c] for c in FEATURE_COLS}])
    combined, iso_s, gbt_s = ensemble_score(state.scaler, state.iso, state.gbt, X, state.iso_calibration)
    risk_score = float(combined[0])

    ring_context = ring_context_override or state.ring_map.get(txn["account_id"])

    result = triage(risk_score=risk_score, feature_context=feats, ring_context=ring_context)

    record = {
        "transaction_id": txn["transaction_id"],
        "account_id": txn["account_id"],
        "path": result.path,
        "risk_score": risk_score,
        "final_status": result.final_status,
        "evidence": result.evidence_summary,
        "grounding_warnings": result.grounding_warnings,
        "llm_confidence": result.decision.confidence if result.decision else None,
        "llm_rationale": result.decision.rationale if result.decision else None,
        "features": feats,
        "ring_context": ring_context,
        "simulated": True,
    }
    log_decision(record)
    return record


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/transactions/recent")
def recent_transactions(limit: int = Query(50, ge=1, le=500)):
    if not os.path.exists(LEDGER_PATH):
        return {"transactions": []}
    lines = []
    with open(LEDGER_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return {"transactions": lines[-limit:][::-1]}  # most recent first


@router.get("/stats/summary")
def stats_summary():
    if not os.path.exists(LEDGER_PATH):
        return {"total": 0}
    df = pd.read_json(LEDGER_PATH, lines=True)
    path_counts = df["path"].value_counts().to_dict()
    status_counts = df["final_status"].value_counts().to_dict()
    chain = verify_chain()
    return {
        "total": len(df),
        "path_counts": path_counts,
        "status_counts": status_counts,
        "ledger_intact": chain["intact"],
    }


@router.get("/clusters")
def clusters():
    from app.api.main import state
    by_type = {}
    for acc_id, info in state.ring_map.items():
        by_type.setdefault(info["cluster_type"], set()).add(acc_id)
    return {
        "n_flagged_accounts": len(state.ring_map),
        "by_type": {k: len(v) for k, v in by_type.items()},
    }


@router.post("/simulate/normal")
def simulate_normal():
    from app.api.main import state
    if len(state.accounts_df) == 0:
        raise HTTPException(500, "no accounts loaded")
    account_row = state.accounts_df.sample(1).iloc[0].to_dict()
    # Reuse a device this account has actually used before, if any exist --
    # otherwise every "normal" simulated transaction incorrectly looks like
    # a new-device event, which is not what "normal" is supposed to mean.
    # Found this live: a simulated normal transaction scored 0.46 and got
    # flagged, traced it to this exact gap. See FAILURES.md.
    known_devices = state.feature_state.account_devices_seen.get(account_row["account_id"])
    if known_devices:
        account_row["_known_device"] = next(iter(known_devices))
    txn = simulate_normal_transaction(account_row)
    return _score_and_triage(txn, state.feature_state)


@router.post("/simulate/ato")
def simulate_ato():
    from app.api.main import state
    if len(state.accounts_df) == 0:
        raise HTTPException(500, "no accounts loaded")
    account_row = state.accounts_df.sample(1).iloc[0].to_dict()
    txn = simulate_ato_transaction(account_row)
    return _score_and_triage(txn, state.feature_state)


@router.post("/simulate/card_testing_burst")
def simulate_burst(n: int = Query(15, ge=3, le=50)):
    from app.api.main import state
    from app.services.ring_detection import build_account_graph, score_clusters

    txns = simulate_card_testing_burst(n=n)

    # Run ring detection on JUST this live batch -- the burst's transactions
    # share a small attacker device pool by construction, so this batch
    # alone should form a real cluster even though these accounts have no
    # history in state.ring_map (which only knows about historical data).
    # Cheap: this is n<=50 rows, not the full dataset.
    burst_df = pd.DataFrame(txns)
    burst_df["timestamp"] = pd.to_datetime(burst_df["timestamp"])
    burst_df["status"] = burst_df["status"]
    live_ring_context = {}
    try:
        G = build_account_graph(burst_df, min_shared_events_by_type={"device_id": 1, "ip_prefix": 4})
        clusters = score_clusters(G, burst_df, min_cluster_size=3)
        for _, row in clusters[clusters["ring_risk_score"] >= 0.2].iterrows():
            for acc in row["account_ids"]:
                live_ring_context[acc] = {
                    "cluster_type": row["cluster_type"],
                    "ring_risk_score": float(row["ring_risk_score"]),
                    "cluster_size": int(row["n_accounts"]),
                    "detected": "live",  # distinguishes from historical-batch detections
                }
    except Exception:
        pass  # ring detection on the live batch is a best-effort enrichment,
              # never something that should break the simulate endpoint itself

    results = [
        _score_and_triage(txn, state.feature_state, ring_context_override=live_ring_context.get(txn["account_id"]))
        for txn in txns
    ]
    return {"n": len(results), "results": results, "live_ring_detected": len(live_ring_context) > 0}
