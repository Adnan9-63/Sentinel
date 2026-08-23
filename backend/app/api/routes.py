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


def _score_and_triage(txn: dict, feature_state) -> dict:
    """Shared logic for both the simulate endpoints: compute features, score
    with the ML ensemble, check ring membership, run the triage gate, log
    the decision. Imported lazily from app.api.main to avoid a circular
    import at module load time."""
    from app.api.main import state

    feats = feature_state.compute_and_update(
        account_id=txn["account_id"], ts=txn["timestamp"], device_id=txn["device_id"],
        ip_address=txn["ip_address"], amount=txn["amount"], lat=txn["geo_lat"], lon=txn["geo_lon"],
        is_new_geo_flag=txn["is_new_geo"], status=txn["status"],
    )

    X = pd.DataFrame([{c: feats[c] for c in FEATURE_COLS}])
    combined, iso_s, gbt_s = ensemble_score(state.scaler, state.iso, state.gbt, X, state.iso_calibration)
    risk_score = float(combined[0])

    ring_context = state.ring_map.get(txn["account_id"])

    result = triage(risk_score=risk_score, feature_context=feats, ring_context=ring_context)

    record = {
        "transaction_id": txn["transaction_id"],
        "account_id": txn["account_id"],
        "path": result.path,
        "risk_score": risk_score,
        "final_status": result.final_status,
        "evidence": result.evidence_summary,
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
    txns = simulate_card_testing_burst(n=n)
    results = [_score_and_triage(txn, state.feature_state) for txn in txns]
    return {"n": len(results), "results": results}
