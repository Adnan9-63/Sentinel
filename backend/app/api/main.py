"""
Sentinel API server.

Run locally:
    cd backend
    uvicorn app.api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API docs (FastAPI
generates this automatically).

Everything is loaded ONCE at startup (historical data replayed into a live
FeatureState, trained ML models, ring cluster map) and kept in memory --
this is a hackathon-scale single-process server, not a production
deployment, and that tradeoff is intentional and documented rather than
pretending otherwise.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from app.core.feature_state import FeatureState
from app.services.ml_detection import load_ensemble
from app.services.ring_detection import build_account_graph, score_clusters
from app.core.paths import DATA_DIR as _DATA_DIR

DATA_DIR = str(_DATA_DIR)


class AppState:
    feature_state: FeatureState = None
    accounts_df: pd.DataFrame = None
    scaler = None
    iso = None
    gbt = None
    iso_calibration: dict = None
    ring_map: dict = None


state = AppState()


def _build_ring_map(threshold: float = 0.3) -> dict:
    txns = pd.read_csv(f"{DATA_DIR}/transactions.csv")
    G = build_account_graph(txns)
    clusters = score_clusters(G, txns)
    flagged = clusters[clusters["ring_risk_score"] >= threshold]
    mapping = {}
    for _, row in flagged.iterrows():
        for acc in row["account_ids"]:
            mapping[acc] = {
                "cluster_type": row["cluster_type"],
                "ring_risk_score": float(row["ring_risk_score"]),
                "cluster_size": int(row["n_accounts"]),
            }
    return mapping


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Startup: loading historical data and replaying feature state...")
    accounts_df = pd.read_csv(f"{DATA_DIR}/accounts.csv")
    txns_df = pd.read_csv(f"{DATA_DIR}/transactions.csv")
    txns_df["timestamp"] = pd.to_datetime(txns_df["timestamp"])
    txns_df = txns_df.sort_values("timestamp").reset_index(drop=True)

    fstate = FeatureState()
    for row in accounts_df.itertuples(index=False):
        fstate.register_account(row.account_id, pd.to_datetime(row.created_at))
    for row in txns_df.itertuples(index=False):
        fstate.compute_and_update(
            account_id=row.account_id, ts=row.timestamp, device_id=row.device_id,
            ip_address=row.ip_address, amount=row.amount, lat=row.geo_lat, lon=row.geo_lon,
            is_new_geo_flag=bool(row.is_new_geo), status=row.status,
        )

    state.feature_state = fstate
    state.accounts_df = accounts_df
    state.scaler, state.iso, state.gbt, state.iso_calibration = load_ensemble()
    state.ring_map = _build_ring_map()

    print(f"Startup complete: {len(accounts_df)} accounts, {len(txns_df)} historical "
          f"transactions replayed, {len(state.ring_map)} accounts in flagged ring clusters.")
    yield
    print("Shutdown.")


app = FastAPI(title="Sentinel API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite / CRA dev servers
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.routes import router  # noqa: E402  (import after `app` exists, routes use `state`)
app.include_router(router, prefix="/api")
