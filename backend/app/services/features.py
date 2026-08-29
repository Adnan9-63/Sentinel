"""
Feature engineering for Sentinel (batch/offline path).

Critical design constraint: every feature for transaction T is computed using
ONLY transactions that happened strictly before T's timestamp. This is a
causal / point-in-time computation -- a fraud detector benchmarked with
features that peek at the future is not measuring anything real.

This module is now a thin wrapper around app.core.feature_state.FeatureState,
which is the single implementation of the feature logic, shared with the
live API's per-transaction scoring path. Keeping one implementation instead
of two means the live demo can never quietly score things differently from
what was benchmarked here.

Run:
    python -m app.services.features
"""

import pandas as pd
from app.core.feature_state import FeatureState
from app.core.paths import DATA_DIR


def build_features(txns_df: pd.DataFrame, accounts_df: pd.DataFrame) -> pd.DataFrame:
    df = txns_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    state = FeatureState()
    for row in accounts_df.itertuples(index=False):
        state.register_account(row.account_id, pd.to_datetime(row.created_at))

    out_rows = []
    for row in df.itertuples(index=False):
        feats = state.compute_and_update(
            account_id=row.account_id,
            ts=row.timestamp,
            device_id=row.device_id,
            ip_address=row.ip_address,
            amount=row.amount,
            lat=row.geo_lat,
            lon=row.geo_lon,
            is_new_geo_flag=bool(row.is_new_geo),
            status=row.status,
        )
        out_rows.append({
            "transaction_id": row.transaction_id,
            "account_id": row.account_id,
            "timestamp": row.timestamp,
            "device_id": row.device_id,
            **feats,
            "label": row.label,  # ground truth, carried through for evaluation ONLY
        })

    return pd.DataFrame(out_rows)


if __name__ == "__main__":
    txns = pd.read_csv(DATA_DIR / "transactions.csv")
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")

    feat_df = build_features(txns, accounts)

    print(f"Feature rows: {len(feat_df)}")
    print("\nFeature summary by label (mean values):")
    numeric_cols = [
        "velocity_1h", "velocity_24h", "device_velocity_1h", "ip_velocity_1h",
        "device_distinct_accounts_before", "amount_zscore", "geo_jump_km",
        "implied_speed_kmh", "account_age_days", "is_unusual_hour",
        "is_new_device_for_account",
    ]
    print(feat_df.groupby("label")[numeric_cols].mean().round(2).T)

    feat_df.to_csv(DATA_DIR / "features.csv", index=False)
    print("\nSaved to data/features.csv")
