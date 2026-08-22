"""
Feature engineering for Sentinel.

Critical design constraint: every feature for transaction T is computed using
ONLY transactions that happened strictly before T's timestamp. This is a
causal / point-in-time computation. It's slower than vectorized rolling
windows over the full dataset, but a fraud detector benchmarked with features
that peek at the future is not measuring anything real -- that's a common,
easy-to-miss mistake and worth being explicit about avoiding.

Run:
    python -m app.services.features
"""

import math
import pandas as pd
import numpy as np
from collections import defaultdict, deque

R_EARTH_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R_EARTH_KM * math.asin(math.sqrt(a))


def ip_prefix(ip: str, octets: int = 2) -> str:
    parts = ip.split(".")
    return ".".join(parts[:octets])


def build_features(txns_df: pd.DataFrame, accounts_df: pd.DataFrame) -> pd.DataFrame:
    df = txns_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ip_prefix"] = df["ip_address"].apply(ip_prefix)

    acc_created = dict(zip(accounts_df["account_id"], pd.to_datetime(accounts_df["created_at"])))

    # Rolling per-account history (timestamps + amounts), updated as we walk forward in time
    account_txn_times = defaultdict(list)      # account_id -> list of past timestamps
    account_amounts = defaultdict(list)        # account_id -> list of past amounts
    account_last_geo = {}                       # account_id -> (lat, lon, ts)
    account_devices_seen = defaultdict(set)     # account_id -> set of device_ids used before
    device_accounts_seen = defaultdict(set)     # device_id -> set of account_ids that used it before
    device_recent_times = defaultdict(list)     # device_id -> recent timestamps (any account)
    ipprefix_recent_times = defaultdict(list)   # ip_prefix -> recent timestamps (any account)

    out_rows = []

    for row in df.itertuples(index=False):
        acc = row.account_id
        ts = row.timestamp
        device = row.device_id
        ipp = row.ip_prefix
        amount = row.amount
        lat, lon = row.geo_lat, row.geo_lon

        # -- velocity features (own-account, causal) --
        past_times = account_txn_times[acc]
        v_1h = sum(1 for t in past_times if (ts - t).total_seconds() <= 3600)
        v_24h = sum(1 for t in past_times if (ts - t).total_seconds() <= 86400)

        # -- device velocity across ALL accounts (card-testing / ring signal) --
        dtimes = device_recent_times[device]
        # prune anything older than 1h to keep this cheap
        dtimes = [t for t in dtimes if (ts - t).total_seconds() <= 3600]
        device_recent_times[device] = dtimes
        device_velocity_1h = len(dtimes)

        # -- IP-prefix velocity across ALL accounts --
        iptimes = ipprefix_recent_times[ipp]
        iptimes = [t for t in iptimes if (ts - t).total_seconds() <= 3600]
        ipprefix_recent_times[ipp] = iptimes
        ip_velocity_1h = len(iptimes)

        # -- device reuse across distinct accounts (ring signal) --
        device_distinct_accounts_before = len(device_accounts_seen[device])

        # -- amount z-score vs this account's own causal history --
        hist_amounts = account_amounts[acc]
        if len(hist_amounts) >= 3:
            mu = float(np.mean(hist_amounts))
            sigma = float(np.std(hist_amounts)) or 1.0
            amount_zscore = (amount - mu) / sigma
        else:
            amount_zscore = 0.0  # insufficient history -> neutral, not a signal yet

        # -- geo jump / implied travel speed vs this account's last known location --
        if acc in account_last_geo:
            plat, plon, pts = account_last_geo[acc]
            dist_km = haversine_km(plat, plon, lat, lon)
            hours = max((ts - pts).total_seconds() / 3600.0, 1e-6)
            implied_speed_kmh = dist_km / hours
        else:
            dist_km = 0.0
            implied_speed_kmh = 0.0

        # -- account age at time of transaction --
        created = acc_created.get(acc, ts)
        account_age_days = max((ts - created).total_seconds() / 86400.0, 0.0)

        # -- unusual hour flag (00:00-05:00 local, a mild signal not a rule) --
        is_unusual_hour = 1 if ts.hour < 5 else 0

        is_new_device_for_account = device not in account_devices_seen[acc]

        out_rows.append({
            "transaction_id": row.transaction_id,
            "account_id": acc,
            "timestamp": ts,
            "device_id": device,
            "ip_prefix": ipp,
            "amount": amount,
            "velocity_1h": v_1h,
            "velocity_24h": v_24h,
            "device_velocity_1h": device_velocity_1h,
            "ip_velocity_1h": ip_velocity_1h,
            "device_distinct_accounts_before": device_distinct_accounts_before,
            "amount_zscore": amount_zscore,
            "geo_jump_km": dist_km,
            "implied_speed_kmh": implied_speed_kmh,
            "account_age_days": account_age_days,
            "is_unusual_hour": is_unusual_hour,
            "is_new_device_for_account": int(is_new_device_for_account),
            "is_new_geo": int(row.is_new_geo),
            "status_failed": int(row.status == "failed"),
            "label": row.label,  # ground truth, carried through for evaluation ONLY
        })

        # -- update rolling state AFTER computing features for this row (causal) --
        account_txn_times[acc].append(ts)
        account_amounts[acc].append(amount)
        account_last_geo[acc] = (lat, lon, ts)
        account_devices_seen[acc].add(device)
        device_accounts_seen[device].add(acc)
        device_recent_times[device].append(ts)
        ipprefix_recent_times[ipp].append(ts)

    feat_df = pd.DataFrame(out_rows)
    return feat_df


if __name__ == "__main__":
    txns = pd.read_csv("/home/claude/sentinel/data/transactions.csv")
    accounts = pd.read_csv("/home/claude/sentinel/data/accounts.csv")

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

    feat_df.to_csv("/home/claude/sentinel/data/features.csv", index=False)
    print("\nSaved to data/features.csv")
