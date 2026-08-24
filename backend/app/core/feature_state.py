"""
FeatureState: the single implementation of Sentinel's causal feature logic.

Why this exists as its own class rather than living only inside
features.py's batch loop: the live API needs to score brand-new
transactions one at a time as they arrive, using the exact same feature
definitions that were benchmarked in ml_detection.py. Re-implementing the
same logic twice (once for batch, once for live) would let the two
versions quietly drift apart over time -- a live demo that scores things
differently from what was benchmarked is a real, embarrassing correctness
bug waiting to happen. So there's exactly one implementation here, and both
features.py (batch/offline) and the live API import and reuse it.
"""

import math
import threading
from collections import defaultdict


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def ip_prefix(ip: str, octets: int = 2) -> str:
    parts = ip.split(".")
    return ".".join(parts[:octets])


class FeatureState:
    """Holds all the rolling, causal state needed to compute features for
    the NEXT transaction. Call compute_and_update() once per transaction,
    in chronological order -- it returns that transaction's feature dict
    AND updates its own internal state so the next call sees this one as
    history. Never call it out of time order; the "causal" guarantee
    (no future leakage) depends on strict chronological calls."""

    def __init__(self):
        self.account_txn_times = defaultdict(list)
        self.account_amounts = defaultdict(list)
        self.account_last_geo = {}
        self.account_devices_seen = defaultdict(set)
        self.device_accounts_seen = defaultdict(set)
        self.device_recent_times = defaultdict(list)
        self.ipprefix_recent_times = defaultdict(list)
        self.account_created_at = {}
        # Same reasoning as audit_ledger.py's lock: this state is shared and
        # mutated by every request, and FastAPI runs sync endpoints across a
        # thread pool within one process. Without this, two concurrent
        # transactions for the same account/device could each read stale
        # velocity counts before either writes back -- not a crash, a
        # silent, hard-to-notice wrong-number bug, which is worse. Added
        # proactively after the same class of bug was CONFIRMED (not just
        # theorized) in the audit ledger under concurrent load.
        self._lock = threading.Lock()

    def register_account(self, account_id, created_at):
        self.account_created_at[account_id] = created_at

    def compute_and_update(
        self, account_id, ts, device_id, ip_address, amount, lat, lon,
        is_new_geo_flag: bool, status: str,
    ) -> dict:
        with self._lock:
            return self._compute_and_update_locked(
                account_id, ts, device_id, ip_address, amount, lat, lon,
                is_new_geo_flag, status,
            )

    def _compute_and_update_locked(
        self, account_id, ts, device_id, ip_address, amount, lat, lon,
        is_new_geo_flag: bool, status: str,
    ) -> dict:
        ipp = ip_prefix(ip_address)

        past_times = self.account_txn_times[account_id]
        v_1h = sum(1 for t in past_times if (ts - t).total_seconds() <= 3600)
        v_24h = sum(1 for t in past_times if (ts - t).total_seconds() <= 86400)

        dtimes = [t for t in self.device_recent_times[device_id] if (ts - t).total_seconds() <= 3600]
        self.device_recent_times[device_id] = dtimes
        device_velocity_1h = len(dtimes)

        iptimes = [t for t in self.ipprefix_recent_times[ipp] if (ts - t).total_seconds() <= 3600]
        self.ipprefix_recent_times[ipp] = iptimes
        ip_velocity_1h = len(iptimes)

        device_distinct_accounts_before = len(self.device_accounts_seen[device_id])

        hist_amounts = self.account_amounts[account_id]
        if len(hist_amounts) >= 3:
            mu = sum(hist_amounts) / len(hist_amounts)
            var = sum((a - mu) ** 2 for a in hist_amounts) / len(hist_amounts)
            sigma = var ** 0.5 or 1.0
            amount_zscore = (amount - mu) / sigma
        else:
            amount_zscore = 0.0

        if account_id in self.account_last_geo:
            plat, plon, pts = self.account_last_geo[account_id]
            dist_km = haversine_km(plat, plon, lat, lon)
            hours = max((ts - pts).total_seconds() / 3600.0, 1e-6)
            implied_speed_kmh = dist_km / hours
        else:
            dist_km = 0.0
            implied_speed_kmh = 0.0

        created = self.account_created_at.get(account_id, ts)
        account_age_days = max((ts - created).total_seconds() / 86400.0, 0.0)

        is_unusual_hour = 1 if ts.hour < 5 else 0
        is_new_device_for_account = int(device_id not in self.account_devices_seen[account_id])

        features = {
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
            "is_new_device_for_account": is_new_device_for_account,
            "is_new_geo": int(is_new_geo_flag),
            "status_failed": int(status == "failed"),
        }

        # update state AFTER computing this transaction's features (causal)
        self.account_txn_times[account_id].append(ts)
        self.account_amounts[account_id].append(amount)
        self.account_last_geo[account_id] = (lat, lon, ts)
        self.account_devices_seen[account_id].add(device_id)
        self.device_accounts_seen[device_id].add(account_id)
        self.device_recent_times[device_id].append(ts)
        self.ipprefix_recent_times[ipp].append(ts)

        return features
