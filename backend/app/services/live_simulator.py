"""
On-demand transaction simulator for the live API.

The offline data_generator.py builds a whole 30-day dataset up front. This
module generates ONE (or a short burst of) new transaction(s) on demand,
against the SAME live FeatureState the API is already running with -- so a
simulated attack shows up as a rising risk score in real time, the same way
a real one would. This is what powers the "watch it happen live" part of
the demo.
"""

import uuid
import math
from datetime import datetime, timedelta
import numpy as np

CITIES = {
    "Bengaluru": (12.9716, 77.5946), "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025), "Patna": (25.5941, 85.1376),
    "Hyderabad": (17.3850, 78.4867), "Pune": (18.5204, 73.8567),
}


def _jitter(lat, lon, km_std=3.0):
    deg_std = km_std / 111.0
    return lat + np.random.normal(0, deg_std), lon + np.random.normal(0, deg_std)


def new_device_id():
    return f"dev_{uuid.uuid4().hex[:12]}"


def new_ip(prefix=None):
    if prefix:
        return f"{prefix}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}"
    return f"{np.random.randint(1, 223)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}"


def simulate_normal_transaction(account_row, now=None):
    """One ordinary transaction for an existing, real account."""
    now = now or datetime.utcnow()
    lat, lon = _jitter(account_row["home_lat"], account_row["home_lon"], km_std=8)
    amount = float(np.round(np.random.lognormal(mean=6.5, sigma=0.9), 2))
    amount = min(amount, 45000)
    return {
        "transaction_id": f"txn_{uuid.uuid4().hex[:12]}",
        "account_id": account_row["account_id"],
        "timestamp": now,
        "device_id": account_row.get("_known_device") or new_device_id(),
        "ip_address": new_ip(),
        "geo_lat": lat, "geo_lon": lon,
        "amount": amount,
        "is_new_geo": False,
        "status": "success",
    }


def simulate_ato_transaction(account_row, now=None):
    """A takeover-style transaction for an existing account: new device, new
    city, unusually large amount -- the loud variant, for a clear demo."""
    now = now or datetime.utcnow()
    other_cities = [c for c in CITIES if c != account_row.get("home_city")]
    city = np.random.choice(other_cities) if other_cities else list(CITIES.keys())[0]
    lat, lon = _jitter(*CITIES[city], km_std=3)
    amount = float(np.round(np.random.uniform(15000, 48000), 2))
    return {
        "transaction_id": f"txn_{uuid.uuid4().hex[:12]}",
        "account_id": account_row["account_id"],
        "timestamp": now,
        "device_id": new_device_id(),
        "ip_address": new_ip(),
        "geo_lat": lat, "geo_lon": lon,
        "amount": amount,
        "is_new_geo": True,
        "status": "success",
    }


def simulate_card_testing_burst(n=15, now=None):
    """A short burst of tiny-amount transactions from a small shared device
    pool against synthetic target accounts, seconds apart -- returns a list
    so the caller can feed them through the pipeline one at a time and watch
    device_velocity_1h (and the resulting risk score) climb across the burst."""
    now = now or datetime.utcnow()
    attacker_devices = [new_device_id() for _ in range(3)]
    ip_prefix = f"{np.random.randint(10, 223)}.{np.random.randint(0, 255)}"
    txns = []
    for i in range(n):
        device = np.random.choice(attacker_devices)
        acc_id = f"acc_live_synthetic_{uuid.uuid4().hex[:8]}"
        city = np.random.choice(list(CITIES.keys()))
        lat, lon = _jitter(*CITIES[city], km_std=1)
        amount = float(np.round(np.random.uniform(1, 49), 2))
        status = "success" if np.random.random() < 0.06 else "failed"
        txns.append({
            "transaction_id": f"txn_{uuid.uuid4().hex[:12]}",
            "account_id": acc_id,
            "timestamp": now + timedelta(seconds=i * 4),  # 4 seconds apart
            "device_id": device,
            "ip_address": new_ip(ip_prefix),
            "geo_lat": lat, "geo_lon": lon,
            "amount": amount,
            "is_new_geo": True,
            "status": status,
        })
    return txns
