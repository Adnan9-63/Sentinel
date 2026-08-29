"""
Synthetic transaction data generator for Sentinel.

Generates a realistic mixed population:
  - normal legitimate accounts (majority)
  - card-testing attack bursts
  - account-takeover (ATO) events
  - coordinated abuse rings (bot-driven clusters)
  - ambiguous "hard" legit cases designed to look risky but aren't

Every row carries a ground-truth label so we can measure real precision/
recall/false-positive cost later -- not a cherry-picked demo.

Run:
    python -m app.services.data_generator
"""

import numpy as np
import pandas as pd
from faker import Faker
import uuid
import math
import json
from datetime import datetime, timedelta
from app.core.paths import DATA_DIR

SEED = 42
rng = np.random.default_rng(SEED)
fake = Faker("en_IN")
Faker.seed(SEED)

SIM_START = datetime(2026, 7, 24)   # 30-day observation window ending Aug 22
SIM_DAYS = 30
SIM_END = SIM_START + timedelta(days=SIM_DAYS)

# A handful of real Indian metro coordinates used as "home" cities for
# legitimate accounts, and as anchors for geo-distance features.
CITIES = {
    "Bengaluru": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Patna": (25.5941, 85.1376),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
}
CITY_NAMES = list(CITIES.keys())

MERCHANT_CATEGORIES = [
    "ecommerce_general", "food_delivery", "electronics", "fashion",
    "travel", "subscription", "gaming", "gifting", "b2b_services",
]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def rand_timestamp(start, end):
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=float(rng.uniform(0, delta)))


def jitter_geo(lat, lon, km_std=3.0):
    # small random jitter around a city center, roughly km_std kilometers
    deg_std = km_std / 111.0
    return lat + rng.normal(0, deg_std), lon + rng.normal(0, deg_std)


def new_device_id():
    return f"dev_{uuid.uuid4().hex[:12]}"


def new_ip(cluster_prefix=None):
    if cluster_prefix:
        return f"{cluster_prefix}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
    return fake.ipv4_public()


# ---------------------------------------------------------------------------
# Population generators
# ---------------------------------------------------------------------------

def gen_normal_accounts(n_accounts):
    accounts = []
    for _ in range(n_accounts):
        city = rng.choice(CITY_NAMES)
        lat, lon = CITIES[city]
        lat, lon = jitter_geo(lat, lon, km_std=5)
        created = rand_timestamp(SIM_START - timedelta(days=365), SIM_START - timedelta(days=10))
        accounts.append({
            "account_id": f"acc_{uuid.uuid4().hex[:10]}",
            "created_at": created,
            "home_city": city,
            "home_lat": lat,
            "home_lon": lon,
            "kyc_status": "verified",
            "account_type": "individual" if rng.random() > 0.12 else "business",
            "ring_id": None,
            "label_class": "normal",
        })
    return accounts


def gen_new_signup_accounts(n_accounts):
    """Accounts created WITHIN the observation window, close to when their
    transactions happen -- i.e. account_age_days will genuinely be small,
    the same as fraud accounts, but the behavior is normal. This is what
    prevents account age from being a clean fraud shortcut."""
    accounts = []
    for _ in range(n_accounts):
        city = rng.choice(CITY_NAMES)
        lat, lon = CITIES[city]
        lat, lon = jitter_geo(lat, lon, km_std=5)
        created = rand_timestamp(SIM_START, SIM_END - timedelta(days=2))
        accounts.append({
            "account_id": f"acc_new_{uuid.uuid4().hex[:10]}",
            "created_at": created,
            "home_city": city,
            "home_lat": lat,
            "home_lon": lon,
            "kyc_status": "verified",
            "account_type": "individual",
            "ring_id": None,
            "label_class": "new_signup",
        })
    return accounts


def gen_normal_transactions(account, n_txn_range=(5, 30)):
    """A normal account: mostly same 1-2 devices, mostly home geo, log-normal amounts."""
    txns = []
    n_devices = rng.integers(1, 3)
    devices = [new_device_id() for _ in range(n_devices)]
    n_txn = int(rng.integers(*n_txn_range))
    for _ in range(n_txn):
        ts = rand_timestamp(SIM_START, SIM_END)
        device = rng.choice(devices)
        # rare legitimate travel (5% of transactions from a different, plausible city)
        if rng.random() < 0.05:
            travel_city = rng.choice(CITY_NAMES)
            lat, lon = jitter_geo(*CITIES[travel_city], km_std=10)
            is_new_geo = True
        else:
            lat, lon = jitter_geo(account["home_lat"], account["home_lon"], km_std=8)
            is_new_geo = False
        amount = float(np.round(rng.lognormal(mean=6.5, sigma=0.9), 2))  # ~ hundreds to low thousands INR
        amount = min(amount, 45000)
        txns.append(_txn_row(
            account["account_id"], ts, device, new_ip(), lat, lon, amount,
            is_new_device=(rng.random() < 0.03), is_new_geo=is_new_geo,
            status="success" if rng.random() > 0.03 else "failed",
            label="normal",
        ))
    return txns


def gen_ambiguous_legit_transactions(account, kind):
    """Deliberately hard cases: legit behavior that LOOKS risky by naive rules."""
    txns = []
    if kind == "payroll_batch":
        # a small business paying many people in a tight window -> high velocity, same device
        device = new_device_id()
        burst_start = rand_timestamp(SIM_START, SIM_END - timedelta(hours=1))
        n = int(rng.integers(20, 40))
        for i in range(n):
            ts = burst_start + timedelta(seconds=float(rng.uniform(0, 1800)))
            amount = float(np.round(rng.normal(15000, 3000), 2))
            amount = max(amount, 2000)
            lat, lon = jitter_geo(account["home_lat"], account["home_lon"], km_std=2)
            txns.append(_txn_row(
                account["account_id"], ts, device, new_ip(), lat, lon, amount,
                is_new_device=False, is_new_geo=False, status="success",
                label="ambiguous_legit_payroll",
            ))
    elif kind == "festival_spree":
        # legit user, real burst of gift-buying, multiple categories, still home geo
        device = new_device_id()
        burst_start = rand_timestamp(SIM_START, SIM_END - timedelta(hours=6))
        n = int(rng.integers(8, 16))
        for i in range(n):
            ts = burst_start + timedelta(minutes=float(rng.uniform(0, 240)))
            amount = float(np.round(rng.lognormal(mean=7.2, sigma=0.6), 2))
            lat, lon = jitter_geo(account["home_lat"], account["home_lon"], km_std=5)
            txns.append(_txn_row(
                account["account_id"], ts, device, new_ip(), lat, lon, amount,
                is_new_device=False, is_new_geo=False, status="success",
                label="ambiguous_legit_festival",
            ))
    elif kind == "genuine_travel":
        # legit international/domestic travel: new geo AND new device (hotel wifi), but
        # plausible travel time between events (not impossible-travel)
        home = (account["home_lat"], account["home_lon"])
        travel_city = rng.choice([c for c in CITY_NAMES])
        dest = CITIES[travel_city]
        device_home = new_device_id()
        device_travel = new_device_id()
        t0 = rand_timestamp(SIM_START, SIM_END - timedelta(days=5))
        # a couple of transactions at home before leaving
        for _ in range(2):
            ts = t0 - timedelta(hours=float(rng.uniform(1, 48)))
            lat, lon = jitter_geo(*home, km_std=5)
            txns.append(_txn_row(
                account["account_id"], ts, device_home, new_ip(), lat, lon,
                float(np.round(rng.lognormal(6.3, 0.7), 2)),
                is_new_device=False, is_new_geo=False, status="success",
                label="ambiguous_legit_travel",
            ))
        # travel window: several transactions at destination over a few days
        for i in range(int(rng.integers(4, 10))):
            ts = t0 + timedelta(hours=float(rng.uniform(3, 96)))
            lat, lon = jitter_geo(*dest, km_std=6)
            txns.append(_txn_row(
                account["account_id"], ts, device_travel, new_ip(), lat, lon,
                float(np.round(rng.lognormal(6.8, 0.8), 2)),
                is_new_device=(i == 0), is_new_geo=True, status="success",
                label="ambiguous_legit_travel",
            ))
    return txns


def gen_card_testing_attack(attack_id):
    """A card-testing burst: one attacker device/IP cluster hammers many accounts
    (mostly freshly created / synthetic) with tiny amounts in a short window,
    high decline rate."""
    txns = []
    ip_prefix = f"{rng.integers(10, 223)}.{rng.integers(0, 255)}"
    attacker_devices = [new_device_id() for _ in range(int(rng.integers(2, 5)))]
    burst_start = rand_timestamp(SIM_START, SIM_END - timedelta(minutes=30))
    n = int(rng.integers(40, 120))
    target_accounts = [f"acc_synthetic_{attack_id}_{i}" for i in range(int(rng.integers(15, 40)))]
    for _ in range(n):
        ts = burst_start + timedelta(seconds=float(rng.uniform(0, 900)))
        device = rng.choice(attacker_devices)
        acc = rng.choice(target_accounts)
        lat, lon = jitter_geo(*CITIES[rng.choice(CITY_NAMES)], km_std=1)
        amount = float(np.round(rng.uniform(1, 49), 2))
        status = "success" if rng.random() < 0.06 else "failed"  # low success = validating stolen cards
        txns.append(_txn_row(
            acc, ts, device, new_ip(ip_prefix), lat, lon, amount,
            is_new_device=True, is_new_geo=True, status=status,
            label="card_testing",
        ))
    return txns


def gen_stealthy_card_testing_attack(attack_id):
    """A patient card-testing attack designed to evade velocity thresholds:
    spread across a much longer window (2-4 hours vs. 15 minutes), a larger
    device/IP pool (mimics rotating proxies), higher and more plausible
    amounts, and a decline rate closer to genuine failed-payment noise. Every
    single-window and single-device velocity feature this detector relies on
    is deliberately weaker here -- this is the adversarial version."""
    txns = []
    ip_prefixes = [f"{rng.integers(10, 223)}.{rng.integers(0, 255)}" for _ in range(int(rng.integers(4, 8)))]
    attacker_devices = [new_device_id() for _ in range(int(rng.integers(6, 12)))]
    burst_start = rand_timestamp(SIM_START, SIM_END - timedelta(hours=4))
    n = int(rng.integers(50, 100))
    target_accounts = [f"acc_synthetic_stealth_{attack_id}_{i}" for i in range(int(rng.integers(20, 45)))]
    for _ in range(n):
        ts = burst_start + timedelta(seconds=float(rng.uniform(0, 4 * 3600)))  # spread over hours
        device = rng.choice(attacker_devices)
        ipp = rng.choice(ip_prefixes)
        acc = rng.choice(target_accounts)
        lat, lon = jitter_geo(*CITIES[rng.choice(CITY_NAMES)], km_std=2)
        amount = float(np.round(rng.uniform(50, 400), 2))  # closer to real purchase amounts
        status = "success" if rng.random() < 0.35 else "failed"  # less extreme decline rate
        txns.append(_txn_row(
            acc, ts, device, new_ip(ipp), lat, lon, amount,
            is_new_device=True, is_new_geo=True, status=status,
            label="card_testing_stealthy",
        ))
    return txns


def gen_new_legitimate_account_transactions(account):
    """A genuinely new, legitimate account: someone who just signed up and
    is making their first few purchases. Exists specifically so the model
    can't use 'brand new account' as a clean proxy for fraud -- without
    this, card-testing and ring accounts (which ARE always brand new by
    construction) are trivially separable by account age alone, and a
    model trained on that shortcut ends up barely using genuine behavioral
    signals for anything else. See FAILURES.md."""
    txns = []
    device = new_device_id()
    signup_time = account["created_at"]
    n_txn = int(rng.integers(1, 6))
    for i in range(n_txn):
        ts = signup_time + timedelta(hours=float(rng.uniform(0, 48 * (i + 1))))
        if ts > SIM_END:
            break
        lat, lon = jitter_geo(account["home_lat"], account["home_lon"], km_std=6)
        amount = float(np.round(rng.lognormal(mean=6.3, sigma=0.8), 2))
        amount = min(amount, 20000)
        txns.append(_txn_row(
            account["account_id"], ts, device, new_ip(), lat, lon, amount,
            is_new_device=(i == 0), is_new_geo=False,
            status="success" if rng.random() > 0.03 else "failed",
            label="normal_new_account",
        ))
    return txns


def gen_ato_event(account):
    """Take an existing normal account and inject a takeover: new device, new
    (often distant/implausible) geo, followed quickly by a high-value txn that
    is anomalous vs. the account's own normal spend."""
    txns = []
    takeover_city = rng.choice([c for c in CITY_NAMES if c != account["home_city"]])
    lat, lon = jitter_geo(*CITIES[takeover_city], km_std=3)
    device = new_device_id()
    ts = rand_timestamp(SIM_START + timedelta(days=5), SIM_END)
    amount = float(np.round(rng.uniform(15000, 48000), 2))  # unusually high vs normal lognormal spend
    txns.append(_txn_row(
        account["account_id"], ts, device, new_ip(), lat, lon, amount,
        is_new_device=True, is_new_geo=True, status="success",
        label="ato",
    ))
    # frequently a second, smaller "test" transaction moments before
    ts2 = ts - timedelta(minutes=float(rng.uniform(2, 20)))
    amount2 = float(np.round(rng.uniform(10, 200), 2))
    txns.append(_txn_row(
        account["account_id"], ts2, device, new_ip(), lat, lon, amount2,
        is_new_device=True, is_new_geo=True, status="success",
        label="ato",
    ))
    return txns


def gen_stealthy_ato_event(account):
    """A more cautious takeover: attacker stays in the account's home city
    (no geo-jump signal) and keeps the amount within a plausible range for
    this account (no amount z-score spike). The ONLY signal is a new device.
    This is deliberately hard -- it's designed to evade exactly the loud
    features gen_ato_event triggers, mirroring how real adaptive fraud tries
    to stay under known detection thresholds rather than announce itself."""
    txns = []
    device = new_device_id()
    ts = rand_timestamp(SIM_START + timedelta(days=5), SIM_END)
    lat, lon = jitter_geo(account["home_lat"], account["home_lon"], km_std=6)  # same city, no geo signal
    amount = float(np.round(rng.lognormal(mean=6.6, sigma=0.7), 2))  # ordinary spend range, no z-score spike
    amount = min(amount, 6000)
    txns.append(_txn_row(
        account["account_id"], ts, device, new_ip(), lat, lon, amount,
        is_new_device=True, is_new_geo=False, status="success",
        label="ato_stealthy",
    ))
    return txns


def gen_abuse_ring(ring_id, n_members_range=(10, 30)):
    """Coordinated bot-driven ring: many freshly-created accounts sharing a
    small pool of devices / a tight IP range, transacting in synchronized
    bursts across several days with similar amounts (structuring-like)."""
    n_members = int(rng.integers(*n_members_range))
    shared_devices = [new_device_id() for _ in range(int(rng.integers(2, 5)))]
    ip_prefix = f"{rng.integers(10, 223)}.{rng.integers(0, 255)}"
    ring_city = rng.choice(CITY_NAMES)
    created_anchor = rand_timestamp(SIM_START, SIM_START + timedelta(days=5))

    accounts = []
    for i in range(n_members):
        created = created_anchor + timedelta(minutes=float(rng.uniform(0, 240)))  # created in a tight window
        lat, lon = jitter_geo(*CITIES[ring_city], km_std=2)
        accounts.append({
            "account_id": f"acc_ring{ring_id}_{i}_{uuid.uuid4().hex[:6]}",
            "created_at": created,
            "home_city": ring_city,
            "home_lat": lat,
            "home_lon": lon,
            "kyc_status": rng.choice(["verified", "pending"], p=[0.4, 0.6]),
            "account_type": "individual",
            "ring_id": f"ring_{ring_id}",
            "label_class": "ring",
        })

    txns = []
    n_bursts = int(rng.integers(2, 5))
    for b in range(n_bursts):
        burst_time = rand_timestamp(SIM_START, SIM_END - timedelta(hours=1))
        base_amount = float(rng.uniform(200, 900))  # similar amounts across the ring in a burst
        participating = rng.choice(accounts, size=int(rng.integers(5, len(accounts) + 1)), replace=False)
        for acc in participating:
            ts = burst_time + timedelta(seconds=float(rng.uniform(0, 120)))  # tight synchronization
            device = rng.choice(shared_devices)
            amount = float(np.round(base_amount + rng.normal(0, 30), 2))
            lat, lon = jitter_geo(*CITIES[ring_city], km_std=2)
            txns.append(_txn_row(
                acc["account_id"], ts, device, new_ip(ip_prefix), lat, lon, amount,
                is_new_device=False, is_new_geo=False, status="success",
                label="ring",
            ))
    return accounts, txns


def _txn_row(account_id, ts, device_id, ip, lat, lon, amount, is_new_device, is_new_geo, status, label):
    return {
        "transaction_id": f"txn_{uuid.uuid4().hex[:12]}",
        "account_id": account_id,
        "timestamp": ts,
        "device_id": device_id,
        "ip_address": ip,
        "geo_lat": round(lat, 5),
        "geo_lon": round(lon, 5),
        "amount": amount,
        "merchant_category": rng.choice(MERCHANT_CATEGORIES),
        "is_new_device": is_new_device,
        "is_new_geo": is_new_geo,
        "status": status,
        "label": label,  # ground truth, NOT a feature the model sees at inference
    }


# ---------------------------------------------------------------------------
# Assemble full dataset
# ---------------------------------------------------------------------------

def build_dataset(
    n_normal_accounts=450,
    n_new_signup_accounts=80,
    n_ato_events=15,
    n_stealthy_ato_events=15,
    n_card_testing_attacks=3,
    n_stealthy_card_testing_attacks=2,
    n_abuse_rings=3,
    n_ambiguous_each=6,
):
    accounts = gen_normal_accounts(n_normal_accounts)
    all_txns = []

    for acc in accounts:
        all_txns.extend(gen_normal_transactions(acc))

    # Genuinely new, legitimate accounts -- see gen_new_signup_accounts'
    # docstring. These exist specifically so account age isn't a clean
    # fraud shortcut for the model to latch onto.
    new_signup_accounts = gen_new_signup_accounts(n_new_signup_accounts)
    for acc in new_signup_accounts:
        all_txns.extend(gen_new_legitimate_account_transactions(acc))
    accounts.extend(new_signup_accounts)

    # ATO: pick real accounts from the normal pool and inject a takeover.
    # Half loud (obvious), half stealthy (deliberately hard) -- drawn from
    # disjoint samples so the same account never gets both.
    ato_pool = list(rng.choice(accounts, size=n_ato_events + n_stealthy_ato_events, replace=False))
    loud_ato_targets = ato_pool[:n_ato_events]
    stealthy_ato_targets = ato_pool[n_ato_events:]
    for acc in loud_ato_targets:
        all_txns.extend(gen_ato_event(acc))
    for acc in stealthy_ato_targets:
        all_txns.extend(gen_stealthy_ato_event(acc))

    # card testing bursts: loud + stealthy variants
    for i in range(n_card_testing_attacks):
        all_txns.extend(gen_card_testing_attack(i))
    for i in range(n_stealthy_card_testing_attacks):
        all_txns.extend(gen_stealthy_card_testing_attack(i))

    # abuse rings
    ring_accounts_all = []
    for r in range(n_abuse_rings):
        ring_accounts, ring_txns = gen_abuse_ring(r)
        ring_accounts_all.extend(ring_accounts)
        all_txns.extend(ring_txns)
    accounts.extend(ring_accounts_all)

    # ambiguous legit hard cases -- drawn from a few dedicated fresh accounts
    ambiguous_kinds = ["payroll_batch", "festival_spree", "genuine_travel"]
    for kind in ambiguous_kinds:
        sample_accs = gen_normal_accounts(n_ambiguous_each)
        for acc in sample_accs:
            acc["label_class"] = f"ambiguous_{kind}"
            all_txns.extend(gen_ambiguous_legit_transactions(acc, kind))
        accounts.extend(sample_accs)

    accounts_df = pd.DataFrame(accounts)
    txns_df = pd.DataFrame(all_txns).sort_values("timestamp").reset_index(drop=True)
    return accounts_df, txns_df


def summarize(accounts_df, txns_df):
    print(f"Accounts: {len(accounts_df)}")
    print(f"Transactions: {len(txns_df)}")
    print("\nLabel distribution (transactions):")
    print(txns_df["label"].value_counts())
    print(f"\nDate range: {txns_df['timestamp'].min()} -> {txns_df['timestamp'].max()}")
    print(f"Unique devices: {txns_df['device_id'].nunique()}")
    print(f"Unique accounts in txns: {txns_df['account_id'].nunique()}")
    fraud_labels = {"card_testing", "card_testing_stealthy", "ato", "ato_stealthy", "ring"}
    ambiguous_labels = [l for l in txns_df["label"].unique() if str(l).startswith("ambiguous")]
    fraud_rate = txns_df["label"].isin(fraud_labels).mean()
    ambiguous_rate = txns_df["label"].isin(ambiguous_labels).mean()
    print(f"\nFraud-labeled rate: {fraud_rate:.2%}")
    print(f"Ambiguous-legit (hard negative) rate: {ambiguous_rate:.2%}")
    print(
        "\nNote: real-world payment fraud base rates are typically well under 1%. "
        "This dataset intentionally oversamples fraud and hard-ambiguous cases "
        "relative to that so the detector has enough positive examples to learn "
        "from -- a standard, disclosed practice in fraud-ML benchmarking, not a "
        "claim that this ratio reflects production traffic."
    )


if __name__ == "__main__":
    accounts_df, txns_df = build_dataset()
    summarize(accounts_df, txns_df)

    accounts_df.to_csv(DATA_DIR / "accounts.csv", index=False)
    txns_df.to_csv(DATA_DIR / "transactions.csv", index=False)
    print("\nSaved to data/accounts.csv and data/transactions.csv")
