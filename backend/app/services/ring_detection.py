"""
Graph-based coordinated abuse-ring detection for Sentinel.

Point-in-time velocity features (see features.py) under-detect coordinated
rings, because each individual ring transaction can look close to normal in
isolation -- the signal only appears when you look at relationships BETWEEN
accounts over their full history: shared devices, shared IP ranges, and
synchronized timing. That's what this module does.

Approach:
  1. Build a graph where nodes are accounts and edges connect accounts that
     share a device_id or an IP prefix.
  2. Find connected components (clusters). A cluster of size 1-2 sharing a
     home wifi router is normal; a cluster of 15 accounts all sharing 3
     devices is not.
  3. Score each cluster on: size, device concentration (txns per unique
     device -- low = many accounts crowding onto few devices), and timing
     synchronization (how tightly transaction timestamps cluster into
     bursts, vs. spread randomly across the observation window).

Run:
    python -m app.services.ring_detection
"""

import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations


def build_account_graph(
    txns_df: pd.DataFrame,
    min_shared_events_by_type: dict = None,
) -> nx.Graph:
    """Edge between two accounts if they share a device_id or ip_prefix.

    Device-id sharing and IP-prefix sharing are NOT equally strong signals.
    device_id is an app/browser fingerprint (effectively a random UUID here)
    -- two distinct accounts using the exact same one is inherently suspicious
    even a single time. ip_prefix is much coarser (a /16-ish block covers a
    whole ISP region), so two unrelated accounts coincidentally sharing one
    happens by chance at real volume -- confirmed empirically: at 7.5k normal
    transactions, thousands of prefixes had 2-4 unrelated accounts collide on
    them purely by chance, and treating that as a signal at threshold=1 chained
    the whole account population into one giant false cluster. IP-prefix edges
    therefore need a much higher repeat-count bar than device edges.
    """
    if min_shared_events_by_type is None:
        min_shared_events_by_type = {"device_id": 1, "ip_prefix": 4}

    df = txns_df.copy()
    df["ip_prefix"] = df["ip_address"].apply(lambda ip: ".".join(ip.split(".")[:2]))

    G = nx.Graph()
    G.add_nodes_from(df["account_id"].unique())

    for key_col in ["device_id", "ip_prefix"]:
        groups = df.groupby(key_col)["account_id"].apply(lambda s: sorted(set(s)))
        for key, accs in groups.items():
            if len(accs) < 2 or len(accs) > 60:
                # a single account, or an entity so widely shared it's
                # uninformative (e.g. a very common IP prefix) -- skip
                continue
            for a, b in combinations(accs, 2):
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                    G[a][b]["shared_keys"].add((key_col, key))
                else:
                    G.add_edge(a, b, weight=1, shared_keys={(key_col, key)})

    # apply per-signal-type thresholds: an edge survives if EITHER it has
    # enough device-sharing events, OR enough ip_prefix-sharing events
    def edge_survives(data):
        by_type = {}
        for key_col, _ in data["shared_keys"]:
            by_type[key_col] = by_type.get(key_col, 0) + 1
        for key_col, count in by_type.items():
            if count >= min_shared_events_by_type.get(key_col, 1):
                return True
        return False

    weak_edges = [(u, v) for u, v, d in G.edges(data=True) if not edge_survives(d)]
    G.remove_edges_from(weak_edges)
    return G


def timing_synchronization_score(timestamps: pd.Series) -> float:
    """0-1 score: how tightly transactions cluster into bursts vs. spread
    evenly across the observation window. Computed as 1 - (normalized
    entropy of the inter-arrival gap distribution binned into hours).
    High score = many transactions crammed into a few tight windows."""
    if len(timestamps) < 3:
        return 0.0
    ts_sorted = timestamps.sort_values()
    span_hours = max((ts_sorted.iloc[-1] - ts_sorted.iloc[0]).total_seconds() / 3600.0, 1.0)
    # bin into hourly buckets relative to the cluster's own span
    hour_bins = ((ts_sorted - ts_sorted.iloc[0]).dt.total_seconds() / 3600.0).astype(int)
    counts = hour_bins.value_counts(normalize=True)
    entropy = -(counts * np.log(counts + 1e-12)).sum()
    max_entropy = np.log(max(len(counts), 2))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return float(1.0 - normalized_entropy)


def score_clusters(G: nx.Graph, txns_df: pd.DataFrame, min_cluster_size: int = 4) -> pd.DataFrame:
    df = txns_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    rows = []
    for cluster_id, component in enumerate(nx.connected_components(G)):
        if len(component) < min_cluster_size:
            continue
        cluster_txns = df[df["account_id"].isin(component)]
        n_devices = cluster_txns["device_id"].nunique()
        n_accounts = len(component)
        n_txns = len(cluster_txns)
        device_concentration = n_accounts / max(n_devices, 1)  # accounts per device; high = suspicious
        sync_score = timing_synchronization_score(cluster_txns["timestamp"])
        median_amount = float(cluster_txns["amount"].median())
        decline_rate = float((cluster_txns["status"] == "failed").mean())
        n_distinct_days = cluster_txns["timestamp"].dt.date.nunique()

        # Card-testing bursts and coordinated rings produce the SAME graph
        # shape (many accounts, few shared devices) -- device concentration
        # and timing sync alone can't tell them apart, and conflating them
        # made card-testing bursts drown out real rings in early testing.
        # What actually distinguishes them: card-testing is tiny amounts,
        # high decline rate, everything crammed into one narrow window.
        # A coordinated ring transacts successfully, at plausible amounts,
        # in repeated bursts spread across multiple days (structuring).
        looks_like_card_testing = (median_amount < 100) and (decline_rate > 0.5) and (n_distinct_days <= 1)

        if looks_like_card_testing:
            cluster_type = "card_testing_burst"
        else:
            cluster_type = "coordinated_ring"

        # composite ring risk score, weights chosen to be interpretable, not tuned/overfit
        ring_risk_score = float(np.clip(
            0.35 * min(device_concentration / 8.0, 1.0)
            + 0.30 * sync_score
            + 0.15 * min(n_accounts / 20.0, 1.0)
            + 0.20 * min(n_distinct_days / 3.0, 1.0),  # rings repeat across days; single bursts don't
            0, 1
        ))
        rows.append({
            "cluster_id": cluster_id,
            "cluster_type": cluster_type,
            "n_accounts": n_accounts,
            "n_devices": n_devices,
            "n_transactions": n_txns,
            "device_concentration": round(device_concentration, 2),
            "timing_sync_score": round(sync_score, 3),
            "median_amount": round(median_amount, 2),
            "decline_rate": round(decline_rate, 3),
            "n_distinct_days": n_distinct_days,
            "ring_risk_score": round(ring_risk_score, 3),
            "account_ids": sorted(component),
        })

    result = pd.DataFrame(rows).sort_values("ring_risk_score", ascending=False).reset_index(drop=True)
    return result


def evaluate_against_ground_truth(clusters_df: pd.DataFrame, accounts_df: pd.DataFrame, threshold: float = 0.5):
    """Evaluate ring detection and card-testing detection SEPARATELY against
    their own ground truth, since they're different attack types that happen
    to produce similar graph shapes."""
    true_ring_accounts = set(accounts_df.loc[accounts_df["ring_id"].notna(), "account_id"])

    flagged = clusters_df[clusters_df["ring_risk_score"] >= threshold]
    if flagged.empty:
        print(f"No clusters scored >= {threshold}. Lower the threshold or check cluster construction.")
        return

    for target_type, true_accounts, label in [
        ("coordinated_ring", true_ring_accounts, "Coordinated abuse rings"),
        ("card_testing_burst", None, "Card-testing bursts"),
    ]:
        subset = flagged[flagged["cluster_type"] == target_type]
        flagged_accounts = set()
        for accs in subset["account_ids"]:
            flagged_accounts.update(accs)

        print(f"\n{label}: {len(subset)} clusters flagged, {len(flagged_accounts)} accounts swept in")

        if true_accounts is not None:
            tp = flagged_accounts & true_accounts
            fp = flagged_accounts - true_accounts
            missed = true_accounts - flagged_accounts
            precision = len(tp) / max(len(flagged_accounts), 1)
            recall = len(tp) / max(len(true_accounts), 1)
            print(f"  True members correctly caught: {len(tp)} / {len(true_accounts)} planted (recall {recall:.1%})")
            print(f"  Wrongly swept in (false positives): {len(fp)} (precision {precision:.1%})")
            if missed:
                print(f"  Missed entirely: {sorted(missed)[:5]}{'...' if len(missed) > 5 else ''}")
        else:
            n_synthetic = sum(1 for a in flagged_accounts if str(a).startswith("acc_synthetic"))
            n_other = len(flagged_accounts) - n_synthetic
            print(f"  Of these, {n_synthetic} are known card-testing target accounts, "
                  f"{n_other} are other accounts swept in alongside them")


if __name__ == "__main__":
    txns = pd.read_csv("/home/claude/sentinel/data/transactions.csv")
    accounts = pd.read_csv("/home/claude/sentinel/data/accounts.csv")

    G = build_account_graph(txns)
    print(f"Graph: {G.number_of_nodes()} accounts, {G.number_of_edges()} edges")

    clusters_df = score_clusters(G, txns)
    print(f"\nClusters found (size >= 4): {len(clusters_df)}")
    print(clusters_df[["cluster_id", "cluster_type", "n_accounts", "n_devices", "device_concentration",
                        "timing_sync_score", "n_distinct_days", "ring_risk_score"]].head(15).to_string(index=False))

    print("\n--- Evaluation against ground truth ---")
    # Operating threshold set at 0.3, not 0.5: a full sweep (see FAILURES.md)
    # showed precision stays at 100% across every threshold from 0.1-0.5 --
    # graph fragments are always pure ring members, never mixed with real
    # accounts. Below 0.3 recall plateaus (69.6%), so there's no benefit to
    # going lower; above it, recall drops fast for zero precision gain.
    evaluate_against_ground_truth(clusters_df, accounts, threshold=0.3)

    clusters_df.drop(columns=["account_ids"]).to_csv(
        "/home/claude/sentinel/data/ring_clusters_summary.csv", index=False
    )
    print("\nSaved to data/ring_clusters_summary.csv")
