"""
Audit ledger for Sentinel.

Every decision the system makes gets one permanent, append-only entry. The
entries are hash-chained -- each entry's hash is computed over its own
content PLUS the previous entry's hash, the same principle git commits and
blockchains use. This means editing or deleting a past entry breaks the
chain from that point forward: verify_chain() will catch it. This is what
"tamper-evident" means here -- not that the file can't be edited (it's a
plain JSONL file, anyone with disk access can edit it), but that editing it
is DETECTABLE, which is the actual property an audit trail needs.

This directly targets the real complaint pattern found in research: a
merchant whose account was frozen with a vague "ToS violation" and no
evidence ever shown. Every entry here has the full evidence list, the risk
score, which path the decision took, and is independently verifiable.
"""

import json
import hashlib
import os
import threading
from datetime import datetime, timezone
from app.core.paths import DATA_DIR

LEDGER_PATH = str(DATA_DIR / "audit_ledger.jsonl")
GENESIS_HASH = "0" * 64

# The chain is only valid if entries are appended strictly one at a time --
# each entry's prev_hash must be the actual previous entry's hash, not a
# hash that TWO entries both read before either finished writing. Found
# this the hard way: 60 concurrent requests all succeeded individually (200
# OK, unique transaction IDs), but the chain came out broken -- several
# hash slots were claimed by 2-3 entries at once, because the read-last-hash
# and append-new-entry steps weren't atomic together. A single process-wide
# lock around that critical section is the correct fix here: FastAPI runs
# sync `def` endpoints in a thread pool within one process (not multiple
# processes), so a threading.Lock genuinely serializes every writer.
_ledger_lock = threading.Lock()


def _hash_entry(prev_hash: str, entry: dict) -> str:
    payload = json.dumps(entry, sort_keys=True, default=str) + prev_hash
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_last_hash(path: str = LEDGER_PATH) -> str:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return GENESIS_HASH
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell() - 1
        chunk = b""
        while pos >= 0:
            f.seek(pos)
            char = f.read(1)
            if char == b"\n" and chunk:
                break
            chunk = char + chunk
            pos -= 1
    if not chunk.strip():
        return GENESIS_HASH
    last_entry = json.loads(chunk)
    return last_entry["entry_hash"]


def log_decision(entry: dict, path: str = LEDGER_PATH) -> dict:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _ledger_lock:
        prev_hash = _get_last_hash(path)
        record = {
            **entry,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "prev_hash": prev_hash,
        }
        record["entry_hash"] = _hash_entry(prev_hash, record)
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    return record


def verify_chain(path: str = LEDGER_PATH) -> dict:
    """Walks the whole ledger and recomputes every hash. Returns whether the
    chain is intact, and the index of the first broken entry if not."""
    if not os.path.exists(path):
        return {"intact": True, "n_entries": 0, "broken_at": None}

    prev_hash = GENESIS_HASH
    n = 0
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            claimed_hash = entry.pop("entry_hash")
            if entry.get("prev_hash") != prev_hash:
                return {"intact": False, "n_entries": n, "broken_at": i,
                         "reason": "prev_hash does not match preceding entry"}
            recomputed = _hash_entry(prev_hash, entry)
            if recomputed != claimed_hash:
                return {"intact": False, "n_entries": n, "broken_at": i,
                         "reason": "entry_hash does not match recomputed hash -- entry was likely edited"}
            prev_hash = claimed_hash
            n += 1
    return {"intact": True, "n_entries": n, "broken_at": None}


if __name__ == "__main__":
    # fresh ledger for this demo
    if os.path.exists(LEDGER_PATH):
        os.remove(LEDGER_PATH)

    print("=== Logging 3 sample decisions ===")
    log_decision({"transaction_id": "txn_001", "path": "auto_allow", "risk_score": 0.12,
                  "final_status": "allowed", "evidence": ["low risk"]})
    log_decision({"transaction_id": "txn_002", "path": "llm_reasoned", "risk_score": 0.61,
                  "final_status": "flagged_for_review", "evidence": ["new device", "unusual hour"]})
    log_decision({"transaction_id": "txn_003", "path": "auto_flag_obvious", "risk_score": 0.97,
                  "final_status": "flagged_for_review", "evidence": ["impossible travel speed"]})

    result = verify_chain()
    print(f"\nChain verification: {result}")
    assert result["intact"] and result["n_entries"] == 3

    print("\n=== Simulating tampering: editing entry 2's risk_score after the fact ===")
    with open(LEDGER_PATH, "r") as f:
        lines = f.readlines()
    tampered = json.loads(lines[1])
    tampered["risk_score"] = 0.05  # attacker tries to quietly downgrade the risk score
    lines[1] = json.dumps(tampered) + "\n"
    with open(LEDGER_PATH, "w") as f:
        f.writelines(lines)

    result = verify_chain()
    print(f"Chain verification after tampering: {result}")
    assert not result["intact"], "tampering was NOT detected -- this would be a real bug"
    print("PASS: tampering was correctly detected")
