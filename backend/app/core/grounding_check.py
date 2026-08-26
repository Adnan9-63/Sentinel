"""
Grounding check for Sentinel's reasoning layer.

The Pydantic schema (risk_decision.py) guarantees the LLM's output has the
right SHAPE. It says nothing about whether the CONTENT is true. A model
could return a perfectly well-formed RiskDecision whose evidence cites a
number that was never in the input at all -- a fabricated claim, not a
malformed one. Schema validation would never catch that.

This is a lightweight, heuristic check, not a proof of factual correctness
-- flagged as informational (attached to the record for the human reviewer
and the audit log), not used to reject or override the decision. That's a
deliberate choice: the check has real false-positive potential (a model
might reasonably say "roughly 20" when the real value is 22, or describe a
ratio that isn't literally one input value), so treating a mismatch as a
hard failure would create more noise than signal. Its job is to surface
"this specific claim doesn't match anything in the input" for a human to
weigh, the same way Razorpay's own stated Agent Studio principle describes
"out-of-scope behavior detection" as a monitoring layer, not just a gate.

Run:
    python -m app.core.grounding_check
"""

import re

# Numbers this small are almost always incidental to natural language
# ("one new device", "a single transaction") rather than a specific data
# claim worth checking -- checking them creates false positives without
# adding real signal.
MIN_MAGNITUDE_TO_CHECK = 10.0

RELATIVE_TOLERANCE = 0.15  # 15% -- a model rounding 22 to "about 20" is fine


def _extract_numbers(text: str) -> list[float]:
    # matches things like 1,234.5 or 22.3 or 95000 or 0.82
    raw = re.findall(r"-?\d[\d,]*\.?\d*", text)
    numbers = []
    for r in raw:
        cleaned = r.replace(",", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return [n for n in numbers if abs(n) >= MIN_MAGNITUDE_TO_CHECK]


def _flatten_reference_values(feature_context: dict, ring_context: dict | None) -> list[float]:
    values = []
    for v in feature_context.values():
        if isinstance(v, (int, float)):
            values.append(float(v))
    if ring_context:
        for v in ring_context.values():
            if isinstance(v, (int, float)):
                values.append(float(v))
    return values


def _is_grounded(claimed: float, reference_values: list[float]) -> bool:
    for ref in reference_values:
        if ref == 0:
            continue
        if abs(claimed - ref) <= RELATIVE_TOLERANCE * max(abs(ref), abs(claimed)):
            return True
        # also allow rounding to a clean multiple (e.g. model says "90000"
        # for a raw implied_speed_kmh of 94823.6)
        if abs(ref) >= 1000 and abs(claimed - ref) / max(abs(ref), 1) <= 0.1:
            return True
    return False


def check_grounding(decision_evidence: list[str], decision_rationale: str,
                     feature_context: dict, ring_context: dict | None = None) -> list[str]:
    """Returns a list of human-readable warnings for any specific number
    mentioned in the LLM's evidence/rationale that doesn't correspond to
    anything in the actual input data. Empty list = nothing flagged."""
    text = " ".join(decision_evidence) + " " + (decision_rationale or "")
    claimed_numbers = _extract_numbers(text)
    reference_values = _flatten_reference_values(feature_context, ring_context)

    warnings = []
    for claimed in claimed_numbers:
        if not _is_grounded(claimed, reference_values):
            warnings.append(
                f"evidence cites {claimed:g}, which does not closely match any "
                f"value actually present in the input data -- possibly fabricated"
            )
    return warnings


if __name__ == "__main__":
    feature_context = {
        "device_velocity_1h": 22, "amount_zscore": 4.1,
        "implied_speed_kmh": 94823.6, "account_age_days": 210,
    }

    print("=== Well-grounded evidence (should produce NO warnings) ===")
    good_evidence = [
        "device velocity of 22 transactions/hour is far above normal",
        "implied travel speed of roughly 95000 km/h is physically impossible",
    ]
    warnings = check_grounding(good_evidence, "Multiple signals point to takeover.", feature_context)
    print(f"Warnings: {warnings}")
    assert warnings == [], "well-grounded evidence should not be flagged"
    print("PASS")

    print("\n=== Fabricated evidence (should be flagged) ===")
    bad_evidence = [
        "device velocity of 500 transactions/hour is extremely suspicious",
        "this account has been flagged 47 times before",
    ]
    warnings = check_grounding(bad_evidence, "History of repeated fraud.", feature_context)
    print(f"Warnings: {warnings}")
    assert len(warnings) == 2, f"expected 2 fabricated numbers flagged, got {len(warnings)}"
    print("PASS: both fabricated numbers (500, 47) correctly flagged, "
          "real numbers (22, ~95000) correctly NOT flagged in the same test")
