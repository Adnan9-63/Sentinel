"""
Triage gate for Sentinel.

Deliberately does NOT send every transaction through the LLM. This is a
design choice, not a limitation -- it's the concrete answer to "does this
system know when NOT to use AI":

  score < LOW_THRESHOLD   -> auto-allow, log only. No LLM call.
  LOW <= score < HIGH     -> genuinely ambiguous. Call the LLM reasoning
                              layer for a structured, evidence-backed
                              explanation.
  score >= HIGH_THRESHOLD -> obvious enough that an LLM call adds latency
                              and cost without adding confidence. Auto-flag
                              for human review with a deterministic evidence
                              summary instead.

Thresholds are not arbitrary round numbers -- they come directly from the
score distribution measured in ml_detection.py: normal transactions in the
held-out test set topped out at risk_score 0.378, and 75% of real fraud
scored above 0.823. LOW=0.40 and HIGH=0.85 sit just outside that observed
range on each side, so the "ambiguous middle band" routed to the LLM is
the genuinely uncertain zone, not a guess.

CRITICAL: regardless of path, "block" is NEVER auto-executed. The most
this gate ever does automatically is flag something for human review. An
account is never frozen by this system without a human confirming it.
"""

from dataclasses import dataclass, field

from app.agents.reasoning_agent import get_risk_decision, call_llm_live
from app.schemas.risk_decision import RiskDecision

LOW_THRESHOLD = 0.40
HIGH_THRESHOLD = 0.85


@dataclass
class TriageResult:
    path: str  # "auto_allow" | "llm_reasoned" | "auto_flag_obvious"
    risk_score: float
    final_status: str  # "allowed" | "flagged_for_review" -- never "blocked"
    decision: RiskDecision | None = None
    evidence_summary: list[str] = field(default_factory=list)


def _deterministic_evidence(feature_context: dict) -> list[str]:
    """For obvious cases that skip the LLM, still produce a concrete,
    specific evidence list -- 'high risk score' alone is not an
    explanation a human reviewer or a merchant can act on."""
    evidence = []
    if feature_context.get("implied_speed_kmh", 0) > 1000:
        evidence.append(
            f"implied travel speed of {feature_context['implied_speed_kmh']:,.0f} km/h "
            "is physically impossible"
        )
    if feature_context.get("amount_zscore", 0) > 5:
        evidence.append(
            f"transaction amount is {feature_context['amount_zscore']:.1f} standard "
            "deviations above this account's normal spend"
        )
    if feature_context.get("device_velocity_1h", 0) > 10:
        evidence.append(
            f"{feature_context['device_velocity_1h']:.0f} transactions from this device "
            "in the past hour, across accounts"
        )
    if feature_context.get("ip_velocity_1h", 0) > 10:
        evidence.append(
            f"{feature_context['ip_velocity_1h']:.0f} transactions from this network "
            "range in the past hour, across accounts"
        )
    if feature_context.get("is_new_device_for_account") and feature_context.get("is_new_geo"):
        evidence.append("new device AND new location for this account in the same transaction")
    if not evidence:
        evidence.append("composite risk score exceeded the high-confidence threshold")
    return evidence


def triage(
    risk_score: float,
    feature_context: dict,
    ring_context: dict | None = None,
    llm_caller=call_llm_live,
) -> TriageResult:
    if risk_score < LOW_THRESHOLD:
        return TriageResult(
            path="auto_allow",
            risk_score=risk_score,
            final_status="allowed",
            evidence_summary=["risk score below auto-allow threshold, no anomalous signals"],
        )

    if risk_score < HIGH_THRESHOLD:
        decision = get_risk_decision(feature_context, ring_context, llm_caller=llm_caller)
        final_status = "allowed" if decision.action == "allow" else "flagged_for_review"
        return TriageResult(
            path="llm_reasoned",
            risk_score=risk_score,
            final_status=final_status,
            decision=decision,
            evidence_summary=decision.evidence,
        )

    # score >= HIGH_THRESHOLD: obvious enough to skip the LLM entirely
    evidence = _deterministic_evidence(feature_context)
    return TriageResult(
        path="auto_flag_obvious",
        risk_score=risk_score,
        final_status="flagged_for_review",
        evidence_summary=evidence,
    )
