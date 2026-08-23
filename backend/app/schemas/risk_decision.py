"""
The contract the LLM reasoning layer must satisfy.

This is the "zero-trust" boundary: the LLM's raw output is treated as
untrusted text until it parses into exactly this shape. If it doesn't,
the caller must not use it -- see reasoning_agent.py's fallback behavior.

Important distinction: `action` here is a RECOMMENDATION, not an executed
command. The triage gate (core/triage_gate.py) NEVER auto-executes "block"
-- a block recommendation always routes to human review with this evidence
attached. This is what makes the system defense-only and un-abusable even
if the LLM's reasoning is wrong or manipulated.
"""

from pydantic import BaseModel, Field, field_validator


class RiskDecision(BaseModel):
    action: str = Field(description="One of: allow, review, block. A recommendation only.")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(min_length=1, max_length=6)
    rationale: str = Field(min_length=10, max_length=600)

    @field_validator("action")
    @classmethod
    def action_must_be_valid(cls, v: str) -> str:
        allowed = {"allow", "review", "block"}
        if v not in allowed:
            raise ValueError(f"action must be one of {allowed}, got {v!r}")
        return v

    @field_validator("evidence")
    @classmethod
    def evidence_not_empty_strings(cls, v: list[str]) -> list[str]:
        cleaned = [e.strip() for e in v if e.strip()]
        if not cleaned:
            raise ValueError("evidence list cannot be all-empty")
        return cleaned
