# Sentinel — Coordinated Abuse-Ring & Fraud-Spike Detector

Built for the Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager.

## The problem

Razorpay's own framing for this track names "AI-enabled fraud hitting Indian BFSI"
as the reason it exists, and lists "Abuse-ring sentinel" as an example direction.
Independently, real merchant complaints (Trustpilot, consumer complaint boards)
show a recurring failure pattern: accounts get frozen for vague reasons with zero
evidence given, funds held for months, no audit trail the merchant can see.

Most fraud tools solve half the problem (catch bad transactions) and ignore the
other half (don't wrongly block good ones, and explain every decision). Sentinel
is built around both halves.

## What Sentinel does

1. **Detects coordinated fraud rings**, not just single suspicious transactions —
   bot-driven / AI-orchestrated attack clusters that share device fingerprints,
   IP proximity, and synchronized timing across nominally "unrelated" accounts.
2. **Scores individual transaction risk** using engineered behavioral features
   (velocity, geo-jump, amount z-score, device/card reuse).
3. **Routes only the ambiguous middle band** through a bounded LLM reasoning
   layer, which must output a structured evidence report — never a free-form
   decision. Obvious cases (very low or very high score) skip the LLM entirely
   — this is a deliberate design choice, not a limitation: it shows the system
   knows when NOT to spend an LLM call.
4. **Never auto-freezes an account outright.** High-risk clusters route to
   human review with the evidence trail attached. This directly targets the
   real complaint pattern found in research: accounts frozen with no
   explanation and no evidence.
5. **Reports the false-positive cost**, not just detection accuracy — the
   buildathon's own stated bar for this track.

## Architecture

```
Synthetic Transaction Stream
        |
Feature Engineering (deterministic)
  - velocity, geo-jump distance, amount z-score, device/card reuse,
    time-of-day anomaly
        |
Graph Layer (ring detection)
  - cluster accounts by shared device fingerprint / IP proximity /
    synchronized transaction timing -> ring_id, ring_risk_score
        |
ML Detection Layer
  - Isolation Forest (unsupervised) + Gradient Boosted Trees (supervised)
  - ensemble -> calibrated probability
        |
Triage Gate (deterministic thresholds)
  - score < T1            -> auto-allow, log only
  - T1 <= score < T2       -> LLM reasoning layer
  - score >= T2            -> auto-flag for human review (never auto-block)
        |
LLM Reasoning Layer (Pydantic-bounded)
  - input: feature vector + ring context
  - output: MUST match schema {action: allow|review|block, confidence,
    evidence: [...], rationale}
  - malformed output -> deterministic fallback to "review" (fail-safe)
        |
Audit Ledger (append-only)
  - every decision fully traceable: features -> ML score -> LLM output ->
    validation result -> final status -> human override (if any)
        |
Dashboard (React)
  - live feed, ring visualizations, confusion matrix, false-positive
    cost tradeoff slider
```

## Repo layout

```
sentinel/
  backend/
    app/
      api/        FastAPI routes
      core/        config, state machine / triage gate
      agents/      LLM reasoning layer, prompt + schema enforcement
      schemas/     Pydantic models (zero-trust validation)
      services/    synthetic data generator, feature engineering, graph clustering
    tests/         benchmark + adversarial test suite
  frontend/         React + Tailwind dashboard
  data/             generated synthetic datasets
  FAILURES.md       running log of real failures hit during build (for the
                     "what broke and how you got out" submission question)
```

## Build log / timeline

- Aug 22-23: repo scaffold, synthetic data generator, feature schema
- Aug 24-26: detection engine (graph clustering + ML ensemble), baseline metrics
- Aug 27-28: LLM reasoning layer + Pydantic gate + audit ledger
- Aug 29-31: off (MLH Fellowship)
- Sep 1-2: React dashboard
- Sep 3: adversarial testing (prompt injection, malformed payloads, concurrent load)
- Sep 4: README/architecture polish, write-up from FAILURES.md
- Sep 5: record demo video, submit
