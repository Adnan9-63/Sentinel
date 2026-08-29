"""
Rate limiting for the Sentinel API.

Per-client-IP limits on the endpoints that do real work -- feature
computation, ML inference, an LLM call, a ledger write. Without this,
anyone with network access to a running instance could hammer
/simulate/card_testing_burst in a loop and generate unbounded load and
audit-ledger writes for free. A fraud-prevention system whose own API has
no protection is a real, pointed weakness, not a hypothetical one.

HONEST SCOPE NOTE, not glossed over: this is IP-based rate limiting on a
single-process hackathon deployment. It stops naive scripted hammering
from one source. It does NOT provide authentication (no login, no API
keys), does NOT stop a distributed attack from many IPs, and does NOT
replace a real API gateway / WAF in front of a production deployment.
Deliberately not building full auth here: it would add real friction to
the exact demo flow this project needs to work smoothly for (the
dashboard hitting these endpoints live, unauthenticated, from a browser),
for a security property that matters far more in an internet-facing
production deployment than in a local hackathon demo. A real production
version would add: API keys or OAuth, a gateway-level WAF, and
distributed rate limiting (e.g. Redis-backed) instead of this
single-process in-memory limiter.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Reads are cheap and safe to allow more of. Simulate endpoints do real
# work (ML inference, an LLM call attempt, a ledger write) and are capped
# tighter. The burst endpoint is capped tightest since a single call
# already generates up to 50 transactions internally.
READ_LIMIT = "60/minute"
SIMULATE_LIMIT = "20/minute"
BURST_LIMIT = "10/minute"
