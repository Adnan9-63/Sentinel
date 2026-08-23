# FAILURES.md — real problems hit during the build, logged as they happen

Rules for this file:
- Log the failure the day it happens, not reconstructed later from memory.
- Include what broke, why, and the actual fix — not a sanitized version.
- No entry gets deleted or prettied up after the fact.

---

## [Aug 22] Setup

Scaffolded repo, wrote synthetic data generator covering five populations:
normal accounts, card-testing bursts, ATO takeovers, coordinated abuse rings,
and three kinds of deliberately ambiguous "hard negative" legit behavior
(payroll batch, festival spree, genuine travel).

**What broke:** first generator run produced a 25.85% fraud-labeled rate and
the summary function printed a comment claiming it was a "realistic imbalance,
not inflated." That was false — real payment fraud base rates are well under
1%, and the print statement was asserting something the code hadn't earned.
Caught it by actually reading the printed distribution instead of assuming the
generator was correct because it ran without errors.

**Fix:** rebalanced generation parameters (more normal accounts/transactions,
smaller card-testing and ring burst sizes) to bring the fraud rate down to
~5.2%, and rewrote the summary to state plainly that this still oversamples
fraud relative to real-world rates on purpose (to give the detector enough
positive examples to learn from) rather than claiming realism it doesn't have.

Lesson: a script running cleanly and printing a confident claim are two
different things — check the claim against the actual numbers, not just
against whether the code executed.

---

## [Aug 22] Graph-based ring detection — three real bugs, in order

Built the account graph (shared device / IP-prefix edges -> connected
components -> risk score) and ran it against the planted ground truth
(3 rings, 56 accounts total) instead of just eyeballing the output.

**Bug 1 — card-testing and rings are graph-structurally identical.**
First evaluation run: the three highest-scoring clusters (risk score 1.0,
sizes 34-36) had ZERO real ring members. They were card-testing bot target
accounts. Card-testing bursts and coordinated rings both produce "many
accounts, few shared devices" — the exact shape the graph layer looks for —
so a single undifferentiated score conflated two different attack types, and
the louder one (single-window, high device concentration) buried the ring
signal entirely. Recall on real rings was 0%.

Fix: added decline_rate, median_amount, and n_distinct_days as cluster
features, and split scoring into two cluster types -- card_testing_burst
(tiny amounts, high decline rate, one-day window) vs coordinated_ring
(successful transactions, spread across multiple days -- structuring).
This wasn't in the original design; the evaluation is what surfaced that two
attack types needed separate treatment, not one.

**Bug 2 — real rings were fragmenting across disconnected subgraphs.**
Even after fix 1, real ring members split across up to 3 separate connected
components for what should have been one 26-member ring. Cause: ring
transactions assign a random device per participant per burst
(`rng.choice(shared_devices)`), which thins direct pairwise device
co-occurrence between any two specific members below the edge-formation
threshold, especially for members who happened to draw different devices
across bursts.

Fix: lowered the device-sharing edge threshold to 1 (any shared device use
between two accounts is inherently suspicious, unlike coincidental IP
overlap -- see bug 3).

**Bug 3 — fixing bug 2 created a worse bug: one 435-account false cluster.**
Lowering the shared-event threshold to 1 for ALL edge types (not just
device) caused a giant false connected component containing 435 of 612
accounts, correctly flagged as "coordinated_ring" but almost entirely real,
unrelated normal accounts. Diagnosed instead of just raising the threshold
back: checked how many normal accounts coincidentally shared an IP prefix.
Answer -- thousands of prefixes had 2-4 unrelated normal accounts collide on
them purely by chance (Faker's public-IP pool is finite relative to ~7,500
transactions), and at threshold=1 those chance collisions chained
transitively into one blob.

Fix: device_id and ip_prefix are not equally strong signals and shouldn't
share one threshold. device_id is an app/browser fingerprint (effectively a
random UUID here) -- any collision between two accounts is real. ip_prefix
is a coarse /16-ish block that unrelated users legitimately collide on by
chance. Split into per-signal-type thresholds: device_id >= 1 shared event,
ip_prefix >= 4 shared events. This removed the false blob entirely while
keeping the real device-sharing signal sensitive.

**Result after all three fixes**, measured (not assumed): card-testing
detection reaches 100% precision / 100% recall on its 105 planted target
accounts. Ring detection reaches 100% precision at every threshold from
0.1-0.5 (fragments are always pure -- never mixed with real accounts), with
recall maxing at 69.6% below threshold 0.3 and dropping above it. Operating
threshold set at 0.3.

**Known remaining limitation, not yet fixed:** 30.4% of real ring members
never form ANY graph edge, even at the lowest threshold tested -- likely
members who only appear in a single burst and never happen to share a
device instance with another member in a way that survives the threshold.
A community-detection algorithm that uses indirect/weak ties (e.g. Louvain)
instead of requiring a direct surviving edge would likely recover more of
these. Left as a documented next step rather than silently claimed as solved.

---

## [Aug 22] ML ensemble — suspiciously perfect numbers, investigated not celebrated

First ensemble run (Isolation Forest + Gradient Boosted Trees on the causal
features): 1.000 precision, 0.969 recall, 0.999 ROC-AUC on the held-out set.
That is not a result to be proud of on a fraud detector -- it's a signal the
benchmark is too easy, since real fraud doesn't separate that cleanly from
real legitimate behavior. Checked the actual score distributions instead of
reporting the headline numbers: normal transactions topped out at risk score
0.38, fraud started at 0.18 in the 1st percentile but 75% of fraud scored
above 0.82. Near-total separation -- because the original fraud patterns
(card testing, ATO) were designed loud: physically-impossible implied travel
speed, 31x amount z-scores, huge velocity spikes. Nobody needs an ML ensemble
to catch a 100,000 km/h transaction; a single threshold rule would do it.

**Fix:** added deliberately stealthy variants of both attacks --
`ato_stealthy` (same city as the account, ordinary spend amount, the ONLY
signal is a new device) and `card_testing_stealthy` (spread over 4 hours
instead of 15 minutes, 6-12 devices and 4-8 IP prefixes instead of 2-5,
higher and more plausible amounts, less extreme decline rate) -- designed
specifically to evade the loud features the detector relies on, mirroring
how real adaptive fraud avoids known thresholds rather than announcing
itself. These make up roughly a third of all fraud rows now, not a token
addition.

**Result:** metrics barely moved (0.970 recall, still 1.000 precision at
threshold 0.5), and false negatives concentrated almost entirely in
`ato_stealthy` (4 of 5 misses) -- which is itself a useful, honest signal:
the stealthy variant genuinely is harder than the loud one, even if not
hard enough to break the model outright.

**Why it's still this easy, and disclosed rather than hidden:** checked
`is_new_device_for_account` base rates directly -- 8.8% in the normal
population vs. 100% in `ato_stealthy`. The normal-account simulation models
users as sticking to the same 1-2 devices for the full 30-day window (only
a 3% per-transaction chance of a new one), so "new device" alone is still a
strong isolated signal here even with geo and amount signals removed. A
more realistic normal population -- people legitimately switching phones,
clearing cookies, using incognito or multiple browsers at a higher base
rate -- would force the model to rely on weaker, combined signals instead
of one dominant flag, and would very likely push recall and precision down
from these numbers. That's flagged here explicitly as a dataset limitation
rather than left implicit: these numbers measure the detector's ability to
learn the designed patterns, not proof of real-world generalization. A
production system would need re-benchmarking on live transaction data,
since real attackers adapt against whatever the detector currently checks
for -- a point the 2026 fraud research (deepfake/synthetic-identity growth,
AI-orchestrated attack tooling) makes directly.

---

## [Aug 22] LLM reasoning layer, triage gate, audit ledger — integrated clean

Built the Pydantic-bounded reasoning agent, the triage gate (auto-allow /
LLM-reasoned / auto-flag-obvious), and the hash-chained audit ledger, then
ran all three together against 322 real scored transactions in one pass.

Worth stating plainly rather than skipping: this phase didn't surface a bug
the way the ring-detection and ML phases did. Checked instead of assumed:
the actual routing counts (150 auto-allow / 97 LLM-band / 75 auto-flag),
the final-status crosstab against true labels (0 fraud reached "allowed";
all 9 false positives capped at "flagged for review," never auto-blocked),
and deliberately ran the tamper-detection test on the ledger (edit an
entry, confirm verify_chain() catches it) rather than assuming it worked.
All checked out. Noted here for contrast with the entries above -- not
every phase needs a manufactured struggle to be worth documenting honestly.

**Real limitation, disclosed directly:** the "llm_reasoned" path in this
run used a rule-based mock, not the real Anthropic API -- this build
environment has no ANTHROPIC_API_KEY. The mock derives its answer from the
same feature values a real prompt would contain, so it's an honest test of
the pipeline's plumbing (routing, schema validation, logging), but it is
NOT evidence of real LLM reasoning quality. That can only be measured
against the live model, which needs a real API key -- next real step once
one is available.
