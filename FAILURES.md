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

---

## [Aug 23] Isolation Forest silently dead for single-transaction scoring

While preparing the live API (which scores one new transaction at a time,
not a batch), tested `ensemble_score()` against a single row before wiring
it up, instead of assuming the batch-tested function would just work.

It didn't. `ensemble_score` rescaled the Isolation Forest's raw anomaly
score using min-max normalization computed FROM THE BATCH PASSED IN. That's
fine for offline evaluation (thousands of rows, a real min and max). For a
single live transaction, min == max == the one value, so the rescaled score
is `(x - x) / (x - x + 1e-9)` = exactly 0.0, always, regardless of how
anomalous the transaction actually was. Confirmed directly: fed it a
genuinely low-risk row and a genuinely high-risk row, both came back with
iso_score = 0.0. Half the ensemble would have silently gone dead the moment
this hit a live API, and every decision would have quietly relied on the
GBT component alone -- with no error, no warning, nothing to notice.

**Fix:** compute calibration bounds (min/max of the Isolation Forest's raw
scores) ONCE, from the training set, save them alongside the trained models,
and reuse those fixed bounds for every future scoring call regardless of
batch size. Re-verified with the same two test rows -- iso_score now moves
correctly (0.08 for the low-risk row, 0.98 for the high-risk one).

Also refactored the causal feature-computation logic (features.py) into a
shared FeatureState class so the live API and the offline benchmark use the
literal same implementation, not two versions that could quietly drift
apart. Verified the refactor changed nothing: reran the full feature
pipeline before and after, diffed every numeric column -- max difference
across all ~8,300 rows was 1.4e-14 (floating point noise, not a real
change).

---

## [Aug 23] Live testing surfaced a real leakage bug the offline benchmark missed

Built the FastAPI backend, started it for real, and hit it with actual HTTP
requests rather than trusting the unit-level tests alone. Simulated a
textbook account-takeover through `POST /api/simulate/ato`: an established
161-day-old account, a purchase 15 standard deviations above its normal
spend, a brand-new device, a city 1,285 km away. Expected: flagged.

Got: risk_score 0.384, auto-allowed. A clean ATO signature slipped through
the exact system built to catch it -- and this happened on the LIVE path,
not in the offline benchmark, because the offline benchmark never
constructs this specific combination in isolation the way a single live
request does.

**Root cause, found by checking feature importances instead of guessing:**
`account_age_days` carried 85.5% of the GBT model's decision weight.
Checked why: in the training data, `card_testing`/`card_testing_stealthy`
accounts were ALWAYS exactly 0 days old (they're synthetic target accounts
that never got a real signup record), and `ring` accounts averaged 10 days.
Every legitimate account was 150-240 days old. The model had learned "brand
new account = fraud" as a clean shortcut -- true by construction in this
dataset, not a real fraud behavior, and it starved genuinely meaningful
signals (amount anomaly, new device, geo jump) of influence because the age
shortcut already explained most of the training loss.

**Fix, at the data level, not by hiding the symptom:** added a cohort of
genuinely new, legitimate signup accounts (`gen_new_signup_accounts` /
`gen_new_legitimate_account_transactions`) -- real first-time users making
their first few ordinary purchases, with account age in the same 0-25 day
range as the fraud accounts. This forces the model to stop using age as a
clean tell. Retrained: `account_age_days` importance dropped to 4.4%.

**That fix had a side effect, chased down rather than left unexplained:**
ring-detection precision on its own strict metric dropped from 100% to
66%. Diagnosed: unrelated to the new signup cohort -- the stealthy
card-testing variant (deliberately built to evade card-testing heuristics)
was evading the ring/card-testing sub-type CLASSIFIER too, and getting
labeled "coordinated_ring" instead of "card_testing_burst." Checked
whether the actual defensive action (flag the cluster) still held up
despite the wrong sub-label: yes -- any-fraud-type precision stayed at
100%, recall 79.9%. Fixed the evaluation to report both the honest primary
number (is this cluster fraud at all) and the secondary, disclosed-as-
best-effort sub-type breakdown, instead of one misleading strict number.

**One more real gap, found by re-testing the same ATO case after the data
fix:** score moved from 0.384 to 0.437 (into the "escalate to LLM
reasoning" band -- a real improvement) but the GBT component alone was
still muted (0.234) for a textbook takeover. Checked feature importances
again: `ip_velocity_1h` now dominated at 86.4%, and ATO has only 45
training examples (isolated single events) against 310 for card-testing
(bursts). Gradient boosting was optimizing for whichever fraud sub-type
was most populous and separable, under-serving the rarer one. Fixed with
sample-weighting (4x) on ato/ato_stealthy rows during training -- not a
full re-architecture, a targeted correction for a specific, diagnosed
imbalance. Result: same ATO case now scores 0.457, feature importance
spread from one dominant signal to a genuine mix (ip_velocity 64%, amount
z-score 12%, new-device 8%, new-geo 5%), and false negatives on real ATO
dropped from 4 to 1 out of 45 total ATO examples in the test set.

**Where this honestly leaves things:** the live-caught bug is meaningfully
better, not perfectly solved -- 0.457 sits in the "ambiguous, escalate to
LLM reasoning" band rather than the "obviously flag" band, which is a
defensible outcome for a single ambiguous signal in isolation (a real
high-spending traveler could look similar), not a confident catch. A
production system would want more ATO training examples and probably a
lower auto-allow threshold specifically for high-amount-anomaly cases.
Documented here rather than tuned further to hit a rounder number.
