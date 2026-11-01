# Cascade Triage

A customer-support ticket router built on the public **Banking77** dataset. A fast scikit-learn classifier and a stack of safety gates decide, for each message, whether it can be handled **automatically** or must go to a **human**. Six fraud-related intents are *protected*: they must never be auto-routed.

> **Status (31 Oct 2026).** Local FastAPI service, 32 automated tests. The LLM fallback is implemented but **tested only with a fake client** and **disabled by default**; no safety claim depends on it. There is **no hosted deployment** (see [Hosting](#hosting)). All numbers below come from scripts in this repo that I ran. **The test split (1,963 messages) has not been used yet.**

## How the numbers were obtained (read this first)

| Role | Data | Used for |
|---|---|---|
| Training | train, 9,158 msgs | fitting the classifier |
| Calibration | validation, 1,962 msgs | choosing `tau_protected`, `tau_confidence`, hyper-parameters. **Numbers on it are not independent estimates.** |
| Robustness check | train, 5-fold out-of-fold (727 protected msgs) | a larger sample for the miss rate. Diagnostic evidence, **not a certified test result** (the threshold was chosen on validation, the guard was patterned partly after validation leaks) |
| Final evaluation | test, 1,963 msgs | **untouched**; will be run **once**, through the API, with frozen thresholds (see [Test-split protocol](#test-split-protocol)) |

## Results

| What | Result | Data | Caveat |
|---|---|---|---|
| Classifier alone | macro-F1 0.9437, accuracy 0.9439, ECE 0.023 | validation | calibration set |
| Full cascade replayed through the API: auto coverage / auto accuracy | **58.36 %** (1,145/1,962) / **99.48 %** | validation | calibration set; `scripts/evaluate_live.py` |
| Total human rate | 41.64 % (817 msgs) = 462 probability gate + 68 lexical guard + 287 low confidence | validation | |
| Protected messages routed to human | 156/156 (46 by the lexical guard, 110 by the probability gate) | validation | *observed* zero, not a guarantee; at this threshold the lexical guard caught nothing the probability gate missed |
| Protected messages missed by the safety gates | **3 of 727 = 0.41 %**; one-sided 95 % upper bound ≈ 1.06 % | train, out-of-fold, `tau_protected = 0.0047` | with 23.7 % of non-protected messages escalated by the gates |
| Local latency (round trip, no load) | p50 29.9 ms, p95 104.7 ms, **p99 402 ms** | 1,962 sequential requests | the 200 ms target is missed at p99; one run, one machine |

My first "0 leaks on validation" claim was weak evidence: with a true miss rate near 0.4 %, seeing 0 in 156 is unremarkable.

### Per protected intent (train, out-of-fold, `tau_protected = 0.0047`)

The aggregate hides one intent. Exact (Clopper-Pearson) **two-sided** 95 % intervals; `scripts/per_intent_oof.py`.

| Protected intent | Messages | Missed | Miss rate | 95 % interval |
|---|---|---|---|---|
| `card_payment_not_recognised` | 146 | **3** | **2.05 %** | 0.43 – 5.89 % |
| `cash_withdrawal_not_recognised` | 140 | 0 | 0 % | 0 – 2.60 % |
| `compromised_card` | 88 | 0 | 0 % | 0 – 4.11 % |
| `direct_debit_payment_not_recognised` | 155 | 0 | 0 % | 0 – 2.35 % |
| `lost_or_stolen_card` | 85 | 0 | 0 % | 0 – 4.25 % |
| `lost_or_stolen_phone` | 113 | 0 | 0 % | 0 – 3.21 % |
| **All protected** | 727 | 3 | 0.41 % | one-sided upper bound ≈ 1.06 % |

All three misses are in one intent, all in the same fold (the fifth of five), and all were **confident misclassifications into neighbouring categories** (confidence ≥ 0.99): "i disagree with a charge" → `fees_charges` (0.999, P(protected) 0.0005, ambiguous label); "I'm not recognizing the card payment." → `payments_and_refunds` (0.997, 0.0022, a morphological variant the guard does not match); "Why is there an uknown charge showing on my account?" → `fees_charges` (0.991, 0.0031, a typo). With three messages this suggests, but does not establish, an overlap with fee and payment intents. A per-fold check (`scripts/fold_check.py`) shows that the fifth fold is **not** a weaker one (accuracy 0.936, against 0.930–0.941 for the others) and that train holds **no exact duplicate text**, so the clustering in one fold is unexplained (it would happen by chance 4 % of the time). The nearest training messages differ for each miss: for "i disagree with a charge" none is close (similarity 0.10; the neighbours are fee questions); for "I'm not recognizing the card payment." a correctly labelled paraphrase exists ("I do not recognize the card payment.", similarity 0.27) but with a different word form; for "…uknown charge…" the neighbours are extra-charge and fee questions. The five other intents show zero misses, but with 85–155 messages their upper bounds are still 2.35–4.25 %. Intervals assume independent messages (near-duplicates and shared folds break this) and were computed after the fact on six intents. *Why this intent is harder is still a hypothesis (n = 3).*

**What a miss means in code** (`scripts/oof_threshold.py`): a truly protected message where the lexical guard did not match **and** P(protected) < `tau_protected`. With confidence ≥ 0.95 it would then be routed to `auto`.

The goal "near-zero leakage for +5–10 % escalation" is **not achievable with TF-IDF + logistic regression on this data**: observed zero misses out-of-fold needs `tau_protected` ≈ 0.0003, i.e. 66.5 % escalation. See `SAFETY_CASE.md`.

## How a ticket is routed

```
text → [1 lexical guard] → [2 protected-probability gate] → [3 confidence gate] → auto
            │ match              │ P(protected) >= 0.0047        │ top prob >= 0.95
            ▼                    ▼                               ▼ no
          human                human                    [4 LLM fallback, off by default]
                                                           │ any failure / invalid / protected output
                                                           ▼
                                                         human
```
- **1 Lexical guard** (`app/gate.py`, `lex_v2_11patterns`): 11 regular expressions; 7 fixed before measuring, **4 added after I saw validation leaks** (validation numbers for the guard are optimistic).
- **2 Protected gate**: probability of the `unauthorized_transactions_security` category. The classifier predicts the 10 *categories*, not the 6 individual intents, so a threshold specific to one intent is not possible without an intent-level detector.
- **3 Confidence gate**: `tau_confidence = 0.95`; rule "auto accuracy ≥ 99.4 %" written after viewing the sweep table.
- **4 LLM fallback** (`app/fallback.py`): only for low-confidence messages that no safety layer flagged; one attempt, no retry, 5 s timeout, strict JSON, local rate limit. It adds **no** safety: protected or invalid outputs go to a human, and protected messages are meant never to reach it (a protected message that slips through the gates with low confidence would count as a miss).
- **Audit log** (SQLite, `app/audit.py`): no raw text, only a SHA-256 hash; per request: route, decision layer, thresholds, guard version, protected probability, latency. Written after the response. Gate probabilities are deliberately not returned by the public API (they would help probing the gates).
- Endpoints: `POST /triage`, `GET /health`, `GET /stats`.

## Run it

```bash
pip install -r requirements.txt            # dev tools: requirements-dev.txt
python -m pytest -v                        # 32 tests
python -m app.main                         # API on http://127.0.0.1:8000
python scripts/evaluate_live.py --base-url http://127.0.0.1:8000        # replays validation; writes a JSON report and a per-request trace (no raw text)
python scripts/evaluate_live.py --csv data/adversarial_cases_template.csv   # custom cases (never used to tune anything)
python scripts/oof_threshold.py            # out-of-fold safety analysis on train
python scripts/per_intent_oof.py           # per-intent misses with exact intervals
```
`evaluate_live.py` refuses the test split unless `--final-test` is given.
<!-- VERIFY before publishing: paths of prepare_data.py / train_baseline.py / compare_hyperparams.py / audit_gate.py in your repo (git ls-files) and add the exact commands here. -->

## Test-split protocol

Written **before** the test split is run. 1) Freeze: git tag `v0-frozen`, thresholds as in `data/thresholds.json`. 2) Run once through the API: `python scripts/evaluate_live.py --split test --final-test`. 3) Report as is: auto coverage, auto accuracy, number of protected messages not routed to a human with the exact 95 % interval, per-intent counts, route and decision-layer counts. 4) **No retuning on the result.** If protected messages leak, I report and analyse it; any improvement is then evaluated on data the test result did not influence. 5) Sensitivity to `tau_protected` is shown on validation and out-of-fold only, never on test.
<!-- DECIDE: a numeric acceptance target (e.g. "upper bound of the protected miss rate <= X %") and what I do if it is exceeded. Write it BEFORE running the test. -->

## Data and splits

Banking77 (13,083 messages, 77 intents; CC BY 4.0, Casanueva et al., 2020). Official train and test files merged and re-split 70/15/15, stratified on intent (`random_state = 42`): 9,158 / 1,962 / 1,963; protected messages 727 / 156 / 156. **This differs from the official split: scores are not comparable with published ones.** 77 intents map to 10 business categories (`data/categories.json`). A first version of that mapping silently dropped 10 intents and left one protected intent unprotected; its score (0.875) was discarded and the mapping rebuilt with assertions.

## Known limitations

- TF-IDF + logistic regression cannot separate some protected paraphrases from routine messages; a global threshold treats the six protected intents as alike, and the data show they are not (see per-intent table).
- A comparison on train (out-of-fold) suggests a dedicated binary detector with word + character n-grams may leak less at equal escalation (1–7 leaks, thresholds optimised on the same data): **promising, not proven**.
- The rule that gave `tau_protected = 0.0047` (half of the lowest validation P(protected)) depends on one extreme message; applied on train it gives 0.0003.
- Calibration was checked globally (ECE 0.023), **not** by sub-class or inside the 0.95–1.0 band.
- The audit disk is ephemeral on free hosts; the API has no authentication or real rate limiting.
- The Gemini fallback has never been called live; model name and free-tier limits are unverified, and prompts sent to a free tier may be used by the provider: public data only.
- Real support traffic differs from Banking77 (short, clean, English): typos, mixed intents, other languages.
- Humans receive "human"-routed tickets in no real queue: the hand-off is not implemented.

## Hosting

I looked for a free host that runs a FastAPI + scikit-learn service. On 30–31 Oct 2026, Render asked for a payment card for verification (per its support) and Hugging Face Docker Spaces showed "paid plan required" at Space creation. I did not find a no-card option I could verify. For evaluation I serve the API from my laptop; the plan is a reproducible repo plus a recorded demo. This is **not** a production deployment.

## Roadmap

1. Review with two classmates plus a live adversarial test (messages written by them). 2. Final benchmark on the test split, once, frozen. 3. Intent-level protected detector (dedicated binary classifier with char n-grams), evaluated by nested cross-validation on data the test result did not influence. 4. Calibration audit in the 0.95–1.0 band and per protected intent; manual error analysis of validation errors. 5. Live test of the LLM fallback only if a claim is ever made about it. 6. A real host if a no-card option can be verified.

## Development notes

Built with AI assistants (code drafting and review). Every reported number was produced by a script in this repository, and I corrected several AI-suggested claims that my own measurements contradicted (see git history).