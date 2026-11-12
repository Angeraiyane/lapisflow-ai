# Cascade Triage

A customer-support ticket router built on the public **Banking77** dataset. A fast scikit-learn classifier and a stack of safety gates decide, for each message, whether it can be handled **automatically** or must go to a **human**. Six fraud-related intents are *protected*: they must never be auto-routed.

> **Status (12 Nov 2026).** Local FastAPI service, 33 automated tests passing. The LLM fallback is implemented but **tested only with a fake client** and **disabled by default**; no safety claim depends on it. There is **no hosted deployment** (see [Hosting](#hosting)). All reported numbers come from scripts in this repository. **The test split (1,963 messages) was used once, on 8 Nov 2026 (see Results).**

## How the numbers were obtained (read this first)

| Role | Data | Used for |
|---|---|---|
| Training | train, 9,158 messages | Fitting the classifier |
| Calibration | validation, 1,962 messages | Choosing `tau_protected`, `tau_confidence` and hyperparameters. **Numbers on this split are not independent estimates.** |
| Robustness check | train, 5-fold out-of-fold (727 protected messages) | A larger sample for estimating the miss rate. Diagnostic evidence, **not a certified test result**: the threshold was chosen on validation and the guard was partly designed after observing validation leaks. |
| Final evaluation | test, 1,963 messages | **Used once, on 8 Nov 2026**, through the API, with frozen thresholds (see [Test-split protocol](#test-split-protocol)). It must not be used again to choose anything. |

## Results

| What | Result | Data | Caveat |
|---|---|---|---|
| Classifier alone | Macro-F1 0.9437, accuracy 0.9439, ECE 0.023 | Validation | Calibration set |
| Full cascade: auto coverage / auto accuracy | **58.36%** (1,145/1,962) / **99.48%** | Validation, 12 Nov | Calibration set; `scripts/evaluate_live.py` |
| Total human rate | 41.64% (817 messages): 462 probability gate + 68 lexical guard + 287 low confidence | Validation, 12 Nov | |
| Protected messages routed to human | 156/156: 46 by the lexical guard, 110 by the probability gate | Validation, 12 Nov | **Zero observed misses, not a guarantee.** |
| Protected messages missed by the safety gates | **3/727 = 0.41%**; one-sided 95% upper bound ≈ 1.06% | Train, out-of-fold, `tau_protected = 0.0047` | 23.7% of non-protected messages escalated by the gates |
| Local latency, validation run | p50 7.23 ms, p95 24.92 ms, p99 30.11 ms, max 205.79 ms | 1,962 sequential requests | One run, one machine, no load |
| **TEST split, single run (8 Nov 2026), full cascade through the API** | **58.94%** auto (1,157/1,963), 41.06% human; auto accuracy **99.14%** (1,147/1,157; exact 95% interval 98.42–99.58%) | Test, frozen V0 | Validation accuracy (99.48%) is within noise (p = 0.45); **the rule "auto accuracy ≥ 99.4%" selected on validation was not met on test** |
| **TEST split: protected messages** | **0/156 not routed to a human**; 113 by the probability gate and 43 by the lexical guard. One-sided 95% upper bound on the miss rate: **1.90%** (two-sided interval upper bound: 2.34%) | Test | Zero observed misses is not a guarantee. At a true miss rate of 0.41%, observing zero misses out of 156 happens about 53% of the time. Per-intent intervals are wide and uninformative. |
| Local latency, test run | p50 23.7 ms, p95 47.6 ms, p99 87.5 ms, max 853 ms | 1,963 sequential requests | One run, one machine, no load |

The 12 Nov validation report records 1,962 successful API requests and no request errors. Its provenance records commit `e4d0fbd51232fc6f1431c975ca801cee1cf5212c` and uncommitted changes. The validation results must therefore be read as a measurement of that recorded working state, not as a clean-commit benchmark.

The validation report's 0/156 protected misses is an observation on a calibration set, not an independent safety estimate. The earlier out-of-fold result of 3/727 remains the more useful diagnostic estimate.

My first "0 leaks on validation" claim was weak evidence: with a true miss rate near 0.4%, seeing zero in 156 is unremarkable.

### Per protected intent (train, out-of-fold, `tau_protected = 0.0047`)

The aggregate hides one intent. Exact (Clopper-Pearson) **two-sided** 95% intervals; `scripts/per_intent_oof.py`.

| Protected intent | Messages | Missed | Miss rate | 95% interval |
|---|---:|---:|---:|---:|
| `card_payment_not_recognised` | 146 | **3** | **2.05%** | 0.43–5.89% |
| `cash_withdrawal_not_recognised` | 140 | 0 | 0% | 0–2.60% |
| `compromised_card` | 88 | 0 | 0% | 0–4.11% |
| `direct_debit_payment_not_recognised` | 155 | 0 | 0% | 0–2.35% |
| `lost_or_stolen_card` | 85 | 0 | 0% | 0–4.25% |
| `lost_or_stolen_phone` | 113 | 0 | 0% | 0–3.21% |
| **All protected** | **727** | **3** | **0.41%** | One-sided upper bound ≈ 1.06% |

All three misses belong to the same intent and the same fold (the fifth of five). They were **confident misclassifications into neighbouring categories** (confidence ≥ 0.99):

- "i disagree with a charge" → `fees_charges` (0.999, P(protected) 0.0005, ambiguous label).
- "I'm not recognizing the card payment." → `payments_and_refunds` (0.997, 0.0022, a morphological variant the guard does not match).
- "Why is there an uknown charge showing on my account?" → `fees_charges` (0.991, 0.0031, a typo).

With only three messages, this suggests but does not establish an overlap with fee and payment intents.

A per-fold check (`scripts/fold_check.py`) shows that the fifth fold is **not** weaker than the others: its accuracy is 0.936, against 0.930–0.941 for the other folds. The training data contain **no exact duplicate text**. The clustering in one fold remains unexplained; it would happen by chance about 4% of the time.

The nearest training messages differ for each miss:

- For "i disagree with a charge", none is close (similarity 0.10; the neighbours are fee questions).
- For "I'm not recognizing the card payment.", a correctly labelled paraphrase exists ("I do not recognize the card payment.", similarity 0.27), but with a different word form.
- For the message containing "uknown charge", the neighbours are extra-charge and fee questions.

The five other intents show zero misses, but with 85–155 messages their upper bounds are still 2.35–4.25%. Intervals assume independent messages, although near-duplicates and shared folds weaken that assumption. They were also computed after the fact on six intents. *Why this intent is harder remains a hypothesis (n = 3).*

**What a miss means in code** (`scripts/oof_threshold.py`): a truly protected message where the lexical guard did not match **and** P(protected) is below `tau_protected`. With confidence ≥ 0.95, it would then be routed to `auto`.

The goal "near-zero leakage for +5–10% escalation" is **not achievable with TF-IDF + logistic regression on this data**: observed zero misses out-of-fold requires `tau_protected` ≈ 0.0003, which produces 66.5% escalation. See `SAFETY_CASE.md`.

### Nested cross-validation: detector comparison

A dedicated binary detector was compared with the current multiclass classifier using nested cross-validation on the combined training and validation data. The test split was not used.

- Outer cross-validation: 5 folds.
- Inner cross-validation: 3 folds.
- Protected messages evaluated: 883.
- Detector A: current multiclass classifier.
- Detector D: binary word + character n-gram detector.

The table reports protected messages missed and the actual escalation rate at three escalation budgets.

| Detector | Budget | Protected missed | Miss rate | 95% interval | Real escalation |
|---|---:|---:|---:|---:|---:|
| A multiclass (current) | 10% | 6/883 | 0.68% | 0.25–1.47% | 8.2% |
| A multiclass (current) | 15% | 3/883 | 0.34% | 0.07–0.99% | 12.2% |
| A multiclass (current) | 20% | 3/883 | 0.34% | 0.07–0.99% | 15.9% |
| D binary word + char | 10% | 8/883 | 0.91% | 0.39–1.78% | 8.6% |
| D binary word + char | 15% | 4/883 | 0.45% | 0.12–1.16% | 13.0% |
| D binary word + char | 20% | 2/883 | 0.23% | 0.03–0.82% | 16.7% |

A paired sign test compared the detectors' missed-message outcomes at each budget:

| Budget | Missed only by A | Missed only by D | Paired sign-test p-value |
|---|---:|---:|---:|
| 10% | 4 | 6 | 0.754 |
| 15% | 2 | 3 | 1.000 |
| 20% | 3 | 2 | 1.000 |

Detector D has fewer misses at the 20% budget, but the paired comparisons do **not** establish a statistically significant superiority at any of the three budgets. The results are therefore **promising but inconclusive**. Detector A remains the reference; detector D remains experimental.

The confidence intervals assume independent messages, although near-duplicates may violate this assumption. These results are nested cross-validation estimates, not final test results.

### Manual error analysis

The manual analysis of `results/validation_errors.csv` covers 110 validation misclassifications. Each error was assigned a primary category based on its apparent cause.

| Error type | Count |
|---|---:|
| Label ambiguity | 55 |
| Vague or short message | 38 |
| Multiple intents | 7 |
| Negation | 5 |
| Other | 2 |
| Word form | 2 |
| Typo | 1 |
| **Total** | **110** |

Fourteen of the 110 errors were marked as security-relevant:

| Error type | Security-relevant errors |
|---|---:|
| Vague or short message | 5 |
| Label ambiguity | 4 |
| Negation | 2 |
| Other | 2 |
| Multiple intents | 1 |
| Word form | 0 |
| Typo | 0 |
| **Total** | **14** |

The main observed weaknesses are semantic overlap between categories and messages that are too vague or short to classify reliably. Spelling errors alone account for very few errors. These categories reflect manual interpretation of the validation errors, not an objective ground-truth taxonomy.

## How a ticket is routed

```text
text → [1 lexical guard] → [2 protected-probability gate] → [3 confidence gate] → auto
             │ match                 │ P(protected) >= 0.0047       │ top prob >= 0.95
             ▼                       ▼                              ▼ no
           human                   human                    [4 LLM fallback, off by default]
                                                                      │ any failure / invalid / protected output
                                                                      ▼
                                                                    human
```

- **1 Lexical guard** (`app/gate.py`, `lex_v2_11patterns`): 11 regular expressions; 7 fixed before measuring, **4 added after validation leaks were observed**. Validation numbers for the guard are therefore optimistic.
- **2 Protected gate**: probability of the `unauthorized_transactions_security` category. The classifier predicts 10 *categories*, not the six individual intents, so a threshold specific to one intent is not possible without an intent-level detector.
- **3 Confidence gate**: `tau_confidence = 0.95`; the rule "auto accuracy ≥ 99.4%" was written after viewing the sweep table.
- **4 LLM fallback** (`app/fallback.py`): only for low-confidence messages that no safety layer flagged; one attempt, no retry, 5-second timeout, strict JSON, local rate limit. It adds **no** safety: protected or invalid outputs go to a human, and protected messages are meant never to reach it. A protected message that slips through the gates with low confidence would count as a miss.
- **Audit log** (SQLite, `app/audit.py`): no raw text, only a SHA-256 hash; per request it records the route, decision layer, thresholds, guard version, protected probability and latency. Written after the response. Gate probabilities are deliberately not returned by the public API because they could help someone probe the gates.
- Endpoints: `POST /triage`, `GET /health`, `GET /stats`.

## Run it

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt

python -m pytest -v  # 33 tests

python -m app.main  # API on http://127.0.0.1:8000

python scripts/evaluate_live.py --base-url http://127.0.0.1:8000
# Replays validation; writes a JSON report and a per-request trace (no raw text).

python scripts/evaluate_live.py --csv data/adversarial_cases_template.csv
# Custom cases (never used to tune anything).

python scripts/oof_threshold.py
# Out-of-fold safety analysis on train.

python scripts/per_intent_oof.py
# Per-intent misses with exact intervals.
```

`evaluate_live.py` refuses the test split unless `--final-test` is given.

## Test-split protocol

Written **before** the test split was run (planned for 3 Nov 2026; **actually run on 8 Nov 2026**, on the frozen V0).

1. **Freeze**: clean git state, tag `v0-frozen`, thresholds as in `data/thresholds.json`. Run `python scripts/leakage_check.py` first (duplicates and near-duplicates between splits).
2. **Run once**, through the API: `python scripts/evaluate_live.py --split test --final-test`.
3. **Report as is**: auto coverage, auto accuracy, number of protected messages not routed to a human with the exact 95% interval, route and decision-layer counts. Per-intent test numbers are noisy (about 26 protected messages per intent); the out-of-fold per-intent table is the reference.
4. **No retuning on the result.** If protected messages leak, report and analyse it; any improvement must be evaluated on data the test result did not influence (nested cross-validation).
5. Sensitivity to `tau_protected` is shown on validation and out-of-fold only, never on test.

**What the test can and cannot show** (156 protected messages; one-sided exact 95% upper bound on the miss rate): 0 misses → 1.90%; 1 → 3.00%; 2 → 3.98%; 3 → 4.90%. It cannot demonstrate a miss rate below about 1.9%. If the true rate were 0.41% (the out-of-fold estimate), zero misses would be observed only about 53% of the time, and two or more about 13%; at a true rate of 1%, two or more misses would occur 46% of the time. A clean result is weak evidence; a result with two or more misses is informative.

**Pre-declared reading of the result.** V0 is a measured baseline: **I do not accept it for unattended routing of protected messages whatever the result**, because a protected message that reaches `auto` is a failure by definition. The advisor-suggested stricter criterion for a later version (observed miss rate ≤ 1% and one-sided 95% upper bound < 3%) is met only with zero misses; one miss gives 3.0047%.

**What happened on 8 Nov (deviations included).** The run was a single execution of `python scripts/evaluate_live.py --split test --final-test` (1,963 requests, 0 errors). Results are in the Results table. Deviations: it ran five days later than planned; the report states `uncommitted_changes: true` (commit `01cee7e`), so the frozen state is documented by that commit, not by a clean tag; and **116 of the 1,963 requests (5.91%) are missing from the audit database** (`database is locked`), without effect on the routes or on the report. A validation replay on the same day had four failed requests out of 1,962; their cause was not established.

## Data and splits

Banking77 (13,083 messages, 77 intents; CC BY 4.0, Casanueva et al., 2020). The official train and test files were merged and re-split 70/15/15, stratified on intent (`random_state = 42`): 9,158 / 1,962 / 1,963; protected messages: 727 / 156 / 156.

**This differs from the official split: scores are not comparable with published ones.** The 77 intents map to 10 business categories (`data/categories.json`).

**Leakage check** (`scripts/leakage_check.py`, text and intent only). Exact duplicates after lower-casing and whitespace normalisation: train∩val 1 text, train∩test 2, val∩test 2 (same labels each time; none is a protected intent); inside train 6, val 1, test 0.

Near-duplicates (TF-IDF cosine): 25 of 1,963 test messages (1.3%) have a neighbour at similarity ≥ 0.95 in train+val (1 with a different intent label), 71 at ≥ 0.90 (2 different), 214 at ≥ 0.80 (15 different). For validation → train: 27 / 59 / 192 (2 / 6 / 22 different).

Banking77 contains paraphrase-level near-duplicates across splits, so scores are slightly optimistic and the independence assumed by the confidence intervals only approximately holds. Neighbours with different labels also show label ambiguity. Banking77 has no user ID, so same-user leakage cannot be checked.

**Sanity checks on validation** (`scripts/confusion_report.py`, confusion matrix and per-class metrics in `results/`): an input-independent baseline (always `card_management`) reaches accuracy 0.164 and macro-F1 0.028, against 0.944 and 0.944 for the model. Largest confusion: `top_up` → `transfers` (16 of 264). The protected category has precision 0.961 and recall 0.949 (148 of 156 predicted correctly). A first version of that mapping silently dropped 10 intents and left one protected intent unprotected; its score (0.875) was discarded and the mapping rebuilt with assertions.

## Known limitations

- Per protected intent on validation, the lexical guard catches 18/19 `lost_or_stolen_card` and 14/24 `lost_or_stolen_phone`, but only 1/31 `card_payment_not_recognised` and 1/33 `direct_debit_payment_not_recognised`. Its patterns are primarily about lost and stolen cards, not payments the customer does not recognise.
- The lexical guard is a deterministic floor, not a detector: on train it catches 219/727 protected messages (30%) with 1.4% false alarms. Checked by hand against `lex_v2_11patterns`, it does **not** match "I don't recognise this payment", "There is an unrecognised transaction", "I did not make this payment", "I do not recognize the card payment." or "Someone used my card in another country". It **does** match "This is not an unauthorized charge, I just want a statement" (negation) and "Where is my card? It was ordered last week" (a delivery question).
- TF-IDF + logistic regression cannot separate some protected paraphrases from routine messages. A global threshold treats the six protected intents alike, while the data show that their error rates differ.
- Nested cross-validation suggests that a dedicated binary detector with word + character n-grams may be competitive, but the paired comparisons do not establish a statistically significant improvement. **Detector D remains experimental; detector A remains the reference.**
- The rule that gave `tau_protected = 0.0047` (half of the lowest validation P(protected)) depends on one extreme message; applied on train it gives 0.0003.
- Calibration was checked globally (ECE 0.023), **not** by sub-class or inside the 0.95–1.0 confidence band.
- **The audit log is best-effort, and it was incomplete during the final test.** In the single test replay (1,963 sequential requests), 116 requests (5.91%) were not written to the audit database (`database is locked`; the project folder was inside OneDrive and writes ran in background threads). Routing and the evaluation report are unaffected because the report is built from API responses, but the audit trail of that run is incomplete. After the test, a 30-second lock timeout and WAL mode were added (`app/audit.py`, covered by a concurrent-write test); **the effect has not been re-measured yet**. Measured with `scripts/audit_coverage.py`.
- The audit disk is ephemeral on free hosts; the API has no authentication or real rate limiting.
- The Gemini fallback has never been called live; the model name and free-tier limits are unverified, and prompts sent to a free tier may be used by the provider. Use public data only.
- Real support traffic differs from Banking77 (short, clean, English): it may contain typos, mixed intents and other languages.
- Human-routed tickets do not enter a real support queue: the hand-off is not implemented.

## Hosting

I looked for a free host that runs a FastAPI + scikit-learn service. On 30–31 Oct 2026, Render asked for a payment card for verification (according to its support), and Hugging Face Docker Spaces showed "paid plan required" at Space creation. I did not find a no-card option I could verify. For evaluation, I serve the API from my laptop; the plan is a reproducible repository plus a recorded demo. This is **not** a production deployment.

## Roadmap

1. Review with two classmates and run a live adversarial test using messages written by them. **Outcome not yet recorded.**
2. **Done (8 Nov):** final benchmark on the test split, once, with frozen thresholds.
3. Continue investigating an intent-level protected detector (dedicated binary classifier with character n-grams). Detector D has been evaluated by nested cross-validation, but has not demonstrated statistically significant superiority over detector A.
4. Extend the calibration audit to the 0.95–1.0 confidence band and each protected intent.
5. Test the LLM fallback live only if a claim is ever made about it.
6. Consider a real host if a no-card option can be verified.

## Development notes

Built with AI assistants (code drafting and review). Every reported number was produced by a script in this repository, and I corrected several AI-suggested claims that my own measurements contradicted (see git history).