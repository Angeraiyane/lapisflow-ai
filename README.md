# Cascade Triage

A customer-support ticket router built on the public **Banking77** dataset. A fast scikit-learn classifier and a stack of safety gates decide, for each message, whether it can be handled **automatically** or must go to a **human**. Six fraud-related intents are *protected*: they must never be auto-routed.

> **Status (31 Oct 2026).** Local FastAPI service, 32 automated tests. The LLM fallback is implemented but **tested only with a fake client** and **disabled by default**. There is **no hosted deployment** (see [Hosting](#hosting)). All numbers below come from scripts in this repo that I ran; the test split has **not** been used yet.

## Results

| What | Result | Data | Caveat |
|---|---|---|---|
| Classifier alone | macro-F1 0.9437, accuracy 0.9439, ECE 0.023 | validation, 1,962 msgs | thresholds and hyper-parameters were chosen on validation, so this is not an independent test |
| Full cascade, replayed through the API | **58.36 %** auto (1,145/1,962), 41.64 % human; accuracy among auto **99.48 %** | validation | same caveat; `scripts/evaluate_live.py` |
| Protected messages, replayed through the API | 156/156 routed to human (46 by the lexical guard, 110 by the probability gate) | validation | *observed* zero, not a guarantee (95 % upper bound ≈ 1.9 % for 0/156); the lexical guard caught nothing the probability gate would have missed at this threshold |
| Protected messages missed by the safety gates | **3 of 727 (0.41 %, 95 % upper bound ≈ 1.06 %)** at `tau_protected = 0.0047`, with **23.7 %** of non-protected messages escalated by the gates | train, 5-fold out-of-fold | larger sample than validation; `scripts/oof_threshold.py` |
| Local latency (API round trip, no load) | p50 29.9 ms, p95 104.7 ms, **p99 402 ms** | 1,962 sequential requests | the 200 ms target is missed at p99; one run, one machine |

My first "0 leaks on validation" claim was weak evidence: with a true miss rate near 0.4 %, seeing 0 in 156 is unremarkable. The out-of-fold analysis above is the number to trust. **The ambition "near-zero leakage for +5–10 % escalation" is not achievable with TF-IDF + logistic regression on this data**; see `SAFETY_CASE.md`.

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
- **1 Lexical guard** (`app/gate.py`, version `lex_v2_11patterns`): 11 regular expressions. 7 were fixed before measuring; **4 were added after I had seen validation leaks**, so validation numbers for this guard are optimistic.
- **2 Protected gate**: probability of the `unauthorized_transactions_security` category (the 6 protected intents form exactly this category).
- **3 Confidence gate**: `tau_confidence = 0.95`. The rule (auto accuracy ≥ 99.4 %) was written after viewing the sweep table.
- **4 LLM fallback** (`app/fallback.py`): only for low-confidence messages that no safety layer flagged. One attempt, no retry, 5 s timeout, strict JSON, local rate limit; any failure, invalid category or protected category → human. Protected messages never reach it (tested with a fake client).
- **Audit log** (SQLite, `app/audit.py`): no raw text, only a SHA-256 hash; stores route, decision layer, thresholds and guard version per request. Written after the response is sent. The disk of free hosts is ephemeral.
- Endpoints: `POST /triage`, `GET /health`, `GET /stats`.

## Run it

```bash
pip install -r requirements.txt            # dev tools: requirements-dev.txt
python -m pytest -v                        # 32 tests
python -m app.main                         # API on http://127.0.0.1:8000
python scripts/evaluate_live.py --base-url http://127.0.0.1:8000   # replays validation; refuses the test split
python scripts/oof_threshold.py            # out-of-fold safety analysis on train
```
<!-- VERIFY before publishing: paths of prepare_data.py / train_baseline.py / compare_hyperparams.py / audit_gate.py in your repo (git ls-files) and add the exact commands here. -->

## Data and splits

Banking77 (13,083 messages, 77 intents; CC BY 4.0, Casanueva et al., 2020). The official train and test files were merged and re-split 70/15/15, stratified on intent (`random_state = 42`): 9,158 / 1,962 / 1,963. **This differs from the official split, so scores are not comparable with published ones.** 77 intents map to 10 business categories (`data/categories.json`). A first version of that mapping silently dropped 10 intents and left one protected intent unprotected; its score (0.875) was discarded and the mapping rebuilt with assertions.

## Known limitations

- TF-IDF + logistic regression cannot separate some protected paraphrases from routine messages. The three misses at the final threshold were all confident (≥ 0.99): an ambiguous label ("i disagree with a charge"), a morphological variant ("not recognizing") and a typo ("uknown").
- A comparison on train (out-of-fold) suggests a dedicated binary detector with word + character n-grams may leak less at equal escalation (counts of 1–7 leaks, thresholds optimised on the same data): **promising, not proven**.
- Calibration was checked globally (ECE 0.023), **not** by sub-class or inside the 0.95–1.0 band.
- The audit disk is ephemeral on free hosts; there is no authentication or real rate limiting on the API.
- The Gemini fallback has never been called live; its model name and free-tier limits are unverified, and prompts sent to the free tier may be used by the provider: use public data only.
- Validation numbers are not independent of my choices. The test split (1,963 messages) is kept for a single final benchmark (`--final-test`).

## Hosting

I looked for a free host that runs a FastAPI + scikit-learn service. On 30–31 Oct 2026: Render asked for a payment card for verification (per its support); Hugging Face Docker Spaces showed "paid plan required" at Space creation. I did not find a no-card option I could verify. For evaluation I serve the API from my laptop through a temporary tunnel, which is **not** a production deployment.

## Roadmap

1. Final benchmark on the untouched test split (once). 2. Nested cross-validation for a binary char-n-gram protected detector. 3. Calibration audit in the 0.95–1.0 band and per protected intent; manual error analysis. 4. Live test of the LLM fallback with a verified model name and quota. 5. A real host, if a no-card option can be verified.

## Development notes

Built with AI assistants (code drafting and review). Every reported number was produced by a script in this repository, and I corrected several AI-suggested claims that my own measurements contradicted (see git history).