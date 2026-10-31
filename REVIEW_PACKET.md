# Review packet — Cascade Triage (read before the session)
*Goal of the review: find reasons my numbers are misleading or my thresholds unsafe. Please attack, not validate. Numbers are from scripts in this repo; the test split has not been used.*

## 1. What it is (3 lines)
A router for customer-support messages from the public Banking77 dataset. Each message goes to **auto** (handled by the model) or **human**. Six fraud-related intents (stolen/lost card or phone, unrecognised card payment, cash withdrawal or direct debit) are **protected**: their final route must always be human.

## 2. How a message is routed
text → lexical guard (11 patterns) → protected-probability gate (P(protected) ≥ 0.0047 → human) → confidence gate (top probability ≥ 0.95 → auto) → otherwise human (an LLM fallback exists but is off, tested with a fake client only; protected messages are meant never to reach it).

## 3. Definitions
- **Leak / miss**: a truly protected message whose final route is not human.
- **Auto coverage**: share of all messages routed to auto. **Auto accuracy**: correct category among auto messages.
- **Escalation by the safety gates**: share of *non-protected* messages sent to a human by the lexical guard or the protected gate (not counting low-confidence escalation).
- **Total human rate**: all messages with final route human.

## 4. Numbers (state which data each comes from)
| Number | Value | Data |
|---|---|---|
| Classifier alone, macro-F1 | 0.9437 | validation, 1,962 msgs |
| Full cascade replayed through the API: auto coverage / auto accuracy | 58.36 % / 99.48 % | validation (thresholds were chosen here: not independent) |
| Total human rate | 41.64 % (817 msgs) | validation replay |
| Escalation by the safety gates | 20.7 % of non-protected (374/1,806) | validation replay |
| Protected messages routed to human | 156/156 (observed; 95 % upper bound on the miss rate ≈ 1.9 %) | validation replay |
| Protected messages missed by the safety gates | **3/727 (0.41 %, 95 % upper bound ≈ 1.06 %)** with **23.7 %** of non-protected escalated by the gates | train, 5-fold out-of-fold |

**How tau_protected = 0.0047 was chosen:** a rule on validation (half of the lowest P(protected) among the 156 validation protected messages, ≈ 0.0093), not a grid search. The out-of-fold analysis later showed the rule's weakness: at this value 3 of 727 protected messages are still missed (all confident, all `card_payment_not_recognised`: an ambiguous label, a morphological variant, a typo). Observed zero misses out-of-fold needs tau ≈ 0.0003, i.e. 66.5 % escalation.

## 5. Statements to score (1 = strongly disagree, 5 = strongly agree)
For each: **score**, **one-sentence reason**, **one concrete attack or missing test**. "Looks fine" without evidence is not accepted.

*Safety claims*
1. The 3 misses out of 727 are real model failures, not labelling or preprocessing errors.
2. Out-of-fold predictions on train are a fair proxy for new tickets.
3. A 95 % upper bound (~1.06 %) is an honest way to report leakage risk.
4. The lexical guard catches a protected message even if the classifier gives it a low probability. (Give an example that breaks it.)
5. Protected messages cannot reach the LLM tier, and a protected message that slips through the gates with low confidence would count as a miss. Is that definition of a leak right?

*Evaluation method*
6. The validation set is representative of real bank support tickets. What shift worries you?
7. Macro-F1 is the right headline metric. What would you headline?
8. tau_protected = 0.0047 is defensible. What should I have done instead?
9. Coverage 58.36 % at 99.48 % accuracy is reported without selection bias. Where could bias creep in?
10. Reporting "not zero, observed 0.41 %" is enough to justify shipping this threshold. What standard of evidence would you require?

*Live stress test (10 min, added after Hack's suggestion)*: each reviewer writes 3–5 adversarial messages (a protected intent disguised as a routine request, a routine request that sounds protected, an ambiguous one) in `data/adversarial_cases_template.csv` format (`text,is_protected,category`). I replay them with `python scripts/evaluate_live.py --csv <file>` and we discuss every message that is not routed to a human although it should be. These messages were never used to choose any threshold.

## 6. Collection template (copy per reviewer)
| # | Score 1–5 | Reason (1 sentence) | Concrete attack / missing test |
|---|---|---|---|
| 1 | | | |
| … | | | |

## 7. After the review (my side)
Every score ≤ 3 becomes an action item: the concern in my own words, the experiment that would address it, and whether it fits the 30-day plan.

## 8. README readability test (separate, 3 questions, time it)
Give the README (not this packet) to a reader who does not know the project and time them: (1) What does it do? (2) What are the key numbers? (3) How do I run it? Target: under 3 minutes (a rule of thumb, not a proof).