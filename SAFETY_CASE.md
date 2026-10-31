# Safety case: why protected intents never auto-route

## Policy
Six Banking77 intents are **protected**: `card_payment_not_recognised`, `cash_withdrawal_not_recognised`, `compromised_card`, `direct_debit_payment_not_recognised`, `lost_or_stolen_card`, `lost_or_stolen_phone`. A wrong automatic answer here (for example a stolen-card report handled as a card-delivery question) delays the customer's ability to block a fraudulent use of their account. Policy: **the final route of a protected message must be `human`**, whatever the model's confidence. This is a rule, not a confidence threshold.

## Defences (in order)
1. **Lexical guard**: deterministic patterns (lost, stolen, locate, location, fraud, not recognised, …). It does not depend on the model. 4 of its 11 patterns were added after seeing validation leaks.
2. **Protected-probability gate**: any message whose probability of the protected category is ≥ 0.0047 goes to a human, even when another category is the top prediction.
3. **LLM-output check**: a fallback answer that is invalid or protected goes to a human; protected messages never reach the LLM tier (tested with a fake client).
4. **Audit**: every decision stores its layer, thresholds and guard version, with a hash instead of raw text.

## Evidence
| Question | Evidence | Limit |
|---|---|---|
| Do protected messages reach `auto` on validation? | 0/156, replayed through the API | observed zero; 95 % upper bound ≈ 1.9 %; thresholds were chosen on this set |
| How often do the gates miss a protected message on unseen-by-model data? | 3/727 (0.41 %, 95 % upper bound ≈ 1.06 %), train 5-fold out-of-fold, at 23.7 % escalation of non-protected messages | the fold models see 80 % of train; the 3 misses were all in one fold; patterns were not tuned on train |
| What does zero observed misses cost? | out-of-fold: tau 0.0003 → 0 misses but 66.5 % escalation | unusable in practice |

## What I could not achieve
The goal "leakage near zero for +5–10 % escalation" is not reachable with this model: at 10 % escalation the same analysis gives 7/727 misses (0.96 %). The remaining misses are confident, so a stricter confidence threshold does not catch them.

## Known failure types (train, out-of-fold, tau 0.0047)
"i disagree with a charge" (ambiguous label, P(protected) 0.0005, confidence 0.999); "I'm not recognizing the card payment." (variant of a guard pattern); "Why is there an uknown charge showing on my account?" (typo).

## Not covered
Adversarial inputs, languages other than English, prompt injection against a live LLM (mitigated only by an "untrusted data" instruction in the prompt, tested with a fake client), API authentication, abuse and rate limits, drift. Calibration inside the auto band and per protected intent has not been audited yet.

## Rules I follow so the numbers stay honest
Thresholds are chosen on validation or out-of-fold train data, never on the test split. The test split is used once. If it shows a leak, I report it and analyse it; I do not retune. "Zero" is always written as "observed zero in n cases".