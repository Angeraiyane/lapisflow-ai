"""Measure the lexical guard on the VALIDATION set only (test set untouched).
Does NOT change main.py. Run from the project root:  python scripts/measure_regex.py

Answers two questions:
  1. How many of the 156 protected messages does the guard catch (and the 2 known leaks)?
  2. How many NON-protected messages does it send to human, and from which intents?
Then: guard + probability gate together, for several tau_protected values.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from app.gate import check_lexical_gate

PROTECTED_CATEGORY = "unauthorized_transactions_security"
os.makedirs("results", exist_ok=True)

val = pd.read_csv("data/val.csv")
model = joblib.load("models/clf_v1.joblib")
classes = list(model.classes_)
proba = model.predict_proba(val["text"])
p_prot = proba[:, classes.index(PROTECTED_CATEGORY)]
is_prot = val["is_protected"].values == 1
n_prot, n_non = int(is_prot.sum()), int((~is_prot).sum())

res = val["text"].apply(check_lexical_gate)
lex = res.apply(lambda r: r["triggered"]).values
val = val.assign(pattern=res.apply(lambda r: r["pattern"]))

print(f"Validation: n={len(val)}, protected={n_prot}, non-protected={n_non}\n")

print("=== 1. LEXICAL GUARD ALONE ===")
print(f"Protected caught: {int((lex & is_prot).sum())}/{n_prot}")
print(f"Non-protected sent to human: {int((lex & ~is_prot).sum())}/{n_non} "
      f"= {100 * (lex & ~is_prot).sum() / n_non:.1f}%")

print("\n=== 2. THE 2 KNOWN LEAKS AT tau_protected=0.05 ===")
known = ["Are you able to guide me to my card's location?", "Are you able to locate my card?"]
for t in known:
    m = val["text"] == t
    print(f"  caught by guard={bool(lex[m.values].any())} | {t}")

print("\n=== 3. NON-PROTECTED MESSAGES HIT BY THE GUARD, BY INTENT (top 10) ===")
fp = val.loc[lex & ~is_prot, ["intent", "text", "pattern"]]
print(fp["intent"].value_counts().head(10).to_string())
fp.to_csv("results/regex_false_positives.csv", index=False)

print("\n=== 4. GUARD + PROBABILITY GATE (human if either fires) ===")
rows = []
for tp in [0.0047, 0.005, 0.01, 0.02, 0.05]:
    sent = lex | (p_prot >= tp)
    rows.append({"tau_protected": tp,
                 "protected_leaked": int((is_prot & ~sent).sum()),
                 "protected_total": n_prot,
                 "nonprotected_sent_to_human_pct": round(100 * sent[~is_prot].mean(), 1)})
out = pd.DataFrame(rows)
out.to_csv("results/regex_plus_gate.csv", index=False)
print(out.to_string(index=False))
print("\nReminder: 0 leaks on 156 is 'observed zero', not a guarantee (95% upper bound ~ 3/156 = 2%).")
print("\n=== 5. THRESHOLD RULE APPLIED TO PROTECTED NOT CAUGHT BY THE GUARD ===")
unc = is_prot & ~lex
tau_star = float(p_prot[unc].min())
tau_rec = round(0.5 * tau_star, 4)
sent = lex | (p_prot >= tau_rec)
print(f"Protected not caught by guard: {int(unc.sum())} | lowest P(protected): {tau_star:.4f} | recommended tau (half): {tau_rec}")
print(f"leaked={int((is_prot & ~sent).sum())}/{n_prot} | non-protected sent to human={100 * sent[~is_prot].mean():.1f}%")