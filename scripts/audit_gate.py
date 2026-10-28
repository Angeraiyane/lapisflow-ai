"""Audit of the two gates on the VALIDATION set (the test set is not used).

1. Protected gate: leakage vs escalation for several tau_protected values.
2. Which protected messages slip through at the current tau_protected.
3. Confidence sweep: accuracy, number of classes inside macro-F1, protected messages in 'auto'
   and protected messages that would reach the LLM tier.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

MODEL_PATH = "models/clf_v1.joblib"
PROTECTED_CATEGORY = "unauthorized_transactions_security"
os.makedirs("results", exist_ok=True)

val = pd.read_csv("data/val.csv")
model = joblib.load(MODEL_PATH)
classes = list(model.classes_)
proba = model.predict_proba(val["text"])
top = np.array(classes)[proba.argmax(axis=1)]
conf = proba.max(axis=1)
p_prot = proba[:, classes.index(PROTECTED_CATEGORY)]
is_prot = val["is_protected"].values == 1
n_prot = int(is_prot.sum())

with open("data/thresholds.json", encoding="utf-8") as f:
    cur = json.load(f)
tau_p_cur, tau_c_cur = cur["tau_protected"], cur["tau_confidence"]
print(f"Validation: n={len(val)}, protected={n_prot}")
print(f"Current thresholds: tau_protected={tau_p_cur}, tau_confidence={tau_c_cur}\n")

# ---- 1. Protected gate: leakage vs escalation ----
print("=== 1. PROTECTED GATE (a leak = a truly protected message NOT sent to human by this gate) ===")
rows = []
for tp in sorted(set([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, tau_p_cur])):
    sent = p_prot >= tp
    rows.append({"tau_protected": tp,
                 "protected_leaked": int((is_prot & ~sent).sum()),
                 "protected_total": n_prot,
                 "nonprotected_sent_to_human_pct": round(100 * sent[~is_prot].mean(), 1)})
gate = pd.DataFrame(rows)
gate.to_csv("results/gate_audit_protected.csv", index=False)
print(gate.to_string(index=False))

# ---- 2. Which protected messages slip through at the current tau_protected ----
leak = is_prot & ~(p_prot >= tau_p_cur)
print(f"\n=== 2. PROTECTED MESSAGES THAT SLIP THROUGH AT tau_protected={tau_p_cur}: {int(leak.sum())} ===")
if leak.any():
    out = val.loc[leak, ["intent", "text"]].copy()
    out["predicted"] = top[leak]
    out["confidence"] = conf[leak].round(3)
    out["p_protected"] = p_prot[leak].round(4)
    print(out.to_string(index=False, max_colwidth=70))
    out.to_csv("results/gate_audit_leaks.csv", index=False)

# ---- 3. Confidence sweep with the current safety gate applied ----
print("\n=== 3. CONFIDENCE SWEEP (current safety gate applied first) ===")
safe = ~(p_prot >= tau_p_cur)
rows = []
for tc in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]:
    auto = safe & (conf >= tc)
    n = int(auto.sum())
    y, yhat = val.loc[auto, "category"], top[auto]
    rows.append({"tau_confidence": tc,
                 "auto_pct": round(100 * auto.mean(), 2),
                 "accuracy_auto": round(accuracy_score(y, yhat), 4) if n else None,
                 "macro_f1_auto": round(f1_score(y, yhat, average="macro", zero_division=0), 4) if n else None,
                 "classes_in_macro_f1": len(set(y) | set(yhat)),
                 "protected_in_auto": int((is_prot & auto).sum()),
                 "protected_reaching_llm_tier": int((is_prot & safe & (conf < tc)).sum())})
conf_df = pd.DataFrame(rows)
conf_df.to_csv("results/gate_audit_confidence.csv", index=False)
print(conf_df.to_string(index=False))
print("\nHow to read it: if 'classes_in_macro_f1' drops from 10 to 9 between two rows, the macro-F1 jump")
print("is partly an artifact (a class disappeared from the metric). Compare 'accuracy_auto' instead.")
print("'protected_reaching_llm_tier' = protected messages that the future LLM fallback would see.")