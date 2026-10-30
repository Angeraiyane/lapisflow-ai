"""Out-of-fold (OOF) analysis of the safety gates on TRAIN only. Test set and patched validation untouched.

Every train message is scored by a model that never saw it (5 folds, stratified on intent).
Reports, for several tau_protected values:
  - gate leaks      : protected messages NOT sent to human by the safety gates (lexical + probability)
  - auto leaks      : protected messages whose FINAL route would be 'auto' (leak AND confidence >= tau_confidence)
  - 95% upper bound (Clopper-Pearson), per-fold leaks, escalation rate on non-protected
Also saves the leaked messages for manual reading. Run from the project root:  python scripts/oof_threshold.py
Caveat: each fold model sees 80% of train, so it is a little less confident than the final model.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from app.gate import check_lexical_gate

PROTECTED_CATEGORY = "unauthorized_transactions_security"
os.makedirs("results", exist_ok=True)
with open("data/thresholds.json", encoding="utf-8") as f:
    th = json.load(f)
tau_c = th["tau_confidence"]

train = pd.read_csv("data/train.csv")
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    ("clf", LogisticRegression(max_iter=2000, C=30, random_state=42)),
])
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train["text"], train["intent"]))
fold_id = np.zeros(len(train), dtype=int)
for i, (_, te) in enumerate(folds):
    fold_id[te] = i
proba = cross_val_predict(pipe, train["text"], train["category"], cv=folds, method="predict_proba")
classes = sorted(train["category"].unique())
p_prot = proba[:, classes.index(PROTECTED_CATEGORY)]
conf = proba.max(axis=1)
is_prot = train["is_protected"].values == 1
lex = np.array([check_lexical_gate(t)["triggered"] for t in train["text"]])
n_prot, n_non = int(is_prot.sum()), int((~is_prot).sum())


def upper95(k, n):  # one-sided 95% Clopper-Pearson upper bound
    return 1.0 if k >= n else float(beta.ppf(0.95, k + 1, n - k))


print(f"TRAIN (out-of-fold): n={len(train)}, protected={n_prot}, non-protected={n_non}, tau_confidence={tau_c}")
print(f"Lexical guard: protected caught {int((lex & is_prot).sum())}/{n_prot} | "
      f"non-protected sent to human {int((lex & ~is_prot).sum())}/{n_non} = {100 * (lex & ~is_prot).sum() / n_non:.1f}%")

unc = is_prot & ~lex
print(f"\nProtected NOT caught by the guard: {int(unc.sum())}  -> distribution of their P(protected):")
for q in [0, 0.5, 1, 2, 5, 10, 25, 50]:
    print(f"  {q:>4}th percentile: {np.percentile(p_prot[unc], q):.4f}")
print("  (the minimum alone is one outlier; read the percentiles)")

rows = []
for tp in [0.0003, 0.0005, 0.001, 0.002, 0.0047, 0.01, 0.02, 0.05]:
    sent = lex | (p_prot >= tp)
    leak = is_prot & ~sent
    k, k_auto = int(leak.sum()), int((leak & (conf >= tau_c)).sum())
    rows.append({
        "tau_protected": tp,
        "gate_leaks": k, "gate_leak_pct": round(100 * k / n_prot, 2), "gate_upper95_pct": round(100 * upper95(k, n_prot), 2),
        "auto_leaks": k_auto, "auto_leak_pct": round(100 * k_auto / n_prot, 2),
        "leaks_per_fold": "/".join(str(int((leak & (fold_id == i)).sum())) for i in range(5)),
        "nonprotected_escalated_pct": round(100 * sent[~is_prot].mean(), 1),
    })
out = pd.DataFrame(rows)
out.to_csv("results/oof_train_threshold.csv", index=False)
print("\n=== GUARD + PROBABILITY GATE (train, out-of-fold) ===")
print(out.to_string(index=False))

for tp in (0.0047, 0.02):
    leak = is_prot & ~(lex | (p_prot >= tp))
    df = train.loc[leak, ["intent", "text"]].assign(p_protected=p_prot[leak].round(4), confidence=conf[leak].round(3))
    df.to_csv(f"results/oof_leaks_tau{tp}.csv", index=False)
    print(f"\nLeaked messages at tau={tp}: {len(df)} (saved to results/oof_leaks_tau{tp}.csv). By intent:")
    print(df["intent"].value_counts().to_string() if len(df) else "  none")
print("\n'Auto leaks' = protected messages that would really reach 'auto'; 'gate leaks' = not sent to human by the safety gates.")