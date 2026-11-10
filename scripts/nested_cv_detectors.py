"""Nested cross-validation: current multiclass protected detector (A) vs binary word+char n-gram detector (D).
Uses TRAIN+VALIDATION only; the TEST split is never read. Run from the project root:
    python scripts/nested_cv_detectors.py            (about 20-40 minutes; prints progress)
    python scripts/nested_cv_detectors.py --quick    (3 outer folds, 2 inner folds: a few minutes)
For each OUTER fold and each escalation budget B (10 %, 15 %, 20 % of non-protected messages sent to a human):
  the threshold is chosen on INNER out-of-fold scores of the outer-training part only, then applied to the untouched outer fold.
Reports protected misses out of all protected messages, exact 95 % interval, real escalation, and a paired comparison.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from app.gate import check_lexical_gate

QUICK = "--quick" in sys.argv
OUTER, INNER = (3, 2) if QUICK else (5, 3)
BUDGETS = [10, 15, 20]
PROTECTED_CATEGORY = "unauthorized_transactions_security"
os.makedirs("results", exist_ok=True)

df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/val.csv")], ignore_index=True)
y_cat, y_bin, intent = df["category"].values, (df["is_protected"].values == 1).astype(int), df["intent"].values
text = df["text"].values
lex = np.array([check_lexical_gate(t)["triggered"] for t in text])
is_prot = y_bin == 1
N = len(df)
print(f"train+validation: n={N}, protected={int(is_prot.sum())} | outer={OUTER} inner={INNER} | TEST NOT USED", flush=True)


def word():
    return TfidfVectorizer(ngram_range=(1, 2), max_features=10000)


def char():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000, sublinear_tf=True)


def lr(**kw):
    return LogisticRegression(max_iter=3000, C=30, random_state=42, **kw)


def fit_score_A(tr, te):
    m = Pipeline([("t", word()), ("c", lr())]).fit(text[tr], y_cat[tr])
    return m.predict_proba(text[te])[:, list(m.classes_).index(PROTECTED_CATEGORY)]


def fit_score_D(tr, te):
    m = Pipeline([("t", FeatureUnion([("w", word()), ("c", char())])), ("c", lr(class_weight="balanced"))]).fit(text[tr], y_bin[tr])
    return m.predict_proba(text[te])[:, 1]


DET = {"A multiclass (current)": fit_score_A, "D binary word+char": fit_score_D}


def inner_oof(fit_score, idx):
    p = np.zeros(len(idx))
    for a, b in StratifiedKFold(INNER, shuffle=True, random_state=1).split(idx, intent[idx]):
        p[b] = fit_score(idx[a], idx[b])
    return p


def pick_tau(p, idx, budget):
    """Smallest threshold whose escalation of non-protected (lexical OR p >= tau) stays within the budget."""
    non = ~is_prot[idx]
    cands = np.unique(np.concatenate([p, [1.01]]))
    for t in cands:  # ascending: smallest tau first
        esc = 100 * (lex[idx] | (p >= t))[non].mean()
        if esc <= budget:
            return t
    return 1.01


leak = {(d, b): np.zeros(N, bool) for d in DET for b in BUDGETS}
esc_sent = {(d, b): np.zeros(N, bool) for d in DET for b in BUDGETS}
for k, (tr, te) in enumerate(StratifiedKFold(OUTER, shuffle=True, random_state=42).split(text, intent), 1):
    for d, fs in DET.items():
        p_in = inner_oof(fs, tr)
        p_out = fs(tr, te)
        for b in BUDGETS:
            tau = pick_tau(p_in, tr, b)
            sent = lex[te] | (p_out >= tau)
            leak[(d, b)][te] = is_prot[te] & ~sent
            esc_sent[(d, b)][te] = sent
        print(f"outer fold {k}/{OUTER} done: {d}", flush=True)


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return f"{100 * lo:.2f}-{100 * hi:.2f}%"


rows = []
nprot = int(is_prot.sum())
for d in DET:
    for b in BUDGETS:
        k_ = int(leak[(d, b)].sum())
        rows.append({"detector": d, "budget_%": b, "protected_missed": f"{k_}/{nprot}", "miss_rate_%": round(100 * k_ / nprot, 2),
                     "ci95": cp(k_, nprot), "real_escalation_%": round(100 * esc_sent[(d, b)][~is_prot].mean(), 1)})
out = pd.DataFrame(rows)
out.to_csv("results/nested_cv_detectors.csv", index=False)
print("\n=== NESTED CV (train+validation) ===")
print(out.to_string(index=False))
print("\n=== PAIRED COMPARISON D vs A (messages missed by only one of them) ===")
for b in BUDGETS:
    a_only = int((leak[("A multiclass (current)", b)] & ~leak[("D binary word+char", b)]).sum())
    d_only = int((leak[("D binary word+char", b)] & ~leak[("A multiclass (current)", b)]).sum())
    p = binomtest(min(a_only, d_only), a_only + d_only, 0.5).pvalue if a_only + d_only else 1.0
    print(f"budget {b}%: missed only by A: {a_only} | only by D: {d_only} | exact sign test p = {p:.3f}")
print("\nDevelopment estimate (test split untouched). Intervals assume independent messages (near-duplicates exist: see leakage_check).")