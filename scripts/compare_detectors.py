"""Compare protected-message detectors on TRAIN, out-of-fold (test set and validation untouched).

Fair metric: with the lexical guard fixed as layer 1, for several escalation budgets (% of non-protected
messages sent to human) report the FEWEST protected leaks any threshold can achieve within that budget.
Detectors: A current multiclass P(protected) | B binary word TF-IDF | C binary char n-grams | D word+char.
Run from the project root:  python scripts/compare_detectors.py   (a few minutes)
Caveat: 4 candidates compared on the same data = a little selection optimism; confirm the winner once on
validation, and use the test set only once at the very end.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from app.gate import check_lexical_gate

PROTECTED_CATEGORY = "unauthorized_transactions_security"
os.makedirs("results", exist_ok=True)
train = pd.read_csv("data/train.csv")
is_prot = train["is_protected"].values == 1
lex = np.array([check_lexical_gate(t)["triggered"] for t in train["text"]])
n_prot, n_non = int(is_prot.sum()), int((~is_prot).sum())
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train["text"], train["intent"]))


def word():
    return TfidfVectorizer(ngram_range=(1, 2), max_features=10000)


def char():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000, sublinear_tf=True)


def lr(**kw):
    return LogisticRegression(max_iter=3000, C=30, random_state=42, **kw)


scores = {}
multi = cross_val_predict(Pipeline([("t", word()), ("c", lr())]), train["text"], train["category"], cv=folds, method="predict_proba")
scores["A multiclass (current)"] = multi[:, sorted(train["category"].unique()).index(PROTECTED_CATEGORY)]
y = is_prot.astype(int)
for name, vec in [("B binary word", word()), ("C binary char_wb(2-5)", char()),
                  ("D binary word+char", FeatureUnion([("w", word()), ("c", char())]))]:
    p = cross_val_predict(Pipeline([("t", vec), ("c", lr(class_weight="balanced"))]), train["text"], y, cv=folds, method="predict_proba")
    scores[name] = p[:, 1]
    print(f"done: {name}", flush=True)

budgets = [5, 10, 15, 20, 25]
rows = []
for name, p in scores.items():
    grid = np.unique(np.concatenate([np.quantile(p, np.linspace(0, 1, 3001)), [1.01]]))
    esc = np.array([100 * (lex | (p >= t))[~is_prot].mean() for t in grid])
    leaks = np.array([int((is_prot & ~(lex | (p >= t))).sum()) for t in grid])
    row = {"detector": name}
    for b in budgets:
        ok = esc <= b
        row[f"leaks@<= {b}% esc"] = int(leaks[ok].min()) if ok.any() else None
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv("results/detector_comparison_oof_train.csv", index=False)
print(f"\nTRAIN out-of-fold: protected={n_prot}, non-protected={n_non}; lexical guard fixed as layer 1")
print("Protected leaks (out of %d) at the best threshold within each escalation budget:\n" % n_prot)
print(out.to_string(index=False))
print("\nLower is better. A detector is only better if it leaks fewer at the SAME escalation budget.")