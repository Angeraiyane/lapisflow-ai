"""Leakage check between train / validation / test (Karpathy's checklist). Reads TEXT and INTENT only:
no model is run and no test performance is computed. Run from the project root: python scripts/leakage_check.py
Reports: exact duplicates (after lower-casing and whitespace normalisation) and near-duplicates (TF-IDF cosine).
Banking77 has no user id, so "same user" leakage cannot be checked (a limitation).
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

os.makedirs("results", exist_ok=True)
sp = {n: pd.read_csv(f"data/{n}.csv") for n in ("train", "val", "test")}


def norm(t):
    return re.sub(r"\s+", " ", str(t).lower().strip())


for d in sp.values():
    d["norm"] = d["text"].map(norm)

rows = []
print("=== EXACT DUPLICATES (normalised text) ===")
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    shared = set(sp[a]["norm"]) & set(sp[b]["norm"])
    diff_label = 0
    for t in shared:
        la = set(sp[a].loc[sp[a]["norm"] == t, "intent"])
        lb = set(sp[b].loc[sp[b]["norm"] == t, "intent"])
        diff_label += int(la != lb)
    print(f"{a} vs {b}: {len(shared)} shared texts ({diff_label} with different intent labels)")
    for t in list(shared)[:3]:
        print("   e.g.", t)
    rows.append({"check": f"exact {a}~{b}", "count": len(shared), "different_labels": diff_label})
for n, d in sp.items():
    print(f"duplicates inside {n}: {int(d['norm'].duplicated().sum())}")

print("\n=== NEAR-DUPLICATES (TF-IDF word 1-2 grams, cosine similarity) ===")
vec = TfidfVectorizer(ngram_range=(1, 2)).fit(pd.concat([d["text"] for d in sp.values()]))
X = {n: vec.transform(d["text"]) for n, d in sp.items()}
for a, ref in [("val", ["train"]), ("test", ["train", "val"])]:
    Xr = X[ref[0]] if len(ref) == 1 else __import__("scipy.sparse", fromlist=["vstack"]).vstack([X[r] for r in ref])
    lab = np.concatenate([sp[r]["intent"].values for r in ref])
    txt = np.concatenate([sp[r]["text"].values for r in ref])
    sims = (X[a] @ Xr.T).toarray()
    best, idx = sims.max(axis=1), sims.argmax(axis=1)
    for thr in (0.95, 0.90, 0.80):
        m = best >= thr
        dl = int((lab[idx[m]] != sp[a]["intent"].values[m]).sum())
        print(f"{a} -> {'+'.join(ref)}: {int(m.sum())} messages with a neighbour at similarity >= {thr} ({dl} with a different intent)")
        rows.append({"check": f"near {a}->{'+'.join(ref)} >= {thr}", "count": int(m.sum()), "different_labels": dl})
    top = np.argsort(-best)[:3]
    for i in top:
        print(f"   top pair (sim {best[i]:.2f}): [{sp[a]['intent'].values[i]}] {sp[a]['text'].values[i]}  <->  [{lab[idx[i]]}] {txt[idx[i]]}")
pd.DataFrame(rows).to_csv("results/leakage_check.csv", index=False)
print("\nSaved results/leakage_check.csv. Preprocessing (TF-IDF vocabulary, classifier) is fitted on train only in train_baseline.py; thresholds were chosen on validation.")