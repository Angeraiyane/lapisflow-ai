"""Do the out-of-fold misses cluster for a reason? Per-fold quality + nearest train neighbours of each miss.
Same 5 folds, model and thresholds as oof_threshold.py. Train only; test untouched. Run: python scripts/fold_check.py
Output is for reading (and a CSV of the neighbours): it does NOT change any threshold or pattern.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from app.gate import check_lexical_gate

PROTECTED_CATEGORY = "unauthorized_transactions_security"
with open("data/thresholds.json", encoding="utf-8") as f:
    tau = json.load(f)["tau_protected"]
train = pd.read_csv("data/train.csv")
pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
                 ("clf", LogisticRegression(max_iter=2000, C=30, random_state=42))])
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train["text"], train["intent"]))
fold_id = np.zeros(len(train), dtype=int)
for i, (_, te) in enumerate(folds):
    fold_id[te] = i
proba = cross_val_predict(pipe, train["text"], train["category"], cv=folds, method="predict_proba")
classes = np.array(sorted(train["category"].unique()))
pred = classes[proba.argmax(axis=1)]
p_prot = proba[:, list(classes).index(PROTECTED_CATEGORY)]
is_prot = train["is_protected"].values == 1
lex = np.array([check_lexical_gate(t)["triggered"] for t in train["text"]])
miss = is_prot & ~(lex | (p_prot >= tau))

print(f"tau_protected = {tau}; misses = {int(miss.sum())}\n")
rows = []
for i in range(5):
    m = fold_id == i
    rows.append({"fold": i, "n": int(m.sum()), "protected": int((m & is_prot).sum()),
                 "accuracy": round(float((pred[m] == train['category'].values[m]).mean()), 4),
                 "macro_f1": round(float(f1_score(train['category'][m], pred[m], average='macro')), 4),
                 "protected_misses": int((m & miss).sum())})
print(pd.DataFrame(rows).to_string(index=False))
print(f"\nExact duplicate texts inside train: {int(train['text'].duplicated().sum())}")

vec = TfidfVectorizer(ngram_range=(1, 2)).fit(train["text"])
X = vec.transform(train["text"])
out = []
for idx in np.where(miss)[0]:
    sims = (X @ X[idx].T).toarray().ravel()
    sims[idx] = -1
    print(f"\nMISS  [{train['intent'].iloc[idx]}] fold {fold_id[idx]}  predicted {pred[idx]}  P(protected) {p_prot[idx]:.4f}\n      {train['text'].iloc[idx]}")
    for j in np.argsort(-sims)[:4]:
        print(f"   sim {sims[j]:.2f}  fold {fold_id[j]}  [{train['intent'].iloc[j]}]  {train['text'].iloc[j]}")
        out.append({"miss_text": train["text"].iloc[idx], "neighbour_text": train["text"].iloc[j], "sim": round(float(sims[j]), 3),
                    "neighbour_intent": train["intent"].iloc[j], "neighbour_fold": int(fold_id[j])})
os.makedirs("results", exist_ok=True)
pd.DataFrame(out).to_csv("results/fold_check_neighbours.csv", index=False)
print("\nSaved results/fold_check_neighbours.csv")