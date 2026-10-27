import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

VERSION = "clf_v1"

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")

# No stop-word removal: sklearn's English list deletes "not", "no", "never", "cannot",
# which carry meaning here ("card not working"). C=30: weaker regularization gives
# probabilities that are much closer to reality (checked below with ECE).
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    ("clf", LogisticRegression(max_iter=2000, C=30, random_state=42)),
])
pipeline.fit(train_df["text"], train_df["category"])

proba = pipeline.predict_proba(val_df["text"])
conf = proba.max(axis=1)
pred = pipeline.classes_[proba.argmax(axis=1)]
correct = pred == val_df["category"].values

macro_f1 = f1_score(val_df["category"], pred, average="macro")
acc = accuracy_score(val_df["category"], pred)

# Expected Calibration Error: gap between stated confidence and real accuracy (lower is better)
bins = np.linspace(0, 1, 11)
ece = 0.0
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (conf > lo) & (conf <= hi)
    if m.sum():
        ece += m.mean() * abs(conf[m].mean() - correct[m].mean())

print(classification_report(val_df["category"], pred))
print(f"Validation macro-F1: {macro_f1:.4f} | accuracy: {acc:.4f} | ECE: {ece:.3f}")

os.makedirs("models", exist_ok=True)
path = f"models/{VERSION}.joblib"
joblib.dump(pipeline, path)
size_mb = os.path.getsize(path) / (1024 * 1024)
with open(f"models/{VERSION}_meta.json", "w", encoding="utf-8") as f:
    json.dump({"version": VERSION, "val_macro_f1": round(macro_f1, 4), "val_accuracy": round(acc, 4),
               "val_ece": round(ece, 3), "size_mb": round(size_mb, 2),
               "params": {"ngram_range": [1, 2], "max_features": 10000, "C": 30, "stop_words": None}}, f, indent=2)
print(f"Saved {path} ({size_mb:.2f} MB)")