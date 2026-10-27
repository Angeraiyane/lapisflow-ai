"""Reproduces the two choices behind clf_v1 on the VALIDATION set (test set untouched)."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")


def evaluate(stop_words, C):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words=stop_words)),
        ("clf", LogisticRegression(max_iter=2000, C=C, random_state=42)),
    ]).fit(train_df["text"], train_df["category"])
    proba = pipe.predict_proba(val_df["text"])
    conf = proba.max(axis=1)
    pred = pipe.classes_[proba.argmax(axis=1)]
    ok = pred == val_df["category"].values
    bins = np.linspace(0, 1, 11)
    ece = sum(((conf > lo) & (conf <= hi)).mean() * abs(conf[(conf > lo) & (conf <= hi)].mean() - ok[(conf > lo) & (conf <= hi)].mean())
              for lo, hi in zip(bins[:-1], bins[1:]) if ((conf > lo) & (conf <= hi)).sum())
    return f1_score(val_df["category"], pred, average="macro"), ece


print(f"{'stop_words':<12}{'C':<6}{'macro-F1':<10}{'ECE':<8}")
for sw, C in [("english", 1), (None, 1), (None, 10), (None, 30), (None, 100)]:
    f1, ece = evaluate(sw, C)
    print(f"{str(sw):<12}{C:<6}{f1:<10.4f}{ece:<8.3f}")