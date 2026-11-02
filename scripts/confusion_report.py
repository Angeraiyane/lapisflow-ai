"""Full confusion matrix, per-class metrics, majority-class baseline, and the safety-gate outcome per protected intent
for the VALIDATION set (test untouched). Run from the project root: python scripts/confusion_report.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from app.gate import check_lexical_gate

PROTECTED_CATEGORY = "unauthorized_transactions_security"
os.makedirs("results", exist_ok=True)
val = pd.read_csv("data/val.csv")
train = pd.read_csv("data/train.csv")
model = joblib.load("models/clf_v1.joblib")
classes = list(model.classes_)
proba = model.predict_proba(val["text"])
pred = np.array(classes)[proba.argmax(axis=1)]
true = val["category"].values
with open("data/thresholds.json", encoding="utf-8") as f:
    tau = json.load(f)["tau_protected"]

cm = pd.DataFrame(confusion_matrix(true, pred, labels=classes), index=[f"true:{c}" for c in classes], columns=classes)
cm.to_csv("results/confusion_matrix_val.csv")
print("=== CONFUSION MATRIX (validation; rows = true, columns = predicted) ===")
print(cm.to_string())
rep = pd.DataFrame(classification_report(true, pred, output_dict=True, zero_division=0)).T.round(4)
rep.to_csv("results/per_class_metrics_val.csv")
print("\n=== PER-CLASS METRICS ===")
print(rep.to_string())

maj = train["category"].value_counts().idxmax()
base = np.array([maj] * len(val))
print(f"\n=== INPUT-INDEPENDENT BASELINE (always '{maj}') ===")
print(f"accuracy {accuracy_score(true, base):.4f}, macro-F1 {f1_score(true, base, average='macro', zero_division=0):.4f}  vs model accuracy {accuracy_score(true, pred):.4f}, macro-F1 {f1_score(true, pred, average='macro'):.4f}")

p_prot = proba[:, classes.index(PROTECTED_CATEGORY)]
lex = np.array([check_lexical_gate(t)["triggered"] for t in val["text"]])
is_prot = val["is_protected"].values == 1
sent = lex | (p_prot >= tau)
print(f"\n=== PROTECTED INTENTS: sent to human by the safety gates (tau_protected={tau}), validation ===")
rows = []
for intent in sorted(val.loc[is_prot, "intent"].unique()):
    m = (val["intent"] == intent).values
    rows.append({"intent": intent, "n_val": int(m.sum()), "sent_to_human_by_gates": int(sent[m].sum()),
                 "caught_by_lexical_guard": int(lex[m].sum()), "model_predicts_protected_category": int((pred[m] == PROTECTED_CATEGORY).sum())})
out = pd.DataFrame(rows)
out.to_csv("results/protected_per_intent_val.csv", index=False)
print(out.to_string(index=False))
print("\nWith ~25 messages per intent in a single split, per-intent numbers are noisy: rely on the out-of-fold per-intent table.")