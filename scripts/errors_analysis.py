"""Manual error analysis of the VALIDATION misclassifications (Karpathy: categorise by hand, then report the real split).
Step 1:  python scripts/error_analysis.py            -> writes results/validation_errors.csv (open it in Excel / a spreadsheet)
         Fill the column error_type by hand, one value per row, from the list printed below.
Step 2:  python scripts/error_analysis.py --summarize -> counts your labels (overall and for security-relevant rows)
Test split untouched. Messages are public Banking77 text.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from app.gate import check_lexical_gate

PROTECTED_CATEGORY = "unauthorized_transactions_security"
OUT = "results/validation_errors.csv"
TYPES = ["typo", "word_form", "negation", "vague_or_short", "label_ambiguity", "vocabulary_gap", "multi_intent", "other"]

if "--summarize" in sys.argv:
    df = pd.read_csv(OUT)
    lab = df["error_type"].fillna("").str.strip()
    print(f"{len(df)} errors, {int((lab != '').sum())} labelled\n")
    print("ALL ERRORS:\n" + lab[lab != ""].value_counts().to_string())
    sec = df["security_relevant"] == 1
    print(f"\nSECURITY-RELEVANT ERRORS ({int(sec.sum())}):\n" + lab[sec & (lab != '')].value_counts().to_string())
    print("\nUnlabelled rows:", int((lab == '').sum()))
    sys.exit()

os.makedirs("results", exist_ok=True)
val = pd.read_csv("data/val.csv")
model = joblib.load("models/clf_v1.joblib")
classes = np.array(model.classes_)
proba = model.predict_proba(val["text"])
pred = classes[proba.argmax(axis=1)]
conf = proba.max(axis=1)
p_prot = proba[:, list(classes).index(PROTECTED_CATEGORY)]
with open("data/thresholds.json", encoding="utf-8") as f:
    th = json.load(f)
err = pred != val["category"].values
lex = np.array([check_lexical_gate(t)["triggered"] for t in val["text"]])
df = pd.DataFrame({
    "row": np.where(err)[0], "intent": val["intent"].values[err], "true_category": val["category"].values[err],
    "pred_category": pred[err], "confidence": conf[err].round(3), "p_protected": p_prot[err].round(4),
    "lexical_guard": lex[err].astype(int),
    "would_be_auto": (conf[err] >= th["tau_confidence"]).astype(int) & ~((p_prot[err] >= th["tau_protected"]) | lex[err]),
    "security_relevant": ((val["is_protected"].values[err] == 1) | (pred[err] == PROTECTED_CATEGORY)).astype(int),
    "text": val["text"].values[err], "error_type": "", "notes": ""})
df["would_be_auto"] = df["would_be_auto"].astype(int)
df.sort_values(["would_be_auto", "security_relevant", "confidence"], ascending=False).to_csv(OUT, index=False)

print(f"Validation errors: {len(df)} of {len(val)} ({100 * len(df) / len(val):.2f}%)")
print(f"  that the cascade would route to AUTO (confident, no safety gate): {int(df['would_be_auto'].sum())}")
print(f"  security-relevant (true protected or predicted protected): {int(df['security_relevant'].sum())}")
print("\nMost frequent confusions (true -> predicted):")
print(df.groupby(["true_category", "pred_category"]).size().sort_values(ascending=False).head(10).to_string())
print(f"\nSaved {OUT}. Fill error_type with one of: {', '.join(TYPES)}")
print("Then run:  python scripts/error_analysis.py --summarize")