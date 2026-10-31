"""Per-protected-intent miss rates with exact (Clopper-Pearson) 95% intervals.
Uses the leaked messages already saved by oof_threshold.py: no model is re-run.
Run from the project root:  python scripts/per_intent_oof.py [results/oof_leaks_tau0.0047.csv]
"""
import sys

import pandas as pd
from scipy.stats import beta

leak_path = sys.argv[1] if len(sys.argv) > 1 else "results/oof_leaks_tau0.0047.csv"
train = pd.read_csv("data/train.csv")
n = train[train["is_protected"] == 1].groupby("intent").size().rename("n_train")
k = pd.read_csv(leak_path).groupby("intent").size().rename("misses")
df = pd.concat([n, k], axis=1).fillna(0).astype(int)
df["miss_pct"] = (100 * df["misses"] / df["n_train"]).round(2)
df["ci95_low_pct"] = [round(100 * (beta.ppf(0.025, m, t - m + 1) if m > 0 else 0.0), 2) for m, t in zip(df["misses"], df["n_train"])]
df["ci95_high_pct"] = [round(100 * (beta.ppf(0.975, m + 1, t - m) if m < t else 1.0), 2) for m, t in zip(df["misses"], df["n_train"])]
print(f"Source of misses: {leak_path}\n")
print(df.sort_values("miss_pct", ascending=False).to_string())
tot_n, tot_k = int(df["n_train"].sum()), int(df["misses"].sum())
print(f"\nAll protected: {tot_k}/{tot_n} = {100 * tot_k / tot_n:.2f}%")
print("Intervals are exact two-sided 95%; small counts per intent make them wide. They assume independent messages (near-duplicates and shared folds break this).")