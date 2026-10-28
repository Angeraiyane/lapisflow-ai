"""Two plots required by the D3 pass/fail check, computed on the VALIDATION set only:
1. results/calibration.png            reliability curve + ECE
2. results/leakage_vs_escalation.png  protected leakage vs % of non-protected sent to human
"""
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROTECTED_CATEGORY = "unauthorized_transactions_security"
os.makedirs("results", exist_ok=True)

val = pd.read_csv("data/val.csv")
model = joblib.load("models/clf_v1.joblib")
classes = list(model.classes_)
proba = model.predict_proba(val["text"])
top = np.array(classes)[proba.argmax(axis=1)]
conf = proba.max(axis=1)
p_prot = proba[:, classes.index(PROTECTED_CATEGORY)]
is_prot = val["is_protected"].values == 1
with open("data/thresholds.json", encoding="utf-8") as f:
    tau_p = json.load(f)["tau_protected"]

# ---- 1. Calibration ----
bins = np.linspace(0, 1, 11)
ece, xs, ys, ns = 0.0, [], [], []
correct = top == val["category"].values
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (conf > lo) & (conf <= hi)
    if m.sum():
        ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
        xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
plt.figure(figsize=(5.5, 5.5))
plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
plt.plot(xs, ys, "o-", color="steelblue", label=f"clf_v1 (ECE={ece:.3f})")
for x, y, n in zip(xs, ys, ns):
    plt.annotate(str(n), (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7)
plt.xlabel("stated confidence"); plt.ylabel("actual accuracy")
plt.title("Calibration on validation (numbers = samples per bin)")
plt.legend(); plt.tight_layout(); plt.savefig("results/calibration.png", dpi=150); plt.close()
print(f"Saved results/calibration.png (ECE={ece:.3f})")

# ---- 2. Leakage vs escalation ----
taus = np.unique(np.concatenate([np.logspace(-4, -0.3, 80), [tau_p]]))
rows = []
for t in taus:
    sent = p_prot >= t
    rows.append({"tau_protected": float(t),
                 "protected_leaked": int((is_prot & ~sent).sum()),
                 "nonprotected_sent_to_human_pct": 100 * sent[~is_prot].mean()})
df = pd.DataFrame(rows)
df.to_csv("results/leakage_vs_escalation.csv", index=False)
cur = df.iloc[(df["tau_protected"] - tau_p).abs().argmin()]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.step(df["nonprotected_sent_to_human_pct"], df["protected_leaked"], where="post", color="firebrick")
ax.axvspan(5, 10, color="gray", alpha=0.15, label="escalation range discussed with Khan (5-10%)")
ax.scatter([cur["nonprotected_sent_to_human_pct"]], [cur["protected_leaked"]], color="black", zorder=5,
           label=f"chosen tau_protected={tau_p} -> {int(cur['protected_leaked'])} leaks, "
                 f"{cur['nonprotected_sent_to_human_pct']:.1f}% escalated")
ax.set_xlabel("% of non-protected messages sent to human by the safety gate")
ax.set_ylabel(f"protected messages leaked (out of {int(is_prot.sum())})")
ax.set_title("Safety gate: leakage vs escalation (validation)")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("results/leakage_vs_escalation.png", dpi=150); plt.close()
print(f"Saved results/leakage_vs_escalation.png | tau_protected={tau_p}: "
      f"{int(cur['protected_leaked'])} leaks, {cur['nonprotected_sent_to_human_pct']:.1f}% non-protected escalated")

# ---- 3. Data-driven threshold: largest tau with zero leakage = lowest P(protected) among truly protected ----
tau_star = float(p_prot[is_prot].min())
tau_rec = round(0.5 * tau_star, 4)          # safety margin: half of the largest zero-leak value
for name, t in [("largest zero-leak tau (no margin)", tau_star), ("recommended tau (half, with margin)", tau_rec)]:
    sent = p_prot >= t
    print(f"{name}: tau={t:.4f} | leaked={int((is_prot & ~sent).sum())}/{int(is_prot.sum())} | "
          f"non-protected sent to human={100 * sent[~is_prot].mean():.1f}%")
print("Note: 0 leaks observed on", int(is_prot.sum()), "protected messages is NOT zero guaranteed (95% upper bound ~ 3/n).")