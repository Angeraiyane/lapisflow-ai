"""Replay messages against a RUNNING /triage endpoint and save a report (standard library only).

Default split is VALIDATION (already used to choose thresholds: say so when you quote the numbers).
The TEST split is refused unless you pass --final-test: keep it for ONE final benchmark.

Examples (server running; local or through a tunnel):
  python scripts/evaluate_live.py --base-url http://127.0.0.1:8000
  python scripts/evaluate_live.py --base-url https://xxxx.trycloudflare.com --limit 300
Output: results/eval_<split>_<date>.json
"""
import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8000")
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--limit", type=int, default=0, help="0 = every message of the split")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--final-test", action="store_true", help="required to use the test split")
ap.add_argument("--csv", help="custom cases (e.g. adversarial): columns text,is_protected[,category]; overrides --split")
args = ap.parse_args()
if args.split == "test" and not args.final_test:
    sys.exit("Refusing to use the TEST split. Pass --final-test only for the one final benchmark.")

src = args.csv or f"data/{args.split}.csv"
with open(src, encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
if args.limit:
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]


def post(text):
    req = urllib.request.Request(args.base_url.rstrip("/") + "/triage", data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t0) * 1000


try:
    from scipy.stats import beta
except ImportError:  # intervals are skipped if scipy is missing
    beta = None


def cp(k, n):
    """Exact two-sided 95% (Clopper-Pearson) interval in %, or None."""
    if beta is None or n == 0:
        return None
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return [round(100 * lo, 2), round(100 * hi, 2)]


def git_info():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": sha, "uncommitted_changes": dirty}
    except Exception:
        return None


def layer_of(reason):
    """Decision layer, read from the API's 'reason' text."""
    if reason.startswith("Lexical"):
        return "lexical_gate"
    if reason.startswith("Safety policy"):
        return "probability_gate"
    if reason.startswith("Confidence threshold"):
        return "confidence_gate"
    if reason.startswith("Low classifier confidence"):
        return "llm_fallback"
    return "low_confidence"


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))], 2) if xs else None


results, errors, lat = [], 0, []
for i, row in enumerate(rows, 1):
    try:
        out, ms = post(row["text"])
    except (urllib.error.URLError, TimeoutError, ValueError):
        errors += 1
        continue
    lat.append(ms)
    results.append({"client_ms": round(ms, 2), "request_id": out.get("request_id", ""), "reason": out.get("reason", ""), "conf": out.get("confidence"),
                    "route": out["route"], "pred": out["category"], "true": row.get("category") or "", "intent": row.get("intent") or "", "text_id": i,
                    "protected": str(row["is_protected"]).strip() == "1", "layer": layer_of(out["reason"])})
    if i % 200 == 0:
        print(f"{i}/{len(rows)}", flush=True)

n = len(results)
if not n:
    sys.exit("No successful request: is the server running and the URL right?")
by_route = {r: sum(x["route"] == r for x in results) for r in ("auto", "llm", "human")}
auto = [x for x in results if x["route"] == "auto"]
auto_lab = [x for x in auto if x["true"]]  # only messages with a known category
prot = [x for x in results if x["protected"]]
leaks = [x for x in prot if x["route"] != "human"]  # strict: final route must be human
def count_layers(items):
    out = {}
    for x in items:
        out[x["layer"]] = out.get(x["layer"], 0) + 1
    return out


report = {
    "date_utc": datetime.now(timezone.utc).isoformat(), "base_url_host": args.base_url.split("//")[-1].split("/")[0],
    "split": ("custom:" + os.path.basename(args.csv)) if args.csv else args.split, "n_requested": len(rows), "n_ok": n, "n_errors": errors,
    "note": ("custom cases, never used to choose thresholds" if args.csv else
             "validation split was used to choose thresholds: not an independent test" if args.split == "val" else "final test split"),
    "route_counts": by_route, "route_pct": {k: round(100 * v / n, 2) for k, v in by_route.items()},
    "auto_accuracy": (round(sum(x["pred"] == x["true"] for x in auto_lab) / len(auto_lab), 4) if auto_lab else None),
    "layers_all_requests": count_layers(results),
    "layers_protected_messages": count_layers(prot),
    "protected_total": len(prot), "protected_not_routed_to_human": len(leaks),
    "protected_leak_pct": round(100 * len(leaks) / len(prot), 2) if prot else None,
    "protected_leak_ci95_pct": cp(len(leaks), len(prot)),
    "per_intent_protected": {
        intent: {"n": sum(1 for x in prot if x["intent"] == intent),
                 "not_routed_to_human": sum(1 for x in leaks if x["intent"] == intent),
                 "ci95_pct": cp(sum(1 for x in leaks if x["intent"] == intent), sum(1 for x in prot if x["intent"] == intent))}
        for intent in sorted({x["intent"] for x in prot if x["intent"]})},
    "git": git_info(),
    "thresholds_file": json.load(open("data/thresholds.json", encoding="utf-8")) if os.path.exists("data/thresholds.json") else None,
    "client_latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99), "max": round(max(lat), 2),
                          "mean": round(statistics.mean(lat), 2)},
}
os.makedirs("results", exist_ok=True)
tag0 = ("custom_" + os.path.splitext(os.path.basename(args.csv))[0]) if args.csv else args.split
trace_path = f"results/trace_{tag0}_{datetime.now(timezone.utc):%Y%m%d}.csv"
with open(trace_path, "w", encoding="utf-8", newline="") as tf:  # no raw text: row index + request_id (join with the audit DB)
    w = csv.writer(tf)
    w.writerow(["row", "request_id", "intent", "true_category", "is_protected", "pred_category", "confidence", "route", "decision_layer", "client_ms", "reason"])
    for x in results:
        w.writerow([x["text_id"], x["request_id"], x["intent"], x["true"], int(x["protected"]), x["pred"], x["conf"], x["route"], x["layer"], x["client_ms"], x["reason"]])
tag = ("custom_" + os.path.splitext(os.path.basename(args.csv))[0]) if args.csv else args.split
path = f"results/eval_{tag}_{datetime.now(timezone.utc):%Y%m%d}.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
print(f"\nSaved {path} and per-request trace {trace_path}. Observed zero is not a guarantee; quote counts with n.")