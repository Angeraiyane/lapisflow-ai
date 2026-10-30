"""Send N requests to the running local API and print p50/p95/p99 (standard library only).
Run the server in another terminal first:  python -m app.main
Then:  python scripts/latency_check.py
Note: sequential calls, NO concurrent load. Say so when you quote the numbers.
"""
import json
import statistics
import time
import urllib.request

URL = "http://127.0.0.1:8000/triage"
TEXTS = [
    "I want to know where my new physical card is right now.",
    "Someone stole my card and took money from an ATM.",
    "How do I top up my account by bank transfer?",
    "Why was I charged a fee for my card payment?",
    "My transfer has not arrived yet.",
]
N = 200
client_ms, server_ms = [], []
for i in range(N):
    body = json.dumps({"text": TEXTS[i % len(TEXTS)]}).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    client_ms.append((time.perf_counter() - t0) * 1000)
    server_ms.append(data["latency_ms"])


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


for name, xs in [("client round-trip", client_ms), ("server latency_ms", server_ms)]:
    print(f"{name:18s} p50={pct(xs, 50):7.2f}  p95={pct(xs, 95):7.2f}  p99={pct(xs, 99):7.2f}  max={max(xs):7.2f}  (n={len(xs)})")