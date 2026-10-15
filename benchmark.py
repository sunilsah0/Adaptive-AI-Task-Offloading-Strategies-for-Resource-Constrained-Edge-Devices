"""
benchmark.py
============
Fair, repeatable comparison of 4 policies: edge_only, cloud_only, static_threshold, dynamic.

How it works (simple version):
  Step 1  Each test question is answered ONCE by the local model and ONCE by the cloud model.
          Answers, times, errors are saved to results/traces.json.
  Step 2  The 4 policies are then "replayed" on those saved answers. A policy only chooses
          which saved answer to use. So every policy sees exactly the same model outputs and
          the only difference between them is the routing decision.
  Step 3  The network is EMULATED: for each scenario an extra delay is added to the cloud time.
          State this clearly in the thesis ("emulated channel"); it is not a real bad network.

Fixes compared with the old version:
  * GSM8K is scored on the FINAL number only (old code gave 1.0 if the right number appeared
    anywhere in the text, and compared "1596.00" with "1596" as different).
  * SQuAD F1 uses the official word-count overlap (old code used a set).
  * Test questions are a seeded random sample (old code took the first 15 = one Wikipedia article),
    and thresholds/latency profiles come from a separate calibration set, not the test set.
  * Failed cloud calls are retried and logged; a query that still fails is dropped for ALL policies.
  * Results include 95% confidence intervals, truncation counts and "on-time accuracy".

Run:
  python benchmark.py --n 50            (first real run; ~15-20 min, mostly the local math answers)
  python benchmark.py --n 50 --reuse    (re-use saved answers, changes in policy/scoring take seconds)
Close results\\*.csv in Excel before running (that was the PermissionError).
"""
import argparse
import json
import math
import os
import random
import re
import string
import time
from collections import Counter

import numpy as np
import pandas as pd

import edge_engine as eng

TASKS = ["squad_v2", "gsm8k"]
POLICIES = ["edge_only", "cloud_only", "static_threshold", "dynamic"]

# Emulated network scenarios. "added" = extra delay in seconds ON TOP of your real link
# (your real link is already inside the measured cloud times). Say "emulated" in the thesis.
SCENARIOS = [
    {"name": "Normal_link", "added": 0.0},     # exactly what was measured, nothing added
    {"name": "Added_0.3s", "added": 0.3},
    {"name": "Added_1s", "added": 1.0},
    {"name": "Added_3s", "added": 3.0},
    {"name": "Outage_30s", "added": 30.0},     # link down: the cloud call only ends at the 30 s timeout
]
# Deadline (SLO) per task, seconds. Reading questions are interactive; math with step-by-step
# reasoning is allowed longer. Justify these numbers in the thesis and show the sweep.
DEADLINES = {"squad_v2": 2.5, "gsm8k": 12.0}


# ============================================================ scoring
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    """Official SQuAD token F1 (counts repeated words correctly)."""
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def extract_final_number(text: str):
    """'Final answer: 1,596.00' -> 1596.0 ; otherwise the LAST number in the text."""
    t = text.lower()
    marked = re.findall(r"final answer\s*[:\-]?\s*\$?\s*(" + _NUM + ")", t)
    found = marked if marked else re.findall(_NUM, t)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def evaluate_accuracy(dataset: str, prediction: str, ground_truth: str) -> float:
    if dataset == "squad_v2":
        return f1_score(prediction, ground_truth)
    if dataset == "gsm8k":
        pred = extract_final_number(prediction)
        gold = float(ground_truth.replace(",", "").strip())
        return 1.0 if pred is not None and abs(pred - gold) < 1e-6 else 0.0
    return 0.0


# ============================================================ data
def load_queries(path: str, n: int, cal_n: int, seed: int):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(seed)
    cal, ev = [], []
    for task in TASKS:
        items = [q for q in data if q["dataset"] == task]
        rng.shuffle(items)
        cal += items[:cal_n]               # used only to measure typical speeds
        ev += items[cal_n:cal_n + n]       # the test queries (never used for tuning)
    return cal, ev


# ============================================================ step 1: collect answers
def collect_cloud(prompt: str, budget: int, pacing: float, max_attempts: int = 4) -> dict:
    """Call the cloud. Only real errors (rate limit, timeout...) are retried.
    The time of every failed attempt is added to `wasted`, so hidden retries can not make the cloud look faster."""
    wasted, attempts, res = 0.0, 0, None
    while attempts < max_attempts:
        attempts += 1
        res = eng.run_cloud(prompt, budget)
        if res["ok"]:
            break
        wasted += res["latency"]
        time.sleep(2.0 * attempts)  # waiting between retries is not counted
    time.sleep(pacing)              # stay under the free-tier rate limit
    res["wasted"], res["attempts"] = wasted, attempts
    return res


def collect_one(q: dict, pacing: float) -> dict:
    task = q["dataset"]
    local = eng.run_local(q["prompt"], eng.TASK_MAX_TOKENS[task])
    cloud = collect_cloud(q["prompt"], eng.CLOUD_MAX_TOKENS[task], pacing)
    return {
        "id": q["id"], "dataset": task, "prompt": q["prompt"],
        "ground_truth": q["ground_truth"], "score": eng.complexity_score(q["prompt"]),
        "local_text": local["text"], "local_lat": local["latency"], "local_trunc": local["truncated"],
        "cloud_text": cloud["text"], "cloud_lat": cloud["latency"], "cloud_trunc": cloud["truncated"],
        "cloud_ok": cloud["ok"], "cloud_empty": cloud["empty"], "cloud_error": cloud["error"],
        "cloud_attempts": cloud["attempts"], "cloud_wasted": cloud["wasted"],
    }


def redo_cloud(traces, pacing, label):
    """Keep the saved LOCAL answers, call the cloud again (use after changing cloud settings)."""
    for i, tr in enumerate(traces, 1):
        c = collect_cloud(tr["prompt"], eng.CLOUD_MAX_TOKENS[tr["dataset"]], pacing)
        tr.update({"cloud_text": c["text"], "cloud_lat": c["latency"], "cloud_trunc": c["truncated"],
                   "cloud_ok": c["ok"], "cloud_empty": c["empty"], "cloud_error": c["error"],
                   "cloud_attempts": c["attempts"], "cloud_wasted": c["wasted"]})
        print(f"  [{label} {i:>3}/{len(traces)}] Q{tr['id']:<3} {tr['dataset']:<8} cloud {c['latency']:.2f}s "
              f"{'EMPTY' if c['empty'] and c['ok'] else ('OK' if c['ok'] else 'FAILED: ' + c['error'])}")


def collect(queries, pacing, label):
    out = []
    for i, q in enumerate(queries, 1):
        tr = collect_one(q, pacing)
        out.append(tr)
        print(f"  [{label} {i:>3}/{len(queries)}] Q{q['id']:<3} {q['dataset']:<8} "
              f"local {tr['local_lat']:.2f}s | cloud {tr['cloud_lat']:.2f}s "
              f"{'EMPTY' if tr['cloud_empty'] and tr['cloud_ok'] else ('OK' if tr['cloud_ok'] else 'FAILED: ' + tr['cloud_error'])}")
    return out


# ============================================================ step 2: replay policies
def make_profile(cal_traces, real_rtt):
    prof = {}
    for task in TASKS:
        rows = [t for t in cal_traces if t["dataset"] == task and t["cloud_ok"]]
        prof[task] = {
            "local": float(np.mean([t["local_lat"] for t in rows])),
            "cloud": max(0.0, float(np.mean([t["cloud_lat"] + t.get("cloud_wasted", 0.0) for t in rows])) - real_rtt),
        }
    return prof


def replay(traces, policy, scenario, deadlines, profile, real_rtt):
    extra = scenario["added"]         # emulated extra delay on top of the real link
    rtt = real_rtt + extra            # what the controller "measures"
    rows = []
    for tr in traces:
        task = tr["dataset"]
        deadline = deadlines[task]
        decision = eng.decide(policy, tr["score"], rtt, deadline,
                              profile[task]["local"], profile[task]["cloud"])
        if decision == "CLOUD":
            lat, text = tr["cloud_lat"] + tr.get("cloud_wasted", 0.0) + extra, tr["cloud_text"]
        else:
            lat, text = tr["local_lat"], tr["local_text"]
        acc = evaluate_accuracy(task, text, tr["ground_truth"])
        met = lat <= deadline
        rows.append({
            "scenario": scenario["name"], "rtt_sec": round(rtt, 3), "added_delay_sec": extra, "policy": policy,
            "query_id": tr["id"], "dataset": task, "decision": decision,
            "latency_sec": round(lat, 3), "deadline_sec": deadline, "deadline_met": met,
            "accuracy": round(acc, 3), "ontime_accuracy": round(acc if met else 0.0, 3),
            "response": text[:300].replace("\n", " "),
        })
    return rows


def ci95_mean(x):
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")


def summarize(df):
    rows = []
    for (sc, pol), d in df.groupby(["scenario", "policy"], sort=False):
        rows.append({
            "scenario": sc, "policy": pol, "n": len(d),
            "mean_latency_s": d.latency_sec.mean(),
            "SLO_%": 100 * d.deadline_met.mean(),
            "SLO_ci95": 100 * ci95_mean(d.deadline_met.astype(float)),
            "accuracy": d.accuracy.mean(), "acc_ci95": ci95_mean(d.accuracy),
            "acc_squad": d[d.dataset == "squad_v2"].accuracy.mean(),
            "acc_gsm8k": d[d.dataset == "gsm8k"].accuracy.mean(),
            "on_time_accuracy": d.ontime_accuracy.mean(),
            "offload_%": 100 * (d.decision == "CLOUD").mean(),
        })
    return pd.DataFrame(rows)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/benchmark_queries.json")
    ap.add_argument("--n", type=int, default=50, help="test queries per dataset")
    ap.add_argument("--cal", type=int, default=5, help="calibration queries per dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pacing", type=float, default=1.0, help="seconds to wait between cloud calls")
    ap.add_argument("--reuse", action="store_true", help="use saved answers in results/traces.json")
    ap.add_argument("--redo-cloud", action="store_true",
                    help="with --reuse: keep saved local answers, call the cloud again")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    trace_path = os.path.join(args.out, "traces.json")

    if args.reuse and os.path.exists(trace_path):
        with open(trace_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        cal_tr, ev_tr, real_rtt = saved["cal"], saved["eval"], saved["real_rtt"]
        print(f"[Reuse] loaded {len(cal_tr)} calibration + {len(ev_tr)} test traces from {trace_path}")
        if args.redo_cloud:
            real_rtt = eng.measure_rtt() or real_rtt
            print(f"[Info] real RTT to cloud: {real_rtt * 1000:.0f} ms")
            redo_cloud(cal_tr, args.pacing, "cal")
            redo_cloud(ev_tr, args.pacing, "test")
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump({"cal": cal_tr, "eval": ev_tr, "real_rtt": real_rtt}, f, ensure_ascii=False, indent=1)
    else:
        cal_q, ev_q = load_queries(args.data, args.n, args.cal, args.seed)
        real_rtt = eng.measure_rtt()
        if real_rtt is None:
            raise SystemExit("Cloud not reachable - check internet / GROQ_API_KEY.")
        print(f"[Info] real RTT to cloud: {real_rtt * 1000:.0f} ms")
        print(f"\n--- Calibration ({len(cal_q)} queries) ---")
        cal_tr = collect(cal_q, args.pacing, "cal")
        print(f"\n--- Test ({len(ev_q)} queries) ---")
        ev_tr = collect(ev_q, args.pacing, "test")
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump({"cal": cal_tr, "eval": ev_tr, "real_rtt": real_rtt}, f, ensure_ascii=False, indent=1)

    # queries where the cloud kept failing are removed for every policy (keeps the comparison paired)
    dropped = [t for t in ev_tr if not t["cloud_ok"]]
    ev_tr = [t for t in ev_tr if t["cloud_ok"]]
    profile = make_profile(cal_tr, real_rtt)

    print("\n" + "=" * 78)
    print("DATA QUALITY CHECKS (look at these before trusting any result)")
    print("=" * 78)
    print(f"Test queries used: {len(ev_tr)}  | dropped (cloud failed after retries): {len(dropped)}")
    for task in TASKS:
        t = [x for x in ev_tr if x["dataset"] == task]
        if not t:
            continue
        above = np.mean([x["score"] > eng.TAU for x in t]) * 100
        print(f"{task:<9} n={len(t):<3} | complexity score mean {np.mean([x['score'] for x in t]):.2f}, "
              f"{above:.0f}% above threshold {eng.TAU} | "
              f"truncated: local {sum(x['local_trunc'] for x in t)}, cloud {sum(x['cloud_trunc'] for x in t)} | "
              f"cloud: EMPTY {sum(x.get('cloud_empty', False) for x in t)}, "
              f"retried after errors {sum(x['cloud_attempts'] > 1 for x in t)}, "
              f"wasted time {sum(x.get('cloud_wasted', 0.0) for x in t):.1f}s")
    print(f"Profile from calibration (seconds): { {k: {a: round(b, 2) for a, b in v.items()} for k, v in profile.items()} }")
    print("Model-only accuracy (no routing):")
    for task in TASKS:
        t = [x for x in ev_tr if x["dataset"] == task]
        if t:
            la = np.mean([evaluate_accuracy(task, x["local_text"], x["ground_truth"]) for x in t])
            ca = np.mean([evaluate_accuracy(task, x["cloud_text"], x["ground_truth"]) for x in t])
            print(f"   {task:<9} local {la:.3f} | cloud {ca:.3f}")

    # ---- replay all scenarios and policies
    all_rows = []
    for sc in SCENARIOS:
        for pol in POLICIES:
            all_rows += replay(ev_tr, pol, sc, DEADLINES, profile, real_rtt)
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out, "per_query_results.csv"), index=False)
    summary = summarize(df)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    print("\n" + "=" * 78)
    print(f"RESULTS (emulated network, n={len(ev_tr)} per policy, deadlines {DEADLINES})")
    print("=" * 78)
    print(summary.round(3).to_string(index=False))

    # ---- sweep: when does channel-awareness matter?
    sweep = []
    for added in [0.0, 0.1, 0.3, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0]:
        for scale in [0.5, 1.0, 2.0]:
            dl = {k: v * scale for k, v in DEADLINES.items()}
            for pol in POLICIES:
                r = pd.DataFrame(replay(ev_tr, pol, {"name": "sweep", "added": added}, dl, profile, real_rtt))
                sweep.append({"rtt": real_rtt + added, "added_delay": added, "deadline_scale": scale, "policy": pol,
                              "SLO_%": 100 * r.deadline_met.mean(), "accuracy": r.accuracy.mean(),
                              "on_time_accuracy": r.ontime_accuracy.mean(),
                              "offload_%": 100 * (r.decision == "CLOUD").mean()})
    pd.DataFrame(sweep).to_csv(os.path.join(args.out, "sweep.csv"), index=False)
    print(f"\nSaved: {args.out}/traces.json, per_query_results.csv, summary.csv, sweep.csv")
    print("If 'dynamic' is not better than 'static_threshold' anywhere, report that honestly.")


if __name__ == "__main__":
    main()