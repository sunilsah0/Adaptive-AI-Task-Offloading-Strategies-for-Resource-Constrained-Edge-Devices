import json
import os
import re
import string
import time
import pandas as pd
from edge_engine import execute_local_inference, execute_cloud_inference, estimate_uncertainty

DATA_PATH = "data/benchmark_queries.json"
RESULTS_PATH = "results/channel_variation_results.csv"

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens) if pred_tokens else 0.0
    r = len(common) / len(gt_tokens) if gt_tokens else 0.0
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

def evaluate_accuracy(dataset: str, prediction: str, ground_truth: str) -> float:
    if dataset == "squad_v2":
        return compute_f1(prediction, ground_truth)
    elif dataset == "gsm8k":
        gt_target = ground_truth.strip().replace(",", "")
        pred_clean = prediction.replace(",", "")
        if re.search(r'\b' + re.escape(gt_target) + r'\b', pred_clean):
            return 1.0
        nums = re.findall(r"[-+]?\d*\.?\d+", pred_clean)
        return 1.0 if nums and nums[-1].rstrip('.') == gt_target else 0.0
    return 0.0

def run_scenario_benchmarks():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    # 10 SQuAD + 10 GSM8K = 20 queries evaluated across network link states
    queries = [q for q in all_queries if q["dataset"] == "squad_v2"][:10] + \
              [q for q in all_queries if q["dataset"] == "gsm8k"][:10]

    scenarios = [
        {"name": "Ideal_WiFi", "channel_rtt": 0.08, "deadline": 2.5},
        {"name": "Degraded_Cellular_4G", "channel_rtt": 0.75, "deadline": 2.5}
    ]

    policies = ["edge_only", "cloud_only", "static_threshold", "dynamic_cuao"]
    records = []

    print("\n====================================================================")
    print("RUNNING CHANNEL-AWARE SENSITIVITY BENCHMARK (IDEAL VS DEGRADED LINK)")
    print("====================================================================")

    for sc in scenarios:
        sc_name = sc["name"]
        rtt_sim = sc["channel_rtt"]
        deadline = sc["deadline"]
        print(f"\n---> SCENARIO: {sc_name} (RTT: {rtt_sim*1000:.0f}ms | Deadline: {deadline}s) <---")

        for pol in policies:
            for q in queries:
                t0 = time.perf_counter()
                prompt = q["prompt"]
                uncertainty = estimate_uncertainty(prompt)

                if pol == "edge_only":
                    resp, raw_lat = execute_local_inference(prompt)
                    tot_lat = raw_lat
                    decision = "LOCAL"

                elif pol == "cloud_only":
                    resp, raw_lat = execute_cloud_inference(prompt)
                    tot_lat = raw_lat + rtt_sim
                    decision = "OFFLOADED"

                elif pol == "static_threshold":
                    # Static threshold ignores channel RTT
                    if uncertainty > 1.5:
                        resp, raw_lat = execute_cloud_inference(prompt)
                        tot_lat = raw_lat + rtt_sim
                        decision = "OFFLOADED"
                    else:
                        resp, raw_lat = execute_local_inference(prompt)
                        tot_lat = raw_lat
                        decision = "LOCAL"

                elif pol == "dynamic_cuao":
                    # Dynamic threshold adapts to channel state
                    dynamic_tau = 1.4 * (1.0 + (rtt_sim / 0.350))
                    remaining = deadline - rtt_sim
                    should_offload = (uncertainty > dynamic_tau) and (remaining > (rtt_sim * 1.5))
                    
                    if should_offload:
                        resp, raw_lat = execute_cloud_inference(prompt)
                        tot_lat = raw_lat + rtt_sim
                        decision = "OFFLOADED"
                    else:
                        resp, raw_lat = execute_local_inference(prompt)
                        tot_lat = raw_lat
                        decision = "LOCAL"

                score = evaluate_accuracy(q["dataset"], resp, q["ground_truth"])
                records.append({
                    "scenario": sc_name,
                    "policy": pol,
                    "query_id": q["id"],
                    "dataset": q["dataset"],
                    "decision": decision,
                    "latency_sec": round(tot_lat, 3),
                    "deadline_met": tot_lat <= deadline,
                    "accuracy": round(score, 3)
                })

    df = pd.DataFrame(records)
    os.makedirs("results", exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)

    print("\n" + "="*85)
    print("               FINAL IEEE THESIS RESULTS ACROSS CHANNEL PROFILES")
    print("="*85)
    summary = df.groupby(["scenario", "policy"]).agg(
        Mean_Latency=("latency_sec", "mean"),
        SLO_Compliance=("deadline_met", lambda x: f"{x.mean()*100:.1f}%"),
        Mean_Accuracy=("accuracy", "mean"),
        Offload_Rate=("decision", lambda x: f"{(x == 'OFFLOADED').mean()*100:.1f}%")
    ).reset_index()
    print(summary.to_string(index=False))
    print("="*85 + "\n")

if __name__ == "__main__":
    run_scenario_benchmarks()