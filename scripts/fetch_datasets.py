import json
import os
from datasets import load_dataset

os.makedirs("data", exist_ok=True)
output_path = "data/benchmark_queries.json"

benchmark_subset = []
query_id = 1

print("[1/2] Downloading SQuAD v2 validation split...")
squad = load_dataset("rajpurkar/squad_v2", split="validation")

count = 0
for item in squad:
    if len(item["answers"]["text"]) > 0:
        context = item["context"]
        question = item["question"]
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer directly in a few words."
        ground_truth = item["answers"]["text"][0]
        
        benchmark_subset.append({
            "id": query_id,
            "dataset": "squad_v2",
            "complexity": "reading_comprehension",
            "prompt": prompt,
            "ground_truth": ground_truth
        })
        query_id += 1
        count += 1
        if count == 100:
            break

print(f"Loaded {count} samples from SQuAD v2.")

print("[2/2] Downloading GSM8K test split...")
gsm8k = load_dataset("openai/gsm8k", "main", split="test")

count = 0
for item in gsm8k:
    question = item["question"]
    full_answer = item["answer"]
    numeric_answer = full_answer.split("####")[-1].strip() if "####" in full_answer else full_answer
    
    prompt = f"Solve this math problem step-by-step. State the final numerical answer clearly at the end:\n{question}"
    
    benchmark_subset.append({
        "id": query_id,
        "dataset": "gsm8k",
        "complexity": "multi_step_reasoning",
        "prompt": prompt,
        "ground_truth": numeric_answer
    })
    query_id += 1
    count += 1
    if count == 100:
        break

print(f"Loaded {count} samples from GSM8K.")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(benchmark_subset, f, indent=2, ensure_ascii=False)

print(f"\nSUCCESS: Created {output_path} with {len(benchmark_subset)} queries.")