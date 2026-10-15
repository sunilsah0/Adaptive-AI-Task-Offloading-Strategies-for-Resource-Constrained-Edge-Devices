"""
diagnose_cloud.py - look at what the cloud model really returned (run after benchmark.py).
Usage:  python diagnose_cloud.py            (reads results/traces.json)
"""
import json

with open("results/traces.json", encoding="utf-8") as f:
    tr = json.load(f)["eval"]

if tr and "cloud_empty" not in tr[0]:
    print("NOTE: this traces.json was made by the OLD benchmark.py. The lines 'empty answers', "
          "'retried' and 'wasted time' below are NOT reliable for it. 'cut off' and the texts are.")

for task in ["squad_v2", "gsm8k"]:
    t = [x for x in tr if x["dataset"] == task]
    print(f"\n=== {task}: {len(t)} queries ===")
    print("cut off at token limit :", sum(x["cloud_trunc"] for x in t))
    print("empty answers          :", sum(x.get("cloud_empty", False) for x in t))
    print("retried after errors   :", sum(x["cloud_attempts"] > 1 for x in t))
    print("failed for good        :", sum(not x["cloud_ok"] for x in t))
    print("wasted retry time (s)  :", round(sum(x.get("cloud_wasted", 0.0) for x in t), 1))
    errs = {x["cloud_error"] for x in t if x["cloud_error"]}
    print("error messages seen    :", errs or "none")

print("\n=== first 4 GSM8K cloud answers that were cut off (last 250 characters) ===")
shown = 0
for x in tr:
    if x["dataset"] == "gsm8k" and x["cloud_trunc"]:
        print(f"\nQ{x['id']} (correct answer: {x['ground_truth']})\n...{x['cloud_text'][-250:]!r}")
        shown += 1
        if shown == 4:
            break
if shown == 0:
    print("none - good")