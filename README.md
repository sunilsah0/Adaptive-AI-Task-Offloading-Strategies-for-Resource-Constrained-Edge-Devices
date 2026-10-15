# Adaptive Edge-Cloud Collaborative Inference: Channel and Complexity-Aware Offloading (CUAO)

This repository contains the source code, experimental framework, and LaTeX documentation for the Master's Degree Project in Computer and Systems Sciences at Stockholm University[cite: 145, 146].

* **Author:** Sunil Kumar Sah[cite: 145, 146]
* **Supervisor:** Associate Professor Praveen Kumar Donta[cite: 145, 150]
* **Department:** Department of Computer and Systems Sciences (DSV), Stockholm University[cite: 145, 146]

---

## 📌 Abstract & Overview

Small language models (SLMs) can run locally on edge devices without network connectivity, but they are clearly less accurate than large cloud-hosted models on multi-step reasoning tasks[cite: 147, 159]. Conversely, while cloud models offer high accuracy, network latency variability can break the strict Service Level Objective (SLO) deadlines of interactive applications[cite: 147, 159]. Existing routing schemes between small and large models mostly decide on query difficulty alone and ignore the state of the network[cite: 147, 159].

This thesis introduces **CUAO (Channel and Uncertainty-Aware Offloading)**, an artifact that pairs a 4-bit quantized `Qwen2.5-0.5B-Instruct` model running via `llama.cpp` on a CPU with a 27-billion-parameter cloud model served by Groq[cite: 147, 168, 169]. It uses a lightweight lexical complexity score, a cached HTTPS Round-Trip Time (RTT) probe, and a four-rule policy to make safe, adaptive offloading decisions[cite: 147, 172, 173, 174].

---

## 🔍 Research Questions

The research is guided by three core research questions[cite: 160]:
1. **RQ1:** How can a resource-constrained device estimate the complexity of a language task before generation without the overhead of token-probability sampling?[cite: 160]
2. **RQ2:** How can measured network round-trip time and an application deadline be integrated into a routing policy that prevents deadline violations under adverse channel conditions?[cite: 160]
3. **RQ3:** To what extent does a channel- and deadline-aware policy outperform static thresholding, edge-only execution, and cloud-only execution across network conditions?[cite: 160]

---

## 🏗️ System Architecture & Artifact Design

CUAO evaluates incoming queries through a five-step controller pipeline[cite: 171]:
1. **Complexity Score ($S(q)$):** Computes numerical and arithmetic cue density from the question text ($S \in [0, 1]$)[cite: 171, 172].
2. **RTT Probe:** Measures path delay over a persistent, warm HTTPS session with a 15-second cache[cite: 172, 173].
3. **Policy Evaluation:** Evaluates four rules in order[cite: 174]:
   - *Rule 1 (Safety Override):* If predicted cloud latency exceeds the deadline ($L_{cloud} > T$), choose local[cite: 174].
   - *Rule 2 (Complexity Escalation):* If the score exceeds threshold ($\tau = 0.40$), choose cloud[cite: 174].
   - *Rule 3 (Local Bottleneck Relief):* If predicted local latency exceeds deadline, choose cloud[cite: 174].
   - *Rule 4 (Edge Default):* Default to local execution[cite: 174].
4. **Execution:** Executes locally via `llama.cpp` or remotely via the Groq API[cite: 171].
5. **Fallback:** Live engine handles cloud failures by falling back to the local model[cite: 171].

---

## 📊 Summary of Key Results

* **Normal Network Conditions:** Cloud-only execution achieves the best single performance (on-time accuracy `0.938`)[cite: 147, 176]. CUAO and static thresholding match each other (`0.813` on-time accuracy) while reducing cloud calls by **49%**, significantly outperforming edge-only execution (`0.513`)[cite: 147, 176].
* **Degraded Links (+3s Delay):** Cloud-only on-time accuracy drops to `0.490`, falling below the routed policies (`0.813`)[cite: 147, 176].
* **Network Outages (+30s):** Cloud-only execution achieves `0%` deadline compliance, and static thresholding achieves `33.3%` on-time accuracy[cite: 147, 176]. In contrast, CUAO's feasibility rule detects the outage, routes all queries locally, and preserves **94% deadline compliance** (`0.513` on-time accuracy)[cite: 147, 176].
* **Parameter Sweep:** Across 27 delay and deadline combinations, CUAO was never worse than the static threshold and was strictly better in 12 cells[cite: 147, 178, 179].

---

## 🗂️ Repository Structure

* `benchmark.py` — Main script for trace collection, scenario evaluation, and policy replay[cite: 168]
* `edge_engine.py` — Implements model wrappers, complexity scoring, RTT probe, and CUAO decision rules[cite: 168]
* `models/` — Directory for local GGUF model files (excluded from Git tracking due to size limits)
* `requirements.txt` — Python dependencies (`numpy`, `pandas`, `llama-cpp-python`, `requests`, `matplotlib`)[cite: 169]

---

## ⚙️ Setup and Reproduction

### 1. Prerequisites & Environment Setup
```powershell
python -m venv thesis_env
thesis_env\Scripts\Activate
pip install -r requirements.txt
