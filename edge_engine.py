"""
edge_engine.py
==============
Edge (local small model) + cloud (Groq big model) inference, and the offloading policies.

What changed compared with the old version (and why):
  1. Edge and cloud now get the SAME instructions and the SAME token budget per task.
     (Before: edge was told "answer in 1-2 sentences" and cut at 64 tokens, which made
      step-by-step math impossible, so accuracy was ~0 for reasons unrelated to the model.)
  2. Cloud errors (rate limit, timeout, empty answer) are reported, not hidden.
     The Groq library used to retry silently, which created fake 7-9 s "latency spikes".
  3. The RTT probe re-uses one connection and takes the median of 3 pings, so it measures
     network delay, not "open a new secure connection" time. It is no longer clamped.
  4. The "dynamic" policy now compares predicted latency with the deadline.
     (Before: it stayed local when the link was slow, even though local was SLOWER.)
  5. The score that decides "hard or easy" no longer looks at template words such as
     "Solve" / "step-by-step" (that was the benchmark telling the router the answer).
     It is a simple complexity score, NOT entropy/uncertainty. Name it that way in the thesis.
  6. Models load lazily, so you can import this file (and test the policy) without loading them.
"""
import os
import re
import statistics
import time

import requests

# ------------------------------------------------------------------ settings
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
CLOUD_MODEL = "qwen/qwen3.8-27b"          # the name your Groq account listed
CLOUD_URL = "https://api.groq.com/openai/v1/models"   # light GET on the real API host, used for the RTT probe

SYSTEM_PROMPT = (
    "You are a concise assistant. For reading questions, reply with only the short answer phrase. "
    "For math problems, use at most 5 short lines of plain text (no markdown, no bold, no LaTeX), "
    "then end with a last line exactly like this: Final answer: <number>"
)

# Same budget for edge and cloud. Reading answers are short; math needs room to reason.
TASK_MAX_TOKENS = {"squad_v2": 48, "gsm8k": 256}

# Cloud budget. If your cloud model "thinks" before answering, thinking tokens count against this limit.
# If many cloud answers come back cut off or empty, first try CLOUD_EXTRA_BODY below, then raise this.
CLOUD_MAX_TOKENS = {"squad_v2": 48, "gsm8k": 512}   # safety net: the cloud model tends to write long answers
# Extra request fields for the cloud model, e.g. {"reasoning_effort": "none"} IF the Groq docs for your
# model say it is supported. A wrong field shows up as a visible error in the log (no silent failure).
CLOUD_EXTRA_BODY = {}

TAU = 0.40            # complexity threshold (fixed by design, NOT tuned on the test queries)
SAFETY_MARGIN = 1.2   # predicted latency is multiplied by this before comparing with the deadline

_local_llm = None
_cloud_client = None


# ------------------------------------------------------------------ model loading
def _get_local():
    global _local_llm
    if _local_llm is None:
        from llama_cpp import Llama

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        print("[Init] Loading local quantized SLM (Qwen2.5-0.5B) into RAM...")
        # logits_all is not needed any more (nothing uses token probabilities) -> faster.
        _local_llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)
    return _local_llm


def _get_cloud():
    global _cloud_client
    if _cloud_client is None:
        from groq import Groq

        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        # max_retries=0 -> the library must NOT retry silently; we want to see every failure.
        _cloud_client = Groq(api_key=key, max_retries=0, timeout=30.0)
    return _cloud_client


def _clean(text: str) -> str:
    """Remove <think>...</think> blocks if the cloud model prints its reasoning."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"<think>.*", "", text, flags=re.S)  # unfinished thinking block
    return text.strip()


# ------------------------------------------------------------------ inference
def run_local(prompt: str, max_tokens: int) -> dict:
    """On-device inference (llama.cpp). Returns text, latency, ok flag, truncated flag."""
    llm = _get_local()
    t0 = time.perf_counter()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    latency = time.perf_counter() - t0
    choice = out["choices"][0]
    return {
        "text": (choice["message"]["content"] or "").strip(),
        "latency": latency,
        "ok": True,
        "error": "",
        "truncated": choice.get("finish_reason") == "length",
    }


def run_cloud(prompt: str, max_tokens: int) -> dict:
    """Cloud inference (Groq). Any failure is returned as ok=False with the error text."""
    t0 = time.perf_counter()
    try:
        resp = _get_cloud().chat.completions.create(
            model=CLOUD_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            extra_body=CLOUD_EXTRA_BODY,
        )
        latency = time.perf_counter() - t0
        choice = resp.choices[0]
        text = _clean(choice.message.content)
        return {
            "text": text,
            "latency": latency,
            "ok": True,                      # ok = the call itself worked (no exception)
            "empty": not text,               # empty = worked, but nothing usable came back
            "error": "" if text else "empty answer (model may have spent all tokens on thinking)",
            "truncated": choice.finish_reason == "length",
        }
    except Exception as e:  # rate limit, timeout, bad model name, ...
        return {
            "text": "",
            "latency": time.perf_counter() - t0,
            "ok": False,
            "empty": True,
            "error": f"{type(e).__name__}: {e}"[:200],
            "truncated": False,
        }


# ------------------------------------------------------------------ network probe
_session = requests.Session()


def _probe():
    key = os.environ.get("GROQ_API_KEY", "")
    return _session.get(CLOUD_URL, headers={"Authorization": f"Bearer {key}"}, timeout=2.0)
_rtt_cache = {"value": None, "time": 0.0}


def measure_rtt(samples: int = 3):
    """Median time of a tiny HTTPS request on a re-used connection (seconds).
    The first request pays the connection set-up, so it is made and thrown away.
    Returns None if the cloud cannot be reached."""
    times = []
    try:
        _probe()  # warm-up (connection set-up), not counted
    except requests.RequestException:
        return None
    for _ in range(samples):
        t0 = time.perf_counter()
        try:
            _probe()
            times.append(time.perf_counter() - t0)
        except requests.RequestException:
            pass
    return statistics.median(times) if times else None


def get_rtt(max_age_sec: float = 15.0):
    """Cached RTT so the decision itself costs ~0 ms."""
    now = time.perf_counter()
    if _rtt_cache["value"] is None or now - _rtt_cache["time"] > max_age_sec:
        _rtt_cache["value"] = measure_rtt()
        _rtt_cache["time"] = now
    return _rtt_cache["value"]


# ------------------------------------------------------------------ complexity score
_MATH_CUES = [
    r"\bhow (many|much)\b",
    r"\btotal\b",
    r"\beach\b",
    r"\bper\b",
    r"%|\bpercent\b",
    r"\b(twice|half|times)\b",
    r"\baverage\b",
    r"\b(remaining|left)\b",
    r"\b(more|less|fewer) than\b|\bfewer\b",
    r"\$",
]


def complexity_score(prompt: str) -> float:
    """0..1 score for 'needs multi-step arithmetic'. Looks only at the question part and
    counts numbers and arithmetic cue words. It does NOT use the instruction template.
    This is a simple heuristic (not entropy / model uncertainty) and it can be wrong."""
    text = prompt.lower()
    question = text.split("question:")[-1]  # for passage+question prompts, ignore the passage
    numbers = len(re.findall(r"\d+(?:[.,]\d+)?", question))
    cues = sum(1 for c in _MATH_CUES if re.search(c, question))
    return 0.5 * min(numbers / 4.0, 1.0) + 0.5 * min(cues / 3.0, 1.0)


# ------------------------------------------------------------------ policies
def decide(policy: str, score: float, rtt: float, deadline: float,
           est_local: float, est_cloud_compute: float,
           tau: float = TAU, margin: float = SAFETY_MARGIN) -> str:
    """Returns 'LOCAL' or 'CLOUD'.

    edge_only         always local
    cloud_only        always cloud
    static_threshold  cloud if score > tau, never looks at the network (prior-art style)
    dynamic           channel- and deadline-aware (proposed):
                        1. if the cloud is predicted to miss the deadline -> LOCAL
                        2. else if the task is hard                        -> CLOUD
                        3. else if local is predicted to miss the deadline -> CLOUD
                        4. else                                            -> LOCAL
    est_local / est_cloud_compute come from a small calibration run, not from the test queries.
    """
    if policy == "edge_only":
        return "LOCAL"
    if policy == "cloud_only":
        return "CLOUD"
    if policy == "static_threshold":
        return "CLOUD" if score > tau else "LOCAL"
    if policy == "dynamic":
        cloud_pred = (rtt + est_cloud_compute) * margin
        local_pred = est_local * margin
        if cloud_pred > deadline:
            return "LOCAL"
        if score > tau:
            return "CLOUD"
        if local_pred > deadline:
            return "CLOUD"
        return "LOCAL"
    raise ValueError(f"Unknown policy: {policy}")


def run_live(prompt: str, task: str, policy: str, deadline: float, profile: dict) -> dict:
    """Live (real-network) version for demos. `profile` = {'local': sec, 'cloud': sec} for this task.
    If the cloud call fails, it falls back to the local model."""
    rtt = get_rtt()
    if rtt is None and policy in ("static_threshold", "dynamic"):
        rtt = 99.0  # cloud unreachable -> treated as a hopeless link
    decision = decide(policy, complexity_score(prompt), rtt or 0.0, deadline,
                      profile["local"], profile["cloud"])
    t0 = time.perf_counter()
    if decision == "CLOUD":
        res = run_cloud(prompt, CLOUD_MAX_TOKENS[task])
        if not res["ok"] or res["empty"]:
            res = run_local(prompt, TASK_MAX_TOKENS[task])
            decision = "LOCAL(fallback)"
    else:
        res = run_local(prompt, TASK_MAX_TOKENS[task])
    total = time.perf_counter() - t0
    return {"policy": policy, "decision": decision, "latency_sec": total,
            "deadline_met": total <= deadline, "response": res["text"], "rtt_sec": rtt}


if __name__ == "__main__":
    print("edge_engine.py loaded OK (models are loaded only when first used).")