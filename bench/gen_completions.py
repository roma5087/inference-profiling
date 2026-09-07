#!/usr/bin/env python3
"""Send the Stage 0 prompt sets to a running OpenAI-compatible server and record completions."""
import argparse
import json
import time
from pathlib import Path

import requests

BENCH_DIR = Path(__file__).parent
RESULTS_DIR = BENCH_DIR.parent / "results"


def load_prompts():
    sanity = json.loads((BENCH_DIR / "prompts_sanity.json").read_text())
    deterministic = json.loads((BENCH_DIR / "prompts_deterministic.json").read_text())
    return sanity, deterministic


def complete(base_url, model, prompt, max_tokens=256):
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["vllm", "sglang"])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = ap.parse_args()

    base_url = f"http://localhost:{args.port}"
    sanity, deterministic = load_prompts()

    out = {"engine": args.engine, "sanity": [], "deterministic": []}

    for prompt in sanity:
        t0 = time.time()
        text = complete(base_url, args.model, prompt)
        out["sanity"].append({"prompt": prompt, "completion": text, "latency_s": round(time.time() - t0, 2)})
        print(f"[sanity] {prompt[:50]!r} -> {text[:60]!r}")

    for item in deterministic:
        t0 = time.time()
        text = complete(base_url, args.model, item["prompt"])
        out["deterministic"].append({
            "prompt": item["prompt"],
            "expected_answer": item["expected_answer"],
            "completion": text,
            "latency_s": round(time.time() - t0, 2),
        })
        print(f"[deterministic] {item['prompt']!r} -> {text[:60]!r}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{args.engine}_stage0_completions.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
