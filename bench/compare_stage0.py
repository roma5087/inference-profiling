#!/usr/bin/env python3
"""Apply Stage 0 pass criteria to vLLM vs SGLang completions."""
import json
import re
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load(engine):
    path = RESULTS_DIR / f"{engine}_stage0_completions.json"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run gen_completions.py --engine {engine} first")
    return json.loads(path.read_text())


def looks_like_garbage(text):
    if not text or not text.strip():
        return "empty completion"
    words = text.split()
    if len(words) >= 8:
        # crude repetition-loop detector: same trigram repeated back-to-back
        trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
        for i in range(len(trigrams) - 3):
            if trigrams[i] == trigrams[i + 1] == trigrams[i + 2]:
                return f"repetition loop around: {trigrams[i]!r}"
    return None


def normalize(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main():
    vllm = load("vllm")
    sglang = load("sglang")

    print("=" * 70)
    print("HARD FAIL CHECKS: garbage on sanity prompts")
    print("=" * 70)
    hard_fail = False
    for engine, data in [("vllm", vllm), ("sglang", sglang)]:
        for item in data["sanity"]:
            problem = looks_like_garbage(item["completion"])
            if problem:
                hard_fail = True
                print(f"[FAIL][{engine}] {item['prompt'][:50]!r}: {problem}")
    if not hard_fail:
        print("PASS -- no garbage detected on either engine across sanity prompts")

    print()
    print("=" * 70)
    print("HARD FAIL CHECKS: deterministic-prompt answer agreement")
    print("=" * 70)
    v_det = {i["prompt"]: i for i in vllm["deterministic"]}
    s_det = {i["prompt"]: i for i in sglang["deterministic"]}
    det_fail = False
    for prompt, v_item in v_det.items():
        s_item = s_det.get(prompt)
        expected = normalize(v_item["expected_answer"])
        v_has = expected in normalize(v_item["completion"])
        s_has = expected in normalize(s_item["completion"]) if s_item else False
        status = "OK" if (v_has and s_has) else "CHECK"
        if not (v_has and s_has):
            det_fail = True
        print(f"[{status}] {prompt!r} expected={v_item['expected_answer']!r} "
              f"vllm_has_it={v_has} sglang_has_it={s_has}")
    if det_fail:
        print("\nFAIL -- one or both engines missed an expected answer; inspect manually before proceeding")
    else:
        print("\nPASS -- both engines produced the expected answer on every deterministic prompt")

    print()
    print("=" * 70)
    print("BASELINE (not gated): exact-completion match rate on sanity prompts")
    print("=" * 70)
    matches = sum(
        1 for v_item, s_item in zip(vllm["sanity"], sglang["sanity"])
        if v_item["completion"].strip() == s_item["completion"].strip()
    )
    print(f"{matches}/{len(vllm['sanity'])} sanity completions were exact string matches between "
          f"engines (not required -- bf16 + different kernels means drift is expected; this is a "
          f"baseline number for the write-up, not a gate)")


if __name__ == "__main__":
    main()
