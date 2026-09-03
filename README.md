# Distributed Inference Profiling

Serve the same model on vLLM and SGLang, find a real throughput or latency gap between
them, and prove — with Nsight Systems and the PyTorch profiler, not just a benchmark
number — what causes it.

## Locked parameters

| Parameter | Value | Why it's fixed |
|---|---|---|
| Model | Llama-3-8B, dense | Smallest loop with one confound at a time. MoE and tensor-parallel are follow-ons — each adds a second variable before the first is resolved. |
| Precision | bf16 | FP8 / AWQ paths aren't implemented identically across engines. A quant mismatch is a confound that leaks into the nsys diff as a false signal. |
| GPU | A100 / H100 80GB | Keeps an 8B model off the memory-bandwidth wall, so a gap is more likely to surface as scheduling / kernel-launch overhead rather than disappearing into a shared bandwidth floor. |
| Topology | 1x GPU | Isolates scheduler and kernel differences before NCCL and cross-device traffic enter the trace. |
| Prompt shape | fixed length (Stage 1 baseline) | Keeps nsys traces readable on the first pass. A ShareGPT-style distribution comes later as a robustness check, not primary evidence. |

Model checkpoints/weights are not committed here (re-downloadable, not the artifact) — see `.gitignore`. Everything else — scripts, configs, traces, parsed results, the write-up — lives in this repo.

## Plan

- [ ] **Stage −1 — GPU selection.** Choose the card because of what it will show, not because it's cheap. A100/H100 80GB over A10/L4, so a real gap surfaces as scheduling/launch overhead instead of both engines converging on a memory-bound floor.

- [ ] **Stage 0 — Environment.** Install vLLM and SGLang against the same checkpoint, bf16 precision, chat template, and stop tokens. Serve identical prompts through each at greedy decoding (temperature 0).
  - Pass criteria:
    - *Not required:* bit-identical token IDs — bf16 plus two different attention kernels and sampling code paths guarantees drift under floating-point non-associativity. Record the observed match rate as a baseline, don't gate on it.
    - *Hard fail:* garbage on either engine (empty completion, repetition loop, mid-word truncation) across ~20 diverse prompts.
    - *Hard fail:* on ~10 deterministic prompts (arithmetic, single-fact QA), the two engines disagree on the *answer* — that's a chat-template/tokenizer/stop-token bug, not float drift, and it's not safe to profile through.

- [ ] **Stage 1 — Black-box benchmark.** Sweep request rate with each engine's own client (`vllm bench_serving`, SGLang's equivalent), open-loop Poisson arrivals, fixed input/output length, ≥60s or ~200+ requests per rate point.
  - Coarse pass, per engine: wide log-spaced grid (e.g. 1, 2, 4, 8, 16, 32, 64 req/s) run independently on each engine to find roughly where its throughput plateaus / p99 inflects. Don't assume the knees line up.
  - Fine pass, both engines together: shared finer-grained range bracketing the union of both knees (~0.5x the lower knee to 1.5x the higher one, ~8-10 points), run on both engines at the same rates.
  - Don't open a profiler until this chart shows something worth explaining.

- [ ] **Stage 2 — Nsight Systems pass.** Capture `nsys` traces for both engines under the load from Stage 1. Catalog kernel names, idle gaps, CPU<->GPU overlap — inventory, don't chase yet.

- [ ] **Stage 3 — Torch profiler pass.** Attach each engine's profiler hook (vLLM: `VLLM_TORCH_PROFILER_DIR`; SGLang's equivalent flag). Pull self-CUDA-time by op. Cross-check against the Stage 2 inventory.

- [ ] **Stage 4 — Isolate the cause.** Pick the single most promising difference from Stages 2-3 and design an experiment that changes only that variable. Forcing the same attention backend on both engines (FlashInfer or FlashAttention) is well-supported and a safe first target.
  - Check before committing to a mechanism: verify it's actually toggleable on both engines. Attention-backend selection is well-exposed on both sides; scheduler-level behavior like chunked-prefill is engine-specific and sometimes has no clean flag.

- [ ] **Stage 5 — Write-up.** The gap (Stage 1), the hypothesis (Stages 2-3), the isolating experiment (Stage 4), the result. Lead with numbers and profiler evidence, not narrative. Lives in `write-up.md`.

## Layout

- `bench/` — benchmark scripts and sweep configs (Stage 1)
- `traces/` — nsys and torch profiler captures (Stage 2-3)
- `results/` — parsed metrics, plots, comparison tables (Stage 1, 4)
- `write-up.md` — Stage 5 deliverable (stub until then)
