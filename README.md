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

## Pinned versions

| Package | Version | Notes |
|---|---|---|
| vLLM | 0.27.1 | A100 default attention backend is not FlashInfer, but `VLLM_ATTENTION_BACKEND=FLASHINFER` forces it (FlashInfer supports SM80/A100). |
| SGLang | 0.5.18 | Already defaults to FlashInfer on non-Hopper GPUs, including A100. |

This resolves Stage 4's open question: forcing both engines onto FlashInfer on A100 is confirmed toggleable, not just planned — SGLang needs no change, vLLM needs the env var above.

**Environment split:** vLLM and SGLang can't share one Python environment — `pip` hard-conflicts on `flashinfer-python` (vLLM 0.27.1 pins `==0.6.16.post3`, SGLang 0.5.18 pins `==0.6.17`). Each engine lives in its own venv (`~/venv-vllm`, `~/venv-sglang`). Downstream effect for Stage 4: after forcing both onto "FlashInfer," the underlying FlashInfer *library version* still differs by one patch release between engines — a real variable, not assumed away. Note it in the write-up; don't chase it as the cause unless the Stage 4 result looks inconsistent with the attention-kernel-selection story.

## Plan

- [ ] **Stage −1 — GPU selection.** Choose the card because of what it will show, not because it's cheap. A100/H100 80GB over A10/L4, so a real gap surfaces as scheduling/launch overhead instead of both engines converging on a memory-bound floor.
  - Default to **A100 80GB** over H100 — cheaper (~$1.6-2/hr on brokers like Brev), and an 8B model at bf16 won't saturate its bandwidth at the batch sizes Stage 1's sweep will hit. Switch to H100 only if Stage 1's coarse pass shows both engines' throughput curves *converging* to the same shape — that's the signal you're bandwidth-bound rather than scheduler-bound, and the fix is headroom, not more analysis.
  - Provider listing risk: some GPU broker listings (e.g. Brev/Hyperstack) are marked pre-release, **cannot be stopped or restarted**, and **delete all instance data irrecoverably** if the org runs out of credits. Check listing details before committing — prefer a stable listing at comparable price if one exists.
  - Disk storage is typically bundled and fixed at this GPU tier (e.g. 850GB SSD, included in the hourly rate) — not a separate sizing decision, and comfortably more than the checkpoint (~16GB) plus trace files need.

- [ ] **Stage 0 — Environment.** First command after SSH, before installing anything: verify `nsys` perf-counter access — some cloud GPU images restrict the counters Nsight Systems needs. On an instance with no stop/restart, a failure caught later means spinning up a fresh instance, not fixing this one in place.
  - Install vLLM 0.27.1 and SGLang 0.5.18 (see Pinned versions) against the same checkpoint, bf16 precision, chat template, and stop tokens. Serve identical prompts through each at greedy decoding (temperature 0).
  - Prompt sets: `bench/prompts_sanity.json` (20 diverse prompts) and `bench/prompts_deterministic.json` (10 arithmetic/single-fact prompts with `expected_answer`).
  - Pass criteria:
    - *Not required:* bit-identical token IDs — bf16 plus two different attention kernels and sampling code paths guarantees drift under floating-point non-associativity. Record the observed match rate as a baseline, don't gate on it.
    - *Hard fail:* garbage on either engine (empty completion, repetition loop, mid-word truncation) on any prompt in `prompts_sanity.json`.
    - *Hard fail:* on any prompt in `prompts_deterministic.json`, the two engines disagree on the *answer* — that's a chat-template/tokenizer/stop-token bug, not float drift, and it's not safe to profile through.

- [ ] **Stage 1 — Black-box benchmark.** Sweep request rate with each engine's own client (`vllm bench_serving`, SGLang's equivalent), open-loop Poisson arrivals, fixed input/output length, ≥60s or ~200+ requests per rate point.
  - Coarse pass, per engine: wide log-spaced grid (e.g. 1, 2, 4, 8, 16, 32, 64 req/s) run independently on each engine to find roughly where its throughput plateaus / p99 inflects. Don't assume the knees line up.
  - Fine pass, both engines together: shared finer-grained range bracketing the union of both knees (~0.5x the lower knee to 1.5x the higher one, ~8-10 points), run on both engines at the same rates.
  - Don't open a profiler until this chart shows something worth explaining.

- [ ] **Stage 2 — Nsight Systems pass.** Capture `nsys` traces for both engines under the load from Stage 1. Catalog kernel names, idle gaps, CPU<->GPU overlap — inventory, don't chase yet.

- [ ] **Stage 3 — Torch profiler pass.** Attach each engine's profiler hook (vLLM: `VLLM_TORCH_PROFILER_DIR`; SGLang's equivalent flag). Pull self-CUDA-time by op. Cross-check against the Stage 2 inventory.

- [ ] **Stage 4 — Isolate the cause.** Pick the single most promising difference from Stages 2-3 and design an experiment that changes only that variable. First target: force both engines onto FlashInfer (SGLang: no change needed; vLLM: `VLLM_ATTENTION_BACKEND=FLASHINFER`) — confirmed toggleable on A100, see Pinned versions.
  - If the attention-backend experiment doesn't explain the gap, the next candidate is scheduler-level behavior (e.g. chunked-prefill) — that's engine-specific and sometimes has no clean flag, so verify it's actually toggleable on both engines before committing to it.

- [ ] **Stage 5 — Write-up & dashboard.** The gap (Stage 1), the hypothesis (Stages 2-3), the isolating experiment (Stage 4), the result. Lead with numbers and profiler evidence, not narrative. Prose lives in `write-up.md`.
  - Dashboard: `dashboard/app.py`, a Streamlit app reading directly from `results/` — interactive throughput/latency curves (Stage 1), the annotated nsys finding and the Stage 4 experiment result. Deploy to Streamlit Community Cloud for a public link. Built once Stage 1 produces real data — not before, and never with placeholder numbers standing in for results.

## Layout

- `bench/` — benchmark scripts and sweep configs (Stage 1)
- `traces/` — nsys and torch profiler captures (Stage 2-3)
- `results/` — parsed metrics, plots, comparison tables (Stage 1, 4)
- `dashboard/` — Streamlit app for Stage 5 (empty until real results exist)
- `write-up.md` — Stage 5 deliverable (stub until then)
