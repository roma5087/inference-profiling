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
| Engine execution | sequential, never concurrent | Both engines share the single GPU's SMs and memory bandwidth. Running them at the same time would make each engine's measured throughput depend on how much of the card the *other* engine happened to be using at that instant — an uncontrolled confound neither Stage 1's numbers nor Stage 2's traces could untangle. One engine fully owns the GPU for its entire sweep before the other starts. |

Model checkpoints/weights are not committed here (re-downloadable, not the artifact) — see `.gitignore`. Everything else — scripts, configs, traces, parsed results, the write-up — lives in this repo.

## Pinned versions

| Package | Version | Notes |
|---|---|---|
| vLLM | 0.27.1 | A100 default attention backend is not FlashInfer, but `VLLM_ATTENTION_BACKEND=FLASHINFER` forces it (FlashInfer supports SM80/A100). |
| SGLang | 0.5.18 | Already defaults to FlashInfer on non-Hopper GPUs, including A100. |

This resolves Stage 4's open question: forcing both engines onto FlashInfer on A100 is confirmed toggleable, not just planned — SGLang needs no change, vLLM needs the env var above.

**Environment fixes needed to actually boot vLLM on this GCP image** (Ubuntu 22.04, driver 595.71.05 / CUDA 13.2, pip-only CUDA toolkit via `nvidia-cuda-nvcc`, no system `/usr/local/cuda`):
1. `flashinfer_python==0.6.16.post3` has a real import-time bug: `flashinfer/comm/fd_exchange.py` type-hints a function return with `array.array[int]`, which isn't runtime-subscriptable and throws `TypeError: 'type' object is not subscriptable` on import. This import is unconditional in vLLM's startup kernel-warmup path (triggered via an unrelated MiniMax-M3 warmup import), so it can't be dodged with `--enforce-eager` or backend flags. Fix: patch the file to add `from __future__ import annotations` as its first line, which makes all annotations lazy. Not our bug — a real upstream compatibility issue in that flashinfer patch release.
2. Triton's CUDA driver JIT needs a C compiler and Python headers neither present by default: `sudo apt-get install -y build-essential python3.10-dev`.
3. `nvcc` isn't on `PATH` and there's no `/usr/local/cuda` — it ships inside the venv via the `nvidia-cuda-nvcc` pip package. Set `CUDA_HOME=<venv>/lib/python3.10/site-packages/nvidia/cu13` and add `$CUDA_HOME/bin` to `PATH` before launching.
4. Even with `CUDA_HOME` set, FlashInfer's own JIT-compiled top-k/top-p sampling kernel fails to build against this environment's CUDA headers (`"CUDA compiler and CUDA toolkit headers are incompatible"` from its bundled `cccl`/`libcudacxx`). First worked around with `VLLM_USE_FLASHINFER_SAMPLER=0` — **superseded once fix #5/#6 below were applied to vLLM's venv too; not needed once the real bug is patched.**
5. SGLang hits the same CTK-compatibility `#error`, but here it's in the actual attention kernel (`batch_prefill_with_kv_cache...`), not an optional sampler — no env var dodges it. `cuda_toolkit.h` has a documented escape hatch for exactly this ("users might want to use a newer CTK than the compiler ships"): define `CCCL_DISABLE_CTK_COMPATIBILITY_CHECK` before its `#ifndef` guard, in both venvs' copies of the header.
6. After that, linking still fails: `-lcudart` isn't found. The pip `nvidia-cuda-nvcc` package ships `libcudart.so.13` under `nvidia/cu13/lib` (not `lib64`, which is what the JIT build's `-L` flag points at), with no unversioned `.so` symlink. Fix: `ln -s lib lib64` and `ln -s libcudart.so.13 libcudart.so` inside `nvidia/cu13/`, in both venvs.
7. SGLang's copy of `flashinfer/comm/fd_exchange.py` (flashinfer 0.6.17) already had the `array.array[int]` bug (#1) fixed upstream — don't reapply that patch there, only vLLM's flashinfer 0.6.16.post3 needs it.

None of these are specific to this project's methodology — they're just what it takes to get vLLM 0.27.1 and SGLang 0.5.18 running on a bare pip venv on a fresh cloud GPU image in September 2026. Worth a line in the write-up as its own small finding.

**Current recommended launch (both engines, with all real fixes applied — no workaround flags needed):**
```
source ~/venv-vllm/bin/activate
export CUDA_HOME=~/venv-vllm/lib/python3.10/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8001
```
(same `CUDA_HOME`/`PATH` pattern for `~/venv-sglang` + `python3 -m sglang.launch_server --model-path ... --port 8002`.) Confirmed: full torch.compile, both CUDA graph modes (PIECEWISE + FULL), and FlashInfer's own sampler all working — this is the fully representative config both engines need for Stage 1's benchmark to be a fair comparison, not an eager-mode-vs-compiled-mode artifact.

**Stage 1 methodology note:** `vllm bench serve` no longer defaults to greedy decoding — actual sampling temperature is server/model-dependent unless set explicitly. Both `sweep_coarse.sh` invocations pass `--temperature 0` so neither engine's default silently becomes an uncontrolled variable in the comparison.

**Environment split:** vLLM and SGLang can't share one Python environment — `pip` hard-conflicts on `flashinfer-python` (vLLM 0.27.1 pins `==0.6.16.post3`, SGLang 0.5.18 pins `==0.6.17`). Each engine lives in its own venv (`~/venv-vllm`, `~/venv-sglang`). Downstream effect for Stage 4: after forcing both onto "FlashInfer," the underlying FlashInfer *library version* still differs by one patch release between engines — a real variable, not assumed away. Note it in the write-up; don't chase it as the cause unless the Stage 4 result looks inconsistent with the attention-kernel-selection story.

## Plan

- [x] **Stage −1 — GPU selection.** Done — GCP-backed A100 80GB (`a2-ultragpu-1g:nvidia-a100-80gb:1`), pausable, driver 595.71.05 / CUDA 13.2. Choose the card because of what it will show, not because it's cheap. A100/H100 80GB over A10/L4, so a real gap surfaces as scheduling/launch overhead instead of both engines converging on a memory-bound floor.
  - Default to **A100 80GB** over H100 — cheaper (~$1.6-2/hr on brokers like Brev), and an 8B model at bf16 won't saturate its bandwidth at the batch sizes Stage 1's sweep will hit. Switch to H100 only if Stage 1's coarse pass shows both engines' throughput curves *converging* to the same shape — that's the signal you're bandwidth-bound rather than scheduler-bound, and the fix is headroom, not more analysis.
  - Provider listing risk: some GPU broker listings (e.g. Brev/Hyperstack) are marked pre-release, **cannot be stopped or restarted**, and **delete all instance data irrecoverably** if the org runs out of credits. Check listing details before committing — prefer a stable listing at comparable price if one exists.
  - Disk storage is typically bundled and fixed at this GPU tier (e.g. 850GB SSD, included in the hourly rate) — not a separate sizing decision, and comfortably more than the checkpoint (~16GB) plus trace files need.

- [x] **Stage 0 — Environment.** Done 2026-09-07. Both engines installed (separate venvs), driver/CUDA confirmed (595.71.05 / CUDA 13.2), seven environment fixes applied (see Pinned versions), both servers boot and answer requests.
  - Prompt sets: `bench/prompts_sanity.json` (20 diverse prompts) and `bench/prompts_deterministic.json` (10 arithmetic/single-fact prompts with `expected_answer`).
  - Pass criteria — **all passed**:
    - Hard fail (garbage on sanity prompts): **PASS**, neither engine produced garbage across all 20.
    - Hard fail (deterministic-answer agreement): **PASS**, both engines got all 10 right.
    - Baseline (not gated): **8/20** sanity completions were exact string matches between engines — real observed drift under bf16 + different kernels, not required to be higher.
  - Results: `results/vllm_stage0_completions.json`, `results/sglang_stage0_completions.json`. Reproduce with `bench/gen_completions.py --engine {vllm,sglang} --port <port>` against a running server, then `bench/compare_stage0.py`.

- [~] **Stage 1 — Black-box benchmark.** First pass done 2026-09-08; **correction in progress** after independent review caught two measurement bugs. Fixed length: 512 input / 128 output tokens (`--dataset-name random --random-range-ratio 0`), open-loop Poisson, `--temperature 0` on both clients.

  **Two bugs found in the numbers below, being re-run:**
  1. **vLLM's "failed" requests (rate ≥32 in the fine-pass table) are a client-side artifact, not a server behavior.** Every failure's error message is `OSError: [Errno 24] Too many open files` — `vllm bench serve`'s own aiohttp client ran out of file descriptors (hit the default 1024 `ulimit -n`) once backlog pushed concurrent in-flight requests past that ceiling. It never reached the server. SGLang's client hit 2127 concurrent requests with zero errors — the two clients weren't running under comparable limits, so "vLLM hard-fails, SGLang never fails" is not yet a trustworthy comparison. The TTFT-based knee (rate 16→20, `failed=0` at both points) is unaffected and stands.
  2. **`--seed 0` was reused unchanged at every rate point**, against a server never restarted between points. Diffing outputs confirmed it: 90% of `rate_12`'s completions are byte-identical to `rate_8`'s. Since both engines have prefix/radix caching on by default, later (larger `num_prompts`) rate points were partially replaying earlier points' exact prompt content — a caching effect that could leak into throughput numbers as if it were a scheduling effect. Fixed in `sweep_coarse.sh`/`sweep_fine.sh`: `--seed "$RATE"` instead of a fixed value.

  Re-running the fine pass with `ulimit -n 65536` and per-rate seeds before trusting the table below.

  - **Coarse pass** (1,2,4,8,16,32,64 req/s, independent per engine) — `results/stage1_coarse/{vllm,sglang}/`. First signal: vLLM ceiling ~17 req/s with hard failures past it; SGLang scales further and never hard-fails.
  - **Fine pass** (8,12,16,20,24,32,40,48,56,64 req/s, same grid both engines) — `results/stage1_fine/{vllm,sglang}/`. Resolved both knees precisely:

    | rate | vLLM throughput | vLLM failed | vLLM p99 TTFT | SGLang throughput | SGLang failed | SGLang p99 TTFT |
    |---|---|---|---|---|---|---|
    | 8 | 7.81 | 0 | 0.18s | 8.01 | 0 | 0.11s |
    | 12 | 11.60 | 0 | 0.24s | 11.08 | 0 | 0.12s |
    | 16 | 15.42 | 0 | 0.27s | 14.41 | 0 | 0.13s |
    | 20 | 15.95 | 0 | **12.37s** | 18.37 | 0 | 0.13s |
    | 24 | 16.46 | 0 | 24.32s | 22.21 | 0 | 0.14s |
    | 32 | 17.01 | 101 | 44.42s | 29.93 | 0 | 0.23s |
    | 40 | 17.14 | 570 | 44.44s | 35.37 | 0 | 0.51s |
    | 48 | 17.28 | 1039 | 44.17s | 39.78 | 0 | **4.73s** |
    | 56 | 17.34 | 1516 | 44.32s | 41.90 | 0 | 11.55s |
    | 64 | 17.33 | 1994 | 44.40s | 43.40 | 0 | 17.38s |

    **vLLM's real knee is between rate 16 and 20** — p99 TTFT jumps from 0.27s to 12.4s in one step, then throughput hard-plateaus at ~17.3 req/s for every rate above that, with a growing count of outright request failures. **SGLang's knee is between rate 40 and 48** — p99 TTFT jumps from 0.51s to 4.7s — roughly **2.4x higher** than vLLM's, and throughput keeps climbing (not a hard plateau) even past its own knee, with zero request failures across the entire grid.

    **This is the gap Stages 2-4 need to explain**, and it's not subtle: two qualitatively different failure modes (hard request rejection vs. graceful latency degradation) at meaningfully different throughput ceilings, on identical hardware, identical workload, identical precision.
  - Don't open a profiler until this chart shows something worth explaining. *(It does.)*

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
