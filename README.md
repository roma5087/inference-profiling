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
| vLLM | 0.28.0 (upgraded 2026-09-08 from 0.27.1) | A100 default attention backend is not FlashInfer, but `VLLM_ATTENTION_BACKEND=FLASHINFER` forces it (FlashInfer supports SM80/A100). Still pins `flashinfer-python==0.6.16.post3` (unchanged) — existing patches applied without modification. |
| SGLang | 0.5.19 (upgraded 2026-09-08 from 0.5.18) | Already defaults to FlashInfer on non-Hopper GPUs, including A100. Its `flashinfer-python` pin moved 0.6.17→0.6.18 as part of this bump. |
| Nsight Systems | 2026.1.3 (upgraded 2026-09-08 from a stale apt default of 2021.3, which predates this driver/CUDA generation) | Installed via NVIDIA's own CUDA apt repo (`cuda-nsight-systems-13-3`), not Ubuntu's default repo. |

**Patch fragility across version bumps** — worth knowing before any future upgrade: SGLang's `flashinfer-python` version bump (0.6.17→0.6.18) triggered a fresh package install, which **silently reverted the CTK-compatibility-check patch** (fix #5 below) since that patch lives inside the flashinfer package's own files, not in a venv-level location a version bump would leave alone. The `array.array[int]` fix (#1) happened to still be unnecessary in 0.6.18 (upstream already carries it, same as 0.6.17), and the `lib64` symlink (#6) survived because it's outside any Python package's file tree. **Lesson: after any `pip install --upgrade` touching `flashinfer-python` (directly or transitively), re-verify fixes #1 and #5 specifically — they're the two that live inside package files pip will overwrite.**

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

- [x] **Stage 1 — Black-box benchmark.** Done 2026-09-08, corrected after independent review caught two measurement bugs in the first pass.

  **Two bugs found and fixed:**
  1. **vLLM's "failed" requests in the first pass were a client-side artifact, not a server behavior.** Every failure's error message was `OSError: [Errno 24] Too many open files` — `vllm bench serve`'s own aiohttp client ran out of file descriptors (hit the default 1024 `ulimit -n`) once backlog pushed concurrent in-flight requests past that ceiling. It never reached the server. Fixed: `ulimit -n 65536` before invoking either benchmark client (see script header comments).
  2. **`--seed 0` was reused unchanged at every rate point** against a server never restarted between points — confirmed via diff that 90% of one rate's completions were byte-identical to another's, i.e. a caching effect was leaking into throughput numbers. Fixed: `--seed "$RATE"` in both sweep scripts.

  **Memory/batching parity** (checked so a memory-budget asymmetry can't be an unstated confound): vLLM's KV cache (470,912 tokens, `gpu_memory_utilization=0.92`, `enable_chunked_prefill=True`) is *larger* than SGLang's (412,646 tokens, `mem_fraction_static=0.83`, `chunked_prefill_size=8192`) — not a memory asymmetry favoring either side.

  - **Coarse pass** (1,2,4,8,16,32,64 req/s, independent per engine, first-pass numbers on vLLM 0.27.1/SGLang 0.5.18, superseded below) — `results/stage1_coarse/{vllm,sglang}/`.
  - **Fine pass, corrected and re-validated on upgraded versions** (8,12,16,20,24,32,40,48,56,64 req/s, same grid both engines, `ulimit -n 65536`, per-rate seeds) — `results/stage1_fine/{vllm,sglang}/`. Run twice: once corrected on vLLM 0.27.1/SGLang 0.5.18, once more after upgrading to vLLM 0.28.0/SGLang 0.5.19 (current pinned versions, see Pinned versions table) to confirm the finding isn't tied to a specific release:

    | rate | vLLM 0.28.0 throughput | vLLM p99 TTFT | SGLang 0.5.19 throughput | SGLang p99 TTFT |
    |---|---|---|---|---|
    | 8 | 7.79 | 0.18s | 7.45 | 0.11s |
    | 12 | 11.68 | 0.23s | 11.43 | 0.11s |
    | 16 | 15.35 | 0.50s | 14.91 | 0.13s |
    | 20 | 16.23 | **11.07s** | 19.21 | 0.16s |
    | 24 | 16.63 | 23.56s | 23.52 | 0.18s |
    | 32 | 17.26 | 48.12s | 30.21 | 0.26s |
    | 40 | 17.46 | 74.41s | 35.13 | 0.52s |
    | 48 | 17.52 | 101.02s | 39.33 | **4.10s** |
    | 56 | 17.52 | 126.09s | 41.58 | 10.43s |
    | 64 | 17.59 | 153.31s | 43.06 | 18.27s |

    **Zero failed requests / zero errors for both engines at every rate point, on both version pairs.** The version upgrade (0.27.1→0.28.0, 0.5.18→0.5.19) changed nothing meaningful — every number above is within noise of the pre-upgrade corrected run. **Both engines fail the same way** (queue and let latency grow, never reject a request outright) — there is no "hard rejection vs. graceful degradation" split. What's real, and now confirmed stable across two version pairs: **vLLM's knee is between rate 16 and 20** (p99 TTFT ~0.5s→~11-12s), **SGLang's knee is between rate 40 and 48** (p99 TTFT ~0.5s→~4.1-4.4s) — a **~2.4x higher throughput ceiling**.

    **This is the gap Stages 2-4 need to explain**: a ~2.4x throughput-ceiling difference between two engines with the same overload behavior (queue, don't reject), on identical hardware, workload, and precision — reproduced across two independent version pairs of each engine.
  - Don't open a profiler until this chart shows something worth explaining. *(It does.)*

- [x] **Stage 2 — Nsight Systems pass.** Done 2026-09-08. Captured both engines at **rate=24** (past vLLM's knee — 16.35 req/s achieved, p99 TTFT 6.4s; comfortably below SGLang's — 24 req/s achieved, p99 TTFT 237ms), same 480-prompt window (512in/128out, seed=999), via `nsys launch`/`start`/`stop` so the ~90s startup/compile phase isn't in the trace — only steady-state serving is. `traces/{vllm,sglang}_rate24.nsys-rep` (Git LFS).

  **Capture method:** `nsys launch --trace=cuda,nvtx,osrt -- <server launch command>`, wait for healthy, `nsys start -o <path>`, run the benchmark client for the load window, `nsys stop`. Analyzed via `nsys stats --report cuda_gpu_kern_sum` (per-kernel time/count) and `cuda_gpu_sum` (aggregate).

  **Inventory findings:**
  - **vLLM: GPU is ~84% utilized** (24.70s busy / 29.36s window), 84,594 kernel launches, avg **292μs/launch**. Two kernels dominate: `ampere_bf16_s16816gemm_bf16_256x128_...` (67.3%, the main GEMM) and `flash::flash_fwd_splitkv_kernel<...>` (26.5%, FlashAttention's split-KV variant — consistent with the FLASH_ATTN backend documented above). Together, 94% of GPU time in two kernel types.
  - **SGLang: GPU is barely busy** — only **1.22s of busy time** in a ~20-26s window (rough estimate from the 480-prompt/24req-s target plus tail latency; not pinned to the second, doesn't change the magnitude here), despite **130,588 kernel launches** (more launches than vLLM, not fewer) at avg **9.3μs/launch** — kernels ~31x smaller on average. No single kernel dominates: the top one (`flashinfer::BatchPrefillWithRaggedKVCacheKernel`) is only 21.1% of SGLang's own GPU time, spread across a long tail of `flashinfer::*` prefill/merge-state kernels and small elementwise ops.
  - **The magnitude is the headline, not just the direction**: at the identical offered load, vLLM's GPU is essentially pegged while SGLang's is mostly idle waiting for the next request. That's consistent with (not yet proof of) vLLM being past capacity while SGLang has substantial headroom — but a ~20x gap in total GPU-seconds to serve a similar volume of requests is large enough that "different attention kernel" alone is a real candidate to investigate directly in Stage 4, not just a footnote.
  - **Don't chase yet** (per plan) — this is inventory. Two live hypotheses for Stage 3/4 to actually test: (a) FlashAttention's split-KV kernel is genuinely less efficient than FlashInfer's paged/ragged kernels for this access pattern — directly testable via the already-confirmed-toggleable FlashInfer-for-both experiment; (b) vLLM's backlog under load forces batch/kernel shapes that are less efficient than SGLang's better-paced batches — a scheduling-side explanation, not a kernel-side one. Stage 3's op-level torch profiler pass should help distinguish which.

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
