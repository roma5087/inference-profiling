# Distributed Inference Profiling: vLLM vs SGLang on a Single A100

Serve Llama-3-8B (bf16) on vLLM and SGLang, on identical hardware and workload, find a real
throughput gap, and prove — with Nsight Systems and the PyTorch profiler, not just a benchmark
number — what causes it.

Full methodology, every version pin, and the raw data live in the [repo](https://github.com/roma5087/inference-profiling)
and its [README](README.md). Interactive charts: `dashboard/app.py`.

## The gap

Open-loop Poisson request-rate sweep (`vllm bench serve` / `sglang.bench_serving`), 512 input /
128 output tokens, temperature 0, identical rate grid on both engines, zero failed requests at
every point on both engines — both engines fail the same way under overload (queue and let
latency grow, never reject a request outright).

| rate | vLLM throughput | vLLM p99 TTFT | SGLang throughput | SGLang p99 TTFT |
|---|---|---|---|---|
| 8 | 7.79 | 0.18s | 7.45 | 0.11s |
| 16 | 15.35 | 0.50s | 14.91 | 0.13s |
| 20 | 16.23 | **11.07s** | 19.21 | 0.16s |
| 24 | 16.63 | 23.56s | 23.52 | 0.18s |
| 40 | 17.46 | 74.41s | 35.13 | 0.52s |
| 48 | 17.52 | 101.02s | 39.33 | **4.10s** |
| 64 | 17.59 | 153.31s | 43.06 | 18.27s |

**vLLM's knee is between rate 16 and 20; SGLang's is between rate 40 and 48 — a ~2.4x higher
throughput ceiling**, reproduced across two independent version pairs of each engine (vLLM
0.27.1→0.28.0, SGLang 0.5.18→0.5.19). That's the gap the rest of this project explains.

## The hypothesis

Two profiling passes at rate=24 (past vLLM's knee, comfortably before SGLang's), same 480-prompt
window:

**Nsight Systems, kernel inventory (bounded steady-state window).** vLLM's GPU is ~84% utilized —
two kernels, the main GEMM and FlashAttention's split-KV kernel, account for 94% of GPU time.
SGLang's GPU is mostly idle in the same narrow window, ~20x less busy — not a different kernel
mix, just far less saturated at this offered load.

**PyTorch profiler, op-level self-CUDA-time (full run).** Aggregated over the whole 480-prompt
run, both engines are GEMM-dominated (vLLM 68.7%, SGLang 83.6%) — confirming the nsys inventory
wasn't misleading, just narrower in scope. The new signal: **vLLM's attention kernel takes 26.6%
of its own GPU time vs. SGLang's 9.6%** (a ~2.8x larger share), and **vLLM's dominant attention
kernel runs ~11.5x slower per launch** (970μs vs. 84μs) than SGLang's. vLLM's GEMM kernels also
run ~2x slower per launch despite similar launch counts — a second, independent signal pointing
at backlog-driven batch inefficiency, not just the attention kernel choice.

Two hypotheses, not mutually exclusive:
1. FlashAttention's split-KV kernel is intrinsically less efficient than FlashInfer's for this access pattern.
2. vLLM's backlog under load forces less efficient batch/kernel shapes — a scheduling effect, not a kernel one.

## The isolating experiment

Change **one** variable: force vLLM onto the same FlashInfer attention backend SGLang already
uses (`vllm serve --attention-backend FLASHINFER` — the toggle mechanism actually changed between
vLLM 0.27.1 and 0.28.0, from an env var to a CLI flag; the old env var silently no-ops on 0.28.0,
caught by checking the server log's backend-selection line before trusting the run). Re-run the
identical fine-sweep methodology. Everything else — model, hardware, workload, scheduler — stays
fixed. Verified no correctness regression first: every rate point generated exactly 128.0 avg
output tokens/request, zero failures.

| rate | vLLM+FlashInfer throughput | vLLM+FlashInfer p99 TTFT | (original vLLM+FlashAttention) |
|---|---|---|---|
| 8 | 7.78 | 0.17s | 7.79 / 0.18s |
| 16 | 15.52 | 0.30s | 15.35 / 0.50s |
| 20 | 18.76 | 2.21s | 16.23 / 11.07s |
| 24 | 20.00 | 8.95s | 16.63 / 23.56s |
| 40 | 20.81 | 52.52s | 17.46 / 74.41s |
| 64 | 20.95 | 118.66s | 17.59 / 153.31s |

## The result

Switching vLLM's attention kernel alone raises its throughput ceiling from **~17.5 to ~20.9
req/s — a real +19%** — and softens the knee (rate 20's p99 TTFT drops from 11.07s to 2.21s).
SGLang's ceiling (43.06 req/s) is untouched by this experiment. **The vLLM/SGLang gap moves from
2.45x to 2.05x — about 27% of the gap closes, not all of it.**

That's not an inconclusive result — it's evidence both hypotheses were partially right. The
attention-kernel choice explains a real, measurable slice of the gap (hypothesis 1). The
remaining ~2.05x gap, now with the attention kernel controlled for, points at scheduling/backlog
effects or a structural difference between the engines — e.g. SGLang's RadixAttention
prefix-cache-aware scheduler — that a single-variable experiment isn't designed to isolate
(hypothesis 2). That's the natural next thing to profile — see below.

## The follow-up: is it scheduling, or is it prefix caching?

Hypothesis 2 named two candidate mechanisms: SGLang's RadixAttention prefix cache, and
backlog-driven batch inefficiency. Prefix caching turned out to be untestable by construction —
this benchmark's synthetic workload uses fully random prompt content with a fresh seed per rate
point, specifically to prevent caching effects from leaking into the throughput numbers. There's
no repeated content for RadixAttention to exploit here, so it's not a live explanation for this
gap.

Polling each engine's live request-scheduling metrics (`/metrics`) during the same rate=24 load
found the real signal: **vLLM keeps ~2.8-3x more requests concurrently in-flight than SGLang at
the identical offered load** (max observed: 256 vs. 84 — vLLM was running right up against its
own default `max_num_seqs=256` admission ceiling). That gives the GEMM-latency finding above a
mechanism: larger concurrent batches produce the larger, less efficient GEMM shapes Stage 3
measured.

The natural next experiment: cap vLLM's admission to SGLang's observed ceiling
(`--max-num-seqs 84`) and re-run the full sweep. **Result: a clean negative, not a fix.**
Throughput ceiling *dropped* ~21% (20.8 → 16.4 req/s) and P99 latency got *worse* at every rate
≥16, not better.

**Batch size is a symptom of SGLang's per-iteration scheduling efficiency, not its cause.**
SGLang runs fewer concurrent requests because it clears each one fast enough that fewer stay
in-flight — it isn't throttling admission to get there. Forcing vLLM into the same small batch
via a hard cap just creates admission-side backpressure without fixing whatever makes each vLLM
decode iteration slower per unit of batched work. The remaining ~2.05x gap is still open; the
real lever is likely something about how each engine interleaves prefill and decode within a
single batching step — a harder thing to isolate with one CLI flag. Full detail, including the
environment work and a real vLLM 0.28.0 KV-cache-sizing bug found along the way (a cold kernel-
autotune cache can silently under-provision KV cache ~19x on first boot), in the
[README's Stage 6](README.md#plan).

## Appendix: what it actually takes to run both engines

Not the headline finding, but a real one: getting vLLM 0.27.1/0.28.0 and SGLang 0.5.18/0.5.19
booting on a bare pip venv on a fresh CUDA 13.2 / A100 cloud image required seven separate
environment fixes — a real import-time bug in a pinned `flashinfer-python` release, a missing C
compiler, no system CUDA toolkit, a CTK-compatibility check that blocks FlashInfer's JIT-compiled
kernels outright on newer toolkits, and a missing linker symlink. None of it is specific to this
project's methodology; all of it is documented, with the exact fix, in the
[README's Pinned versions section](README.md#pinned-versions) — worth knowing before trying to
reproduce any of this cold in September 2026.
