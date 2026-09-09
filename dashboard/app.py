import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Distributed Inference Profiling", layout="wide")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

RATES = [8, 12, 16, 20, 24, 32, 40, 48, 56, 64]

COLORS = {
    "vllm": "#7c3aed",
    "vllm_flashinfer": "#0891b2",
    "sglang": "#ea580c",
}


@st.cache_data
def load_sweep(subdir: str) -> pd.DataFrame:
    rows = []
    d = RESULTS / subdir
    for rate in RATES:
        f = d / f"rate_{rate}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        rows.append(
            {
                "rate": rate,
                "throughput": data["request_throughput"],
                "p99_ttft_s": data["p99_ttft_ms"] / 1000,
                "completed": data["completed"],
                "failed": data.get("failed") or 0,
            }
        )
    return pd.DataFrame(rows)


vllm = load_sweep("stage1_fine/vllm")
sglang = load_sweep("stage1_fine/sglang")
vllm_fi = load_sweep("stage4_flashinfer/vllm")

st.title("Distributed Inference Profiling")
st.caption(
    "vLLM vs SGLang serving Llama-3-8B (bf16) on a single A100 80GB — "
    "same model, same hardware, same workload. Where does the throughput gap come from?"
)

st.divider()

# ---------------------------------------------------------------------------
# Stage 1: the gap
# ---------------------------------------------------------------------------
st.header("1. The gap")
st.markdown(
    "Open-loop Poisson request-rate sweep, 512 input / 128 output tokens, temperature 0, "
    "zero failed requests at every rate point on both engines."
)

c1, c2, c3 = st.columns(3)
c1.metric("vLLM throughput ceiling", "~17.5 req/s", "knee at rate 16→20")
c2.metric("SGLang throughput ceiling", "~43.1 req/s", "knee at rate 40→48")
c3.metric("Gap", "2.45x", "same hardware, same workload")

col_thr, col_lat = st.columns(2)

with col_thr:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vllm["rate"], y=vllm["throughput"], name="vLLM (FlashAttention)",
                              mode="lines+markers", line=dict(color=COLORS["vllm"], width=3)))
    fig.add_trace(go.Scatter(x=sglang["rate"], y=sglang["throughput"], name="SGLang (FlashInfer)",
                              mode="lines+markers", line=dict(color=COLORS["sglang"], width=3)))
    fig.add_trace(go.Scatter(x=RATES, y=RATES, name="offered load (y=x)",
                              mode="lines", line=dict(color="gray", width=1, dash="dot")))
    fig.update_layout(title="Achieved throughput vs. offered request rate",
                       xaxis_title="request rate (req/s)", yaxis_title="achieved throughput (req/s)",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width="stretch")

with col_lat:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vllm["rate"], y=vllm["p99_ttft_s"], name="vLLM (FlashAttention)",
                              mode="lines+markers", line=dict(color=COLORS["vllm"], width=3)))
    fig.add_trace(go.Scatter(x=sglang["rate"], y=sglang["p99_ttft_s"], name="SGLang (FlashInfer)",
                              mode="lines+markers", line=dict(color=COLORS["sglang"], width=3)))
    fig.update_layout(title="p99 time-to-first-token vs. offered request rate",
                       xaxis_title="request rate (req/s)",
                       yaxis=dict(title="p99 TTFT (s)", type="log", dtick=1),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Stage 2-3: the hypothesis
# ---------------------------------------------------------------------------
st.header("2. The hypothesis — where is the GPU spending its time?")
st.markdown(
    "Nsight Systems (kernel inventory) and the PyTorch profiler (op-level self-CUDA-time), both captured "
    "at rate=24 — past vLLM's knee, comfortably before SGLang's. Two engines, two different attention kernels."
)

kernel_df = pd.DataFrame(
    [
        {"engine": "vLLM (FlashAttention split-KV)", "category": "GEMM", "pct": 68.7},
        {"engine": "vLLM (FlashAttention split-KV)", "category": "Attention", "pct": 26.6},
        {"engine": "vLLM (FlashAttention split-KV)", "category": "Other", "pct": 4.7},
        {"engine": "SGLang (FlashInfer)", "category": "GEMM", "pct": 83.6},
        {"engine": "SGLang (FlashInfer)", "category": "Attention", "pct": 9.6},
        {"engine": "SGLang (FlashInfer)", "category": "Other", "pct": 6.8},
    ]
)

col_bar, col_notes = st.columns([1, 1])

with col_bar:
    fig = go.Figure()
    for cat, color in [("GEMM", "#64748b"), ("Attention", "#dc2626"), ("Other", "#cbd5e1")]:
        sub = kernel_df[kernel_df["category"] == cat]
        fig.add_trace(go.Bar(x=sub["engine"], y=sub["pct"], name=cat, marker_color=color))
    fig.update_layout(barmode="stack", title="Self-CUDA-time by op category (torch profiler, full run @ rate=24)",
                       yaxis_title="% of total GPU kernel time",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width="stretch")

with col_notes:
    st.markdown(
        """
**nsys kernel inventory (bounded steady-state window, rate=24):**
- vLLM: GPU **~84% utilized**, two kernels (GEMM + FlashAttention split-KV) account for **94%** of GPU time
- SGLang: GPU mostly idle in the same narrow window — not a different kernel mix, just far less saturated at this offered load

**torch profiler op-level breakdown (full run, rate=24):**
- vLLM's attention kernel takes **26.6%** of its own GPU time vs. SGLang's **9.6%** — a ~2.8x larger share
- vLLM's dominant attention kernel runs **~11.5x slower per launch** (970μs vs. 84μs) than SGLang's
- vLLM's GEMM kernels also run ~2x slower per launch despite similar launch counts — consistent with backlog-driven batch inefficiency, not just the kernel choice

**Two live hypotheses going into Stage 4:**
1. FlashAttention's split-KV kernel is intrinsically less efficient than FlashInfer's for this access pattern
2. vLLM's backlog under load forces less efficient batch/kernel shapes — a scheduling effect, not a kernel one
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Stage 4: the isolating experiment
# ---------------------------------------------------------------------------
st.header("3. The isolating experiment")
st.markdown(
    "Change **one** variable — force vLLM onto the same FlashInfer attention backend SGLang already uses — "
    "and re-run the identical sweep. Everything else (model, hardware, workload, scheduler) stays fixed."
)

c1, c2, c3 = st.columns(3)
c1.metric("vLLM ceiling, FlashAttention", "~17.5 req/s")
c2.metric("vLLM ceiling, FlashInfer", "~20.9 req/s", "+19%")
c3.metric("Gap to SGLang", "2.45x → 2.05x", "-27% of the gap closed")

fig = go.Figure()
fig.add_trace(go.Scatter(x=vllm["rate"], y=vllm["throughput"], name="vLLM (FlashAttention)",
                          mode="lines+markers", line=dict(color=COLORS["vllm"], width=3)))
fig.add_trace(go.Scatter(x=vllm_fi["rate"], y=vllm_fi["throughput"], name="vLLM (FlashInfer)",
                          mode="lines+markers", line=dict(color=COLORS["vllm_flashinfer"], width=3)))
fig.add_trace(go.Scatter(x=sglang["rate"], y=sglang["throughput"], name="SGLang (FlashInfer)",
                          mode="lines+markers", line=dict(color=COLORS["sglang"], width=3)))
fig.add_trace(go.Scatter(x=RATES, y=RATES, name="offered load (y=x)",
                          mode="lines", line=dict(color="gray", width=1, dash="dot")))
fig.update_layout(title="Throughput vs. offered request rate — before and after the attention-backend swap",
                   xaxis_title="request rate (req/s)", yaxis_title="achieved throughput (req/s)",
                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Stage 5: the result
# ---------------------------------------------------------------------------
st.header("4. The result")
st.markdown(
    """
Switching vLLM's attention kernel alone closes **about a quarter** of the gap to SGLang, not all of it.
That's not an inconclusive result — it's evidence that **both hypotheses were partially right**:

- The attention-kernel choice (FlashAttention split-KV vs. FlashInfer) is a real, measurable inefficiency —
  worth ~19% of vLLM's throughput ceiling on its own.
- The remaining ~2.05x gap is *not* explained by the attention kernel, since that variable is now controlled
  for. It points at scheduling/backlog-driven batch inefficiency, or a structural difference between the
  engines (e.g. SGLang's RadixAttention prefix-cache-aware scheduler) — out of scope for this project's
  single-variable Stage 4, but the clear next thing to isolate.

Full methodology, every bug hit and fixed along the way, and the complete numbers are in the
[repo README](https://github.com/roma5087/inference-profiling) and `write-up.md`.
    """
)
