#!/usr/bin/env python3
"""Parse a PyTorch profiler Chrome-trace (.trace.json.gz) and print self-CUDA-time
by kernel name, for cross-checking against the Stage 2 nsys kernel inventory.

Tolerant of truncated gzip streams (missing trailer): decompresses whatever
prefix is recoverable, then incrementally JSON-decodes individual events out
of the traceEvents array, discarding only the final partial event."""
import argparse
import gzip
import json
import sys
import zlib
from collections import defaultdict


def load_events_tolerant(path):
    raw = open(path, "rb").read()
    d = zlib.decompressobj(zlib.MAX_WBITS | 16)
    try:
        text = d.decompress(raw).decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  partial decompress raised {e!r}, using what we have", file=sys.stderr)
        text = getattr(d, "_unconsumed_tail", b"")
    if not text:
        # fall back: manual chunked decompress tolerating a bad trailer
        text_parts = []
        d2 = zlib.decompressobj(zlib.MAX_WBITS | 16)
        chunk_size = 1 << 20
        for i in range(0, len(raw), chunk_size):
            try:
                text_parts.append(d2.decompress(raw[i:i + chunk_size]))
            except zlib.error:
                break
        text = b"".join(text_parts).decode("utf-8", errors="ignore")

    print(f"  recovered {len(text)/1e6:.1f}MB of decompressed JSON text", file=sys.stderr)

    marker = '"traceEvents"'
    idx = text.find(marker)
    if idx == -1:
        raise RuntimeError("no traceEvents key found in recovered text")
    start = text.find("[", idx)
    pos = start + 1
    decoder = json.JSONDecoder()
    events = []
    n = len(text)
    while pos < n:
        while pos < n and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= n or text[pos] == "]":
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        events.append(obj)
        pos = end
    print(f"  salvaged {len(events)} complete trace events (stream was truncated)", file=sys.stderr)
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_path")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    print(f"Loading {args.trace_path} ...", file=sys.stderr)
    try:
        with gzip.open(args.trace_path, "rt") as f:
            data = json.load(f)
        events = data["traceEvents"] if isinstance(data, dict) else data
    except (EOFError, OSError, json.JSONDecodeError) as e:
        print(f"  clean load failed ({e!r}); salvaging truncated gzip stream", file=sys.stderr)
        events = load_events_tolerant(args.trace_path)

    print(f"Loaded {len(events)} trace events", file=sys.stderr)

    kernel_time = defaultdict(lambda: [0, 0])  # name -> [total_dur_us, count]
    cat_totals = defaultdict(int)

    for e in events:
        cat = e.get("cat")
        dur = e.get("dur")
        if dur is None:
            continue
        cat_totals[cat] += dur
        if cat == "kernel":
            name = e.get("name", "?")
            kernel_time[name][0] += dur
            kernel_time[name][1] += 1

    print("\n=== Event category totals (us) ===")
    for cat, total in sorted(cat_totals.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat:>20}: {total/1000:>12.1f} ms")

    total_kernel_us = sum(v[0] for v in kernel_time.values())
    print(f"\n=== Self-CUDA-time by kernel name (top {args.top}) ===")
    print(f"Total GPU kernel time: {total_kernel_us/1e6:.3f}s across {sum(v[1] for v in kernel_time.values())} launches, {len(kernel_time)} distinct kernels\n")
    rows = sorted(kernel_time.items(), key=lambda x: -x[1][0])[:args.top]
    for name, (dur_us, count) in rows:
        pct = 100 * dur_us / total_kernel_us if total_kernel_us else 0
        avg_us = dur_us / count if count else 0
        short_name = name if len(name) <= 80 else name[:77] + "..."
        print(f"  {pct:5.1f}%  {dur_us/1000:>10.1f}ms  {count:>7}  avg={avg_us:>8.2f}us  {short_name}")


if __name__ == "__main__":
    main()
