#!/usr/bin/env python3
"""Stage 6 inventory: poll an engine's /metrics endpoint at fixed intervals
during a load window, logging batch-size-over-time (running/waiting request
counts) so Stage 4's remaining ~2.05x gap can be checked against an actual
scheduling/batching signal instead of inferred from Stage 3's GEMM latency.

Usage: poll_metrics.py --engine {vllm,sglang} --url <metrics-url> --outfile <path> [--interval 0.2]
Runs until killed (SIGTERM/SIGINT) or the parent orchestrating script stops it.
"""
import argparse
import csv
import re
import signal
import sys
import time

import urllib.request

VLLM_PATTERNS = {
    "running": re.compile(r'^vllm:num_requests_running\{[^}]*\}\s+([\d.]+)', re.M),
    "waiting": re.compile(r'^vllm:num_requests_waiting\{[^}]*\}\s+([\d.]+)', re.M),
}
SGLANG_PATTERNS = {
    "running": re.compile(r'^sglang:num_running_reqs\{[^}]*\}\s+([\d.]+)', re.M),
    "waiting": re.compile(r'^sglang:num_queue_reqs\{[^}]*\}\s+([\d.]+)', re.M),
}

_stop = False


def _handle_stop(signum, frame):
    global _stop
    _stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--interval", type=float, default=0.2)
    args = ap.parse_args()

    patterns = VLLM_PATTERNS if args.engine == "vllm" else SGLANG_PATTERNS

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    with open(args.outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "running", "waiting"])
        f.flush()
        t0 = time.monotonic()
        n = 0
        while not _stop:
            loop_start = time.monotonic()
            try:
                with urllib.request.urlopen(args.url, timeout=2) as resp:
                    body = resp.read().decode()
                running_m = patterns["running"].search(body)
                waiting_m = patterns["waiting"].search(body)
                if running_m is None or waiting_m is None:
                    raise RuntimeError(
                        f"metric pattern did not match /metrics response "
                        f"(running matched={running_m is not None}, "
                        f"waiting matched={waiting_m is not None}) - "
                        f"engine's metric names/format may have changed"
                    )
                writer.writerow([f"{loop_start - t0:.3f}", running_m.group(1), waiting_m.group(1)])
                f.flush()
                n += 1
            except Exception as e:
                print(f"poll error at t={loop_start - t0:.3f}: {e}", file=sys.stderr)
            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, args.interval - elapsed))
    print(f"wrote {n} samples to {args.outfile}", file=sys.stderr)


if __name__ == "__main__":
    main()
