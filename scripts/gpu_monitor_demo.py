#!/usr/bin/env python
"""Exercise bench/common end to end: monitor the GPU while an fp16 matmul loop runs.

Usage:
    python scripts/gpu_monitor_demo.py [--seconds 5] [--size 4096]

Prints run metadata and the GPU monitor summary (paste-friendly) and saves the run to
bench/results/raw/<date>/gpu_monitor_demo_<HHMMSS>.json. Exits 0 without CUDA.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `bench` importable

from bench.common.gpu_monitor import GpuMonitor, format_summary  # noqa: E402
from bench.common.metadata import collect_metadata  # noqa: E402
from bench.common.results import save_result  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--size", type=int, default=4096, help="square matmul size (fp16)")
    ap.add_argument("--interval-ms", type=float, default=200.0)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("No CUDA device available; nothing to demo. Exiting cleanly.")
        return 0

    n = args.size
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    for _ in range(3):  # warm-up (cuBLAS heuristics, clocks ramp)
        a @ b
    torch.cuda.synchronize()

    iters = 0
    with GpuMonitor(interval_s=args.interval_ms / 1000) as mon:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            for _ in range(10):
                a @ b
            torch.cuda.synchronize()
            iters += 10
        elapsed = time.perf_counter() - t0
    result = mon.result

    meta = collect_metadata()
    tflops = 2 * n**3 * iters / elapsed / 1e12
    path = save_result(
        "gpu_monitor_demo",
        config={"seconds": args.seconds, "size": n, "dtype": "float16",
                "interval_ms": args.interval_ms},
        metrics={"iterations": iters, "elapsed_s": elapsed, "matmul_tflops": tflops},
        gpu_samples=result,
        metadata=meta,
    )

    git, gpu, pk = meta["git"], meta["gpu"], meta["packages"]
    print(f"git {git['sha'][:10]} dirty={git['dirty']} | {gpu['name']} | driver {gpu['driver']} "
          f"| CUDA {gpu['cuda_runtime']}")
    print("packages: " + ", ".join(f"{k} {v}" for k, v in pk.items()))
    print(f"matmul {n}x{n} fp16: {iters} iters in {elapsed:.2f} s -> {tflops:.1f} TFLOP/s")
    print(format_summary(result))
    print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
