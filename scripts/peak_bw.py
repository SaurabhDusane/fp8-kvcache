#!/usr/bin/env python
"""Measure achievable DRAM bandwidth on the GPU (the roofline ceiling for later kernels).

Methods (fp16 buffers):
  torch_copy     dst.copy_(src)               bytes = read N + write N
  triton_copy    Triton elementwise copy       bytes = read N + write N
  triton_reduce  Triton read-only sum          bytes = read N + 4 B per program (partial sums)

Decode attention is read-dominated, so ``triton_reduce`` is the ceiling later kernels are
compared against (``metrics.read_peak_gbps`` in the saved JSON).

Method: for each buffer size (default 256, 512, 1024 MiB; skipped if free VRAM is short),
run ``repeats`` rounds; each round times every method once with ``triton.testing.do_bench``
(which flushes L2 before each call), so methods are interleaved (A B C, B C A, ...). Per round
we keep the median and p20/p80 time; per (method, size) we report the median GB/s across
rounds and its min/max. The GPU monitor runs throughout; each timed call gets its own median SM
clock and is flagged throttled if >15% below the session max (CLAUDE.md rule 4).

Usage:
    python scripts/peak_bw.py                       # full run (plugged in, best performance)
    python scripts/peak_bw.py --sizes-mib 256 --repeats 2     # quick check
    python scripts/peak_bw.py --from-json <raw json>          # rebuild summary, no GPU needed

Outputs: bench/results/raw/<date>/peak_bw_<HHMMSS>.json and bench/results/summary/peak_bw.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `bench` importable

from bench.common.gpu_monitor import GpuMonitor, format_summary, is_throttled  # noqa: E402
from bench.common.metadata import collect_metadata  # noqa: E402
from bench.common.results import save_result, write_summary  # noqa: E402

MIB = 1 << 20
METHODS: tuple[str, ...] = ("torch_copy", "triton_copy", "triton_reduce")
DEFAULT_SIZES_MIB: tuple[int, ...] = (256, 512, 1024)

# Spec-sheet DRAM bandwidth for the RTX 4080 Laptop GPU: 12 GB GDDR6, 192-bit bus, 18 Gbps
# -> 18 * 192 / 8 = 432 GB/s. UNVERIFIED: laptop OEMs can configure memory clocks differently.
# Verify against `nvidia-smi -q -d CLOCK` (Max Clocks -> Memory) and the OEM spec; the run
# records nvidia-smi's max memory clock for that cross-check. Override with --spec-gbps.
SPEC_GBPS = 432.0
SPEC_SOURCE = "RTX 4080 Laptop spec sheet (192-bit GDDR6 @ 18 Gbps); UNVERIFIED"

# Headroom kept free beyond our buffers: do_bench's 256 MiB L2-flush buffer, the CUDA context,
# and the Windows display sharing this VRAM under WSL.
DEFAULT_RESERVE_MIB = 768


def bytes_moved(method: str, size_bytes: int, n_programs: int = 0) -> int:
    """DRAM bytes that must move for one call of ``method`` on a ``size_bytes`` buffer."""
    if method in ("torch_copy", "triton_copy"):
        return 2 * size_bytes  # read src + write dst
    if method == "triton_reduce":
        return size_bytes + 4 * n_programs  # read src + one fp32 partial per program
    raise ValueError(f"unknown method {method!r}")


def required_bytes(size_bytes: int, reserve_bytes: int) -> int:
    """VRAM needed to run all methods at this size: src + dst + headroom."""
    return 2 * size_bytes + reserve_bytes


def gbps(n_bytes: int, ms: float) -> float:
    return n_bytes / (ms * 1e-3) / 1e9


def method_order(round_idx: int) -> list[str]:
    """Rotate the start method each round (A B C, B C A, C A B) to spread position bias."""
    k = round_idx % len(METHODS)
    return list(METHODS[k:] + METHODS[:k])


def nvidia_smi_max_mem_clock_mhz() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.max.memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


# --------------------------------------------------------------------------- aggregation

def aggregate(measurements: list[dict[str, Any]], spec_gbps: float) -> list[dict[str, Any]]:
    """One row per (method, size) from per-round measurements, in METHODS x size order."""
    rows = []
    keys = sorted({(m["method"], m["size_bytes"]) for m in measurements},
                  key=lambda k: (k[1], METHODS.index(k[0])))
    for method, size in keys:
        ms = [m for m in measurements if m["method"] == method and m["size_bytes"] == size]
        bw = [m["gbps"] for m in ms]
        med = statistics.median(bw)
        flags = [m["throttled"] for m in ms]
        rows.append({
            "method": method,
            "size_mib": size // MIB,
            "bytes_moved": ms[0]["bytes_moved"],
            "repeats": len(ms),
            "gbps_median": med,
            "gbps_min": min(bw),
            "gbps_max": max(bw),
            "time_ms_median": statistics.median(m["ms_median"] for m in ms),
            "pct_of_spec": 100.0 * med / spec_gbps if spec_gbps else None,
            "throttled_repeats": sum(1 for f in flags if f),
            "unknown_throttle_repeats": sum(1 for f in flags if f is None),
        })
    return rows


def peaks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, methods in (("read", ("triton_reduce",)), ("copy", ("torch_copy", "triton_copy"))):
        cand = [r for r in rows if r["method"] in methods]
        if cand:
            best = max(cand, key=lambda r: r["gbps_median"])
            out[f"{label}_peak_gbps"] = best["gbps_median"]
            out[f"{label}_peak_method"] = best["method"]
            out[f"{label}_peak_size_mib"] = best["size_mib"]
    return out


def summary_markdown(record: dict[str, Any]) -> str:
    """Summary built only from a saved result record (CLAUDE.md rule 7)."""
    meta, cfg, met = record["metadata"], record["config"], record["metrics"]
    gpu = meta.get("gpu", {})
    git = meta.get("git", {})
    throttle = (record.get("gpu") or {}).get("throttle") or {}
    spec = cfg["spec_gbps"]
    lines = [
        "# Peak DRAM bandwidth",
        "",
        f"- Run: {record.get('saved_at')} · git `{str(git.get('sha'))[:10]}` "
        f"(dirty: {git.get('dirty')}) · {gpu.get('name')} · driver {gpu.get('driver')}",
        f"- Spec: {spec:g} GB/s ({cfg['spec_source']}); nvidia-smi max memory clock: "
        f"{cfg.get('nvidia_smi_max_mem_clock_mhz')} MHz",
        f"- do_bench: warmup {cfg['warmup_ms']} ms, rep {cfg['rep_ms']} ms, L2 flushed per call; "
        f"{cfg['repeats']} interleaved rounds; dtype {cfg['dtype']}; "
        f"reduce grid = {cfg.get('reduce_programs')} programs",
        f"- GPU throttled (whole run): {throttle.get('throttled')} "
        f"(median SM {throttle.get('median_sm_mhz')} MHz vs max {throttle.get('session_max_sm_mhz')} MHz)",
    ]
    if cfg.get("skipped_sizes_mib"):
        lines.append(f"- Skipped sizes (insufficient free VRAM): {cfg['skipped_sizes_mib']} MiB")
    lines += [
        "",
        "| method | size MiB | GB/s median | GB/s min–max | % of spec | ms median | throttled rounds |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in met["rows"]:
        pct = "n/a" if r["pct_of_spec"] is None else f"{r['pct_of_spec']:.1f}"
        thr = f"{r['throttled_repeats']}/{r['repeats']}"
        if r.get("unknown_throttle_repeats"):
            thr += f" ({r['unknown_throttle_repeats']} unknown)"
        lines.append(
            f"| {r['method']} | {r['size_mib']} | {r['gbps_median']:.1f} | "
            f"{r['gbps_min']:.1f}–{r['gbps_max']:.1f} | {pct} | {r['time_ms_median']:.3f} | {thr} |")
    if "read_peak_gbps" in met:
        lines += ["", f"**Read-only peak (decode ceiling): {met['read_peak_gbps']:.1f} GB/s** "
                      f"({met['read_peak_method']}, {met['read_peak_size_mib']} MiB)"]
    if "copy_peak_gbps" in met:
        lines.append(f"Copy peak: {met['copy_peak_gbps']:.1f} GB/s "
                     f"({met['copy_peak_method']}, {met['copy_peak_size_mib']} MiB)")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- GPU run

def _check_correctness(torch, kernels, src, dst, partials) -> None:
    """Fail loudly if a method computes the wrong thing (cheap relative to the benchmark)."""
    dst.zero_()
    dst.copy_(src)
    if not torch.equal(dst, src):
        raise RuntimeError("torch_copy produced a wrong result")
    dst.zero_()
    kernels.triton_copy(src, dst)
    if not torch.equal(dst, src):
        raise RuntimeError("triton_copy produced a wrong result")
    got = kernels.triton_reduce_sum(src, partials).double().sum().item()
    ref = torch.sum(src, dtype=torch.float64).item()
    # Inputs are uniform in [0, 1): a dropped/duplicated 8192-element block shifts the sum by
    # ~4096, i.e. >1.5e-5 relative even at 1 GiB, above fp32 accumulation error (~1e-7).
    if abs(got - ref) > 1e-5 * abs(ref):
        raise RuntimeError(f"triton_reduce sum mismatch: got {got}, expected {ref}")


def run(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        print("No CUDA device available; peak_bw needs a GPU. Exiting cleanly.")
        return 0
    import triton.testing

    from bench.kernels import bw_kernels as kernels

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    n_programs = props.multi_processor_count * args.reduce_programs_per_sm
    reserve = args.reserve_mib * MIB
    quantiles = [0.5, 0.2, 0.8]
    measurements: list[dict[str, Any]] = []
    skipped: list[int] = []
    run_sizes: list[int] = []

    with GpuMonitor(interval_s=0.2) as mon:
        for size_mib in args.sizes_mib:
            size = size_mib * MIB
            free, _ = torch.cuda.mem_get_info(dev)
            need = required_bytes(size, reserve)
            if free < need:
                print(f"skip {size_mib} MiB: need {need / MIB:.0f} MiB free, have {free / MIB:.0f} MiB")
                skipped.append(size_mib)
                continue
            run_sizes.append(size_mib)
            n = size // 2  # fp16 elements
            # Random (incompressible) data: avoids any benefit from memory compression of zeros.
            g = torch.Generator(device=dev).manual_seed(0)
            src = torch.rand(n, generator=g, device=dev, dtype=torch.float16)
            dst = torch.empty_like(src)
            partials = torch.empty(n_programs, device=dev, dtype=torch.float32)
            _check_correctness(torch, kernels, src, dst, partials)

            fns: dict[str, Callable[[], object]] = {
                "torch_copy": lambda: dst.copy_(src),
                "triton_copy": lambda: kernels.triton_copy(src, dst),
                "triton_reduce": lambda: kernels.triton_reduce_sum(src, partials),
            }
            for rnd in range(args.repeats):
                for method in method_order(rnd):
                    t0 = mon.elapsed()
                    med, p20, p80 = triton.testing.do_bench(
                        fns[method], warmup=args.warmup_ms, rep=args.rep_ms, quantiles=quantiles)
                    t1 = mon.elapsed()
                    nb = bytes_moved(method, size, n_programs)
                    measurements.append({
                        "method": method, "size_bytes": size, "round": rnd,
                        "ms_median": med, "ms_p20": p20, "ms_p80": p80,
                        "bytes_moved": nb, "gbps": gbps(nb, med),
                        "gbps_p20_time": gbps(nb, p20), "gbps_p80_time": gbps(nb, p80),
                        "t_start": t0, "t_end": t1,
                    })
                    print(f"  round {rnd} {method:<14} {size_mib:>5} MiB  {gbps(nb, med):8.1f} GB/s")
            del src, dst, partials
            torch.cuda.empty_cache()
    result = mon.result

    # Per-call throttle flag: median SM clock during that call vs the session max.
    session_max = result.summary["sm_clock_mhz"]["max"]
    for m in measurements:
        m["sm_clock_median_mhz"] = result.window_median("sm_clock_mhz", m["t_start"], m["t_end"])
        m["throttled"] = is_throttled(m["sm_clock_median_mhz"], session_max)

    if not measurements:
        print("Nothing measured (all sizes skipped).")
        return 1
    rows = aggregate(measurements, args.spec_gbps)
    config = {
        "sizes_mib": run_sizes, "skipped_sizes_mib": skipped, "repeats": args.repeats,
        "warmup_ms": args.warmup_ms, "rep_ms": args.rep_ms, "quantiles": quantiles,
        "dtype": "float16", "methods": list(METHODS),
        "reduce_programs": n_programs, "reduce_programs_per_sm": args.reduce_programs_per_sm,
        "sm_count": props.multi_processor_count, "reserve_mib": args.reserve_mib,
        "spec_gbps": args.spec_gbps, "spec_source": SPEC_SOURCE,
        "nvidia_smi_max_mem_clock_mhz": nvidia_smi_max_mem_clock_mhz(),
    }
    metrics = {"rows": rows, **peaks(rows), "measurements": measurements}
    path = save_result("peak_bw", config, metrics, result, metadata=collect_metadata())
    return report(path)


def report(path: Path) -> int:
    record = json.loads(Path(path).read_text())
    md = summary_markdown(record)
    [md_path] = write_summary("peak_bw", md)
    print()
    print(md)
    print(format_summary_from_record(record))
    print(f"saved: {path}")
    print(f"summary: {md_path}")
    return 0


def format_summary_from_record(record: dict[str, Any]) -> str:
    from bench.common.gpu_monitor import MonitorResult

    gpu = record.get("gpu") or {}
    res = MonitorResult(samples=gpu.get("samples", []), interval_s=gpu.get("interval_s", 0.2),
                        duration_s=gpu.get("duration_s", 0.0), errors=gpu.get("errors", 0),
                        warnings=gpu.get("warnings", []))
    return format_summary(res)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes-mib", type=int, nargs="+", default=list(DEFAULT_SIZES_MIB))
    ap.add_argument("--repeats", type=int, default=5, help="interleaved rounds (>=5 per CLAUDE.md)")
    ap.add_argument("--warmup-ms", type=int, default=25)
    ap.add_argument("--rep-ms", type=int, default=200)
    ap.add_argument("--reduce-programs-per-sm", type=int, default=4)
    ap.add_argument("--reserve-mib", type=int, default=DEFAULT_RESERVE_MIB)
    ap.add_argument("--spec-gbps", type=float, default=SPEC_GBPS)
    ap.add_argument("--from-json", type=Path, help="rebuild summary from a saved run; no GPU")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.from_json:
        return report(args.from_json)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
