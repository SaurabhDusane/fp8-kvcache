#!/usr/bin/env python
"""Baseline serving sweep: vLLM with --kv-cache-dtype auto (FP16) vs fp8.

For each KV-cache dtype: launch ``vllm serve`` (same model, --max-model-len 4096,
--gpu-memory-utilization 0.85), wait for /health, then for each trace walk a rate list with
``repeats`` runs per rate, extending the list until goodput collapses (see RatePlan), then shut
the server down and move on. Server logs are saved and their "GPU KV cache size" /
"Maximum concurrency" lines parsed into the results.

If a config fails to start (or the server dies mid-sweep), the error is captured from its log,
reported, and the sweep continues with the next config. Settings are never changed to make a
config fit.

At the end: bench/results/summary/baseline.md is generated from the saved JSON and printed.

Usage:
    python scripts/run_baseline_sweep.py                       # full sweep, defaults below
    python scripts/run_baseline_sweep.py --traces chat --rates chat=1,2 --repeats 1   # quick
    python scripts/run_baseline_sweep.py --summarize-only <sweep_id>
"""

from __future__ import annotations

import argparse
import datetime as dt
import signal
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `bench` importable

from bench.common.metadata import collect_metadata  # noqa: E402
from bench.common.results import load_results, results_root, save_result, write_summary  # noqa: E402
from bench.serving import run as serving_run  # noqa: E402
from bench.serving.sweep import (  # noqa: E402
    RatePlan, aggregate_runs, extract_startup_error, num_requests_for, parse_rate_spec,
    parse_vllm_log, seed_for, summary_markdown,
)
from bench.serving.vllm_server import ServerStartError, VllmServer, build_vllm_cmd  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TRACES = ("chat", "long_context", "multi_turn")
# Starting points only: the plan extends (x2) past the end of a list until goodput collapses.
DEFAULT_RATES: dict[str, list[float]] = {
    "chat": [1, 2, 4, 8, 16],
    "long_context": [0.5, 1, 2, 4],
    "multi_turn": [0.25, 0.5, 1, 2],   # sessions/s
    "fixed": [1, 2, 4, 8],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FP16 vs FP8 KV-cache serving baseline sweep")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=None, help="default: the model")
    ap.add_argument("--kv-cache-dtypes", nargs="+", default=["auto", "fp8"])
    ap.add_argument("--traces", nargs="+", default=list(DEFAULT_TRACES),
                    choices=list(DEFAULT_RATES))
    ap.add_argument("--rates", action="append", default=[], metavar="TRACE=R1,R2,...",
                    help="override a trace's starting rate list (repeatable)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--duration-s", type=float, default=45.0,
                    help="target seconds of arrivals per run (sets the request count)")
    ap.add_argument("--min-requests", type=int, default=30)
    ap.add_argument("--max-requests", type=int, default=300)
    ap.add_argument("--slo-ttft-ms", type=float, default=500.0)
    ap.add_argument("--slo-itl-ms", type=float, default=50.0)
    ap.add_argument("--collapse-attainment", type=float, default=0.5,
                    help="a point is collapsed when median SLO attainment is below this")
    ap.add_argument("--points-after-collapse", type=int, default=1)
    ap.add_argument("--no-extend", action="store_true",
                    help="don't extend past the rate list when goodput hasn't collapsed")
    ap.add_argument("--max-rate", type=float, default=64.0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--server-arg", action="append", default=[], dest="server_args",
                    help="extra argument passed verbatim to vllm serve (repeatable; recorded)")
    ap.add_argument("--vllm-bin", default="vllm")
    ap.add_argument("--startup-timeout-s", type=float, default=900.0)
    ap.add_argument("--sharegpt", choices=("auto", "off"), default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize-only", metavar="SWEEP_ID", default=None)
    return ap.parse_args(argv)


def estimate_minutes(args: argparse.Namespace, rates: dict[str, list[float]]) -> float:
    """Lower-bound runtime for the starting rate lists (extensions/overload drain add more)."""
    per_run_s = args.duration_s + 20  # arrivals + drain/warmup/trace build
    runs = sum(len(r) for r in rates.values()) * args.repeats * len(args.kv_cache_dtypes)
    startup_s = 120 * len(args.kv_cache_dtypes)
    return (runs * per_run_s + startup_s) / 60


def run_config(args: argparse.Namespace, sweep_id: str, dtype: str,
               rates: dict[str, list[float]], log_dir: Path,
               run_paths: list[str], plan_notes: dict[str, list[str]]) -> dict[str, Any]:
    cmd = build_vllm_cmd(args.model, dtype, port=args.port, max_model_len=args.max_model_len,
                         gpu_memory_utilization=args.gpu_memory_utilization,
                         extra_args=args.server_args, vllm_bin=args.vllm_bin)
    log_path = log_dir / f"{sweep_id}_{dtype}.log"
    status: dict[str, Any] = {"kv_cache_dtype": dtype, "cmd": cmd, "log_path": str(log_path),
                              "status": "starting"}
    print(f"\n=== [{dtype}] starting: {' '.join(cmd)}\n    log: {log_path}", flush=True)
    server = VllmServer(cmd, args.port, log_path, startup_timeout_s=args.startup_timeout_s)
    try:
        status["startup_s"] = server.start()
    except ServerStartError as exc:
        status.update(status="failed_to_start", error=f"{exc}\n{exc.log_excerpt}",
                      log_info=parse_vllm_log(server.log_text()))
        print(f"!!! [{dtype}] FAILED TO START: {exc}\n{exc.log_excerpt}", flush=True)
        return status
    status["log_info"] = info = parse_vllm_log(server.log_text())
    print(f"=== [{dtype}] up in {status['startup_s']:.0f} s; KV cache tokens "
          f"{info['kv_cache_tokens']}, max concurrency {info['max_concurrency']}x", flush=True)
    server_cfg = {"sweep_id": sweep_id, "kv_cache_dtype": dtype, "server_cmd": cmd,
                  "server_log": str(log_path), "server_log_info": info,
                  "server_startup_s": status["startup_s"]}
    try:
        for trace in args.traces:
            plan = RatePlan(rates[trace], collapse_attainment=args.collapse_attainment,
                            points_after_collapse=args.points_after_collapse,
                            extend=not args.no_extend, max_rate=args.max_rate)
            while (rate := plan.next_rate()) is not None:
                n = num_requests_for(rate, trace, args.duration_s, args.min_requests,
                                     args.max_requests)
                attainments = []
                for rep in range(args.repeats):
                    print(f"\n--- [{dtype}] {trace} rate {rate:g} repeat {rep + 1}/{args.repeats} "
                          f"({n} requests)", flush=True)
                    reset = server.reset_prefix_cache()
                    run_args = serving_run.parse_args([
                        "--base-url", server.base_url, "--model", args.model,
                        "--tokenizer", args.tokenizer or args.model, "--trace", trace,
                        "--rate", str(rate), "--num-requests", str(n),
                        "--seed", str(seed_for(args.seed, rate, rep)),
                        "--slo-ttft-ms", str(args.slo_ttft_ms), "--slo-itl-ms", str(args.slo_itl_ms),
                        "--max-model-len", str(args.max_model_len), "--sharegpt", args.sharegpt,
                        "--tag", f"baseline:{sweep_id}"])
                    metrics, path = serving_run.run(run_args, extra_config={
                        **server_cfg, "rate": rate, "repeat": rep,
                        "prefix_cache_reset": reset})
                    run_paths.append(str(path))
                    attainments.append(metrics["slo_attainment"])
                    if not server.alive():
                        raise RuntimeError(f"server died during {trace} at rate {rate:g}")
                valid = [a for a in attainments if a is not None]
                plan.record(rate, statistics.median(valid) if valid else None)
            plan_notes[f"{dtype}/{trace}"] = plan.notes
            print(f"=== [{dtype}] {trace}: " + "; ".join(plan.notes), flush=True)
        status["status"] = "ok"
    except (RuntimeError, SystemExit) as exc:  # SystemExit: run.py found the server unreachable
        status.update(status="crashed", error=f"{exc}\n{extract_startup_error(server.log_text())}")
        print(f"!!! [{dtype}] {status['error']}", flush=True)
    finally:
        rc = server.stop()
        status["exit_code"] = rc
        print(f"=== [{dtype}] server stopped (exit code {rc})", flush=True)
    return status


def summarize(sweep_id: str) -> str:
    sweeps = load_results("baseline_sweep_*")
    if sweeps.empty or sweep_id not in set(sweeps["config.sweep_id"]):
        raise SystemExit(f"no saved baseline_sweep record for sweep {sweep_id}")
    sweep = sweeps[sweeps["config.sweep_id"] == sweep_id].iloc[-1]["_raw"]
    runs = load_results("serving_*")
    recs = ([r for r in runs["_raw"] if r["config"].get("sweep_id") == sweep_id]
            if not runs.empty else [])
    md = summary_markdown(sweep, aggregate_runs(recs))
    [path] = write_summary("baseline", md)
    print("\n" + md)
    print(f"summary: {path}")
    return md


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarize_only:
        summarize(args.summarize_only)
        return 0
    rates = parse_rate_spec(args.rates, args.traces, DEFAULT_RATES)
    sweep_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = results_root() / "raw" / dt.date.today().isoformat() / "vllm_logs"
    print(f"sweep {sweep_id}: configs {args.kv_cache_dtypes}, traces {args.traces}, "
          f"starting rates {rates}, {args.repeats} repeats; at least "
          f"~{estimate_minutes(args, rates):.0f} min before extensions", flush=True)

    # Turn SIGTERM into KeyboardInterrupt so the server is always shut down.
    prev_handler = signal.signal(signal.SIGTERM,
                                 lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    servers: list[dict[str, Any]] = []
    run_paths: list[str] = []
    plan_notes: dict[str, list[str]] = {}
    state = "complete"
    try:
        for dtype in args.kv_cache_dtypes:
            servers.append(run_config(args, sweep_id, dtype, rates, log_dir, run_paths, plan_notes))
    except KeyboardInterrupt:
        state = "interrupted"
        print("\ninterrupted; saving what was measured", flush=True)
    finally:
        signal.signal(signal.SIGTERM, prev_handler)
    config = {**{k: v for k, v in vars(args).items() if k != "summarize_only"},
              "sweep_id": sweep_id, "rates": rates, "state": state}
    save_result("baseline_sweep", config,
                {"servers": servers, "run_paths": run_paths, "plan_notes": plan_notes},
                None, metadata=collect_metadata())
    summarize(sweep_id)
    failed = [s["kv_cache_dtype"] for s in servers if s["status"] != "ok"]
    if failed:
        print(f"\nWARNING: configs not fully measured: {failed} (see Server configs section)")
    return 1 if failed or state != "complete" else 0


if __name__ == "__main__":
    sys.exit(main())
