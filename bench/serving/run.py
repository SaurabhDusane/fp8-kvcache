"""Serving benchmark CLI.

    python -m bench.serving.run --trace multi_turn --rate 2 --num-requests 200 \
        --slo-ttft-ms 500 --slo-itl-ms 50

Builds a deterministic trace (lengths measured with the model's tokenizer), sends a few warmup
requests (separate prompts, discarded), runs the trace with the GPU monitor on, prints a compact
summary and saves everything (config, trace stats, metrics, per-request records, GPU samples)
via bench/common/results.py as ``serving_<trace>``.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from bench.common.gpu_monitor import GpuMonitor, format_summary
from bench.common.results import save_result
from bench.serving.loadgen import run_trace
from bench.serving.metrics import compute_metrics, format_table
from bench.serving.tokenization import load_tokenizer
from bench.serving.traces import TRACES, build_fixed, build_trace

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="async load generator for an OpenAI-compatible server")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default=None, help="served model id (default: first of /v1/models)")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer for length measurement (default: the model; 'stub' for tests)")
    ap.add_argument("--trace", choices=TRACES, required=True)
    ap.add_argument("--rate", type=float, default=math.inf,
                    help="Poisson session arrival rate per second (default inf = all at once)")
    ap.add_argument("--max-concurrency", type=int, default=None,
                    help="cap on sessions in flight; with --rate inf this is closed-loop")
    ap.add_argument("--num-requests", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slo-ttft-ms", type=float, default=500.0)
    ap.add_argument("--slo-itl-ms", type=float, default=50.0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--think-time-s", type=float, default=1.0, help="multi_turn mean think time")
    ap.add_argument("--input-len", type=int, default=512, help="fixed trace user-content tokens")
    ap.add_argument("--output-len", type=int, default=128, help="fixed trace output tokens")
    ap.add_argument("--sharegpt", choices=("auto", "off"), default="auto",
                    help="chat trace: try ShareGPT (HF-cached) or use the synthetic fallback")
    ap.add_argument("--warmup-requests", type=int, default=4)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--tag", default="", help="free-form label stored in the result config")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-gpu-monitor", action="store_true")
    return ap.parse_args(argv)


def resolve_model(base_url: str, model: str | None) -> tuple[str, list[str]]:
    """Check the server is up and return (model id to use, served model ids)."""
    try:
        r = httpx.get(base_url.rstrip("/") + "/v1/models", timeout=10)
        r.raise_for_status()
        served = [m["id"] for m in r.json().get("data", [])]
    except Exception as exc:
        raise SystemExit(f"server at {base_url} not reachable ({type(exc).__name__}: {exc}). "
                         "Is vLLM running?") from exc
    if model is None:
        if not served:
            raise SystemExit("server lists no models; pass --model")
        model = served[0]
    elif served and model not in served:
        raise SystemExit(f"model {model!r} not served; server has {served}")
    return model, served


def trace_options(args: argparse.Namespace) -> dict[str, Any]:
    if args.trace == "multi_turn":
        return {"think_time_mean_s": args.think_time_s}
    if args.trace == "fixed":
        return {"input_len": args.input_len, "output_len": args.output_len}
    if args.trace == "chat":
        return {"sharegpt": args.sharegpt}
    return {}


def run(args: argparse.Namespace, extra_config: dict[str, Any] | None = None,
        ) -> tuple[dict[str, Any], Path | None]:
    """Run one benchmark; returns (metrics, saved result path or None).

    ``extra_config`` is merged into the saved config (used by sweeps for sweep id, kv dtype,
    rate, repeat, server info)."""
    model, served = resolve_model(args.base_url, args.model)
    tok = load_tokenizer(args.tokenizer or model)
    trace = build_trace(args.trace, tok, args.num_requests, seed=args.seed,
                        max_model_len=args.max_model_len, **trace_options(args))
    # Warmup prompts are separate random text (different seed) so they don't prime the prefix
    # cache for measured requests.
    warm = build_fixed(tok, args.warmup_requests, np.random.default_rng(args.seed + 10_000),
                       args.max_model_len, input_len=32, output_len=16)
    stats = trace.stats()
    print(f"trace {trace.name} ({trace.source}): {stats['num_requests']} requests in "
          f"{stats['num_sessions']} sessions; prompt tokens median "
          f"{stats['prompt_tokens'].get('median')}, output median {stats['output_tokens'].get('median')}"
          + (f"; fallback: {trace.meta['fallback_reason']}" if "fallback_reason" in trace.meta else ""))

    monitor = None if args.no_gpu_monitor else GpuMonitor(interval_s=0.2)
    if monitor:
        monitor.start()
    try:
        records = asyncio.run(run_trace(
            trace, args.base_url, model, rate=args.rate, max_concurrency=args.max_concurrency,
            seed=args.seed, timeout_s=args.timeout_s, warmup_turns=list(warm.turns())))
    finally:
        gpu = monitor.stop() if monitor else None
    metrics = compute_metrics(records, args.slo_ttft_ms, args.slo_itl_ms)
    metrics["warmup_requests"] = sum(r.warmup for r in records)

    mode = (f"Poisson {args.rate:g}/s" if math.isfinite(args.rate) else "all at t=0") + (
        f", max concurrency {args.max_concurrency}" if args.max_concurrency else "")
    title = f"== serving {trace.name} | {model} | {mode} =="
    print(format_table(metrics, title))
    if gpu is not None:
        print(format_summary(gpu))

    if not args.no_save:
        config = {k: (str(v) if isinstance(v, float) and not math.isfinite(v) else v)
                  for k, v in vars(args).items()}
        config.update(model=model, served_models=served, trace_stats=stats,
                      arrival_mode=mode, **(extra_config or {}))
        path = save_result(f"serving_{trace.name}", config,
                           {**metrics, "records": [r.to_dict() for r in records]}, gpu)
        print(f"saved: {path}")
        return metrics, path
    return metrics, None


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
