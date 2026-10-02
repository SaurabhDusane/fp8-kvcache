"""Quantization error on real KV (a measurement, not a test).

For each captured layer (tests/data, from scripts/capture_kv.py): build an fp16 paged cache and
an fp8 paged cache from the same K/V, run the same attention implementation on both with the
captured next-step queries, and compare the outputs (fp8 vs fp16):

  max abs, mean abs, max abs relative to max |fp16 output|, cosine similarity over the whole
  layer (all prompts' outputs flattened), and the worst per-prompt cosine.

``--impl reference`` (default) uses the PyTorch reference, so the numbers are pure quantization
error. With a registered kernel (``--impl v1_fp8_paged``) they also include kernel error.
``--granularities`` compares fp8 scale granularities (default: the chosen per-KV-head).

Usage:
    python -m bench.kernels.quant_error [--impl reference] [--granularities kv_head tensor ...]

Saves bench/results/raw/<date>/quant_error_<HHMMSS>.json and writes
bench/results/summary/quant_error.md (generated from the saved JSON), and prints it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import torch

from bench.common.gpu_monitor import GpuMonitor
from bench.common.metadata import collect_metadata
from bench.common.results import save_result, write_summary
from bench.kernels.kv_capture import DATA_DIR, load_captured_paged, read_manifest
from kvcache.cache.fp8 import GRANULARITIES
from kvcache.reference.decode_attention import paged_decode_attention


def compare_outputs(ref: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    """fp8 (``test``) vs fp16 (``ref``) outputs, each [batch, heads, head_dim]."""
    r, t = ref.float(), test.float()
    diff = (t - r).abs()
    per_prompt_cos = torch.nn.functional.cosine_similarity(t.flatten(1), r.flatten(1), dim=1)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_abs_rel": float(diff.max() / r.abs().max().clamp_min(1e-30)),
        "cosine": float(torch.nn.functional.cosine_similarity(t.flatten(), r.flatten(), dim=0)),
        "min_prompt_cosine": float(per_prompt_cos.min()),
        "min_prompt_index": int(per_prompt_cos.argmin()),
        "ref_mean_abs": float(r.abs().mean()),
    }


def get_impl(name: str) -> tuple[Callable[..., torch.Tensor], tuple[str, ...]]:
    """(attention fn, supported fp8 granularities)."""
    if name == "reference":
        return paged_decode_attention, GRANULARITIES
    from kvcache.kernels import get_kernel

    spec = get_kernel(name)
    if not spec.supports_fp8:
        raise SystemExit(f"kernel {name} does not support fp8 caches")
    return spec.fn, spec.scale_granularities


def measure_layer(layer: int, impl: Callable[..., torch.Tensor], granularity: str,
                  block_size: int, device: str, data_dir: Path) -> dict[str, Any]:
    c16, q, payload = load_captured_paged(layer, block_size, kv_dtype="fp16", device=device,
                                          data_dir=data_dir)
    c8, _, _ = load_captured_paged(layer, block_size, kv_dtype="fp8", scale_granularity=granularity,
                                   device=device, data_dir=data_dir)
    sm = payload["sm_scale"]
    out16 = impl(q, c16.key_cache, c16.value_cache, c16.block_tables, c16.context_lens, sm)
    out8 = impl(q, c8.key_cache, c8.value_cache, c8.block_tables, c8.context_lens, sm,
                k_scale=c8.k_scale, v_scale=c8.v_scale)
    m = compare_outputs(out16, out8)
    m.update(layer=layer, granularity=granularity, num_prompts=q.shape[0],
             tokens=int(c16.context_lens.sum()),
             worst_prompt=payload["prompts"][m["min_prompt_index"]])
    if granularity in ("tensor", "kv_head"):  # small enough to record
        m["k_scale"] = c8.k_scale.flatten().tolist()
        m["v_scale"] = c8.v_scale.flatten().tolist()
    return m


def _g(v: float) -> str:
    return f"{v:.3e}" if abs(v) < 1e-2 else f"{v:.4f}"


def summary_markdown(record: dict[str, Any]) -> str:
    cfg, rows = record["config"], record["metrics"]["rows"]
    meta = record.get("metadata", {})
    git = meta.get("git", {})
    lines = [
        "# FP8 KV-cache quantization error on real KV",
        "",
        f"- Model `{cfg['model']}` · captured layers {cfg['layers']} · {rows[0]['num_prompts'] if rows else 0} "
        f"prompts · block size {cfg['block_size']} · impl `{cfg['impl']}` · device {cfg['device']}",
        f"- git `{str(git.get('sha'))[:10]}` (dirty: {git.get('dirty')}) · {meta.get('gpu', {}).get('name')}",
        "- fp8 (e4m3, scale = amax/448, static) cache vs fp16 cache, same attention implementation, "
        "same captured queries. "
        + ("Reference implementation: pure quantization error." if cfg["impl"] == "reference"
           else "Includes kernel error of the implementation."),
        "",
        "| layer | scale granularity | max abs | mean abs | max abs / max abs(fp16) | cosine | "
        "worst prompt cosine (prompt) | mean abs(fp16 out) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['layer']} | {r['granularity']} | {_g(r['max_abs'])} | {_g(r['mean_abs'])} | "
            f"{_g(r['max_abs_rel'])} | {r['cosine']:.6f} | {r['min_prompt_cosine']:.6f} "
            f"({r['worst_prompt']}) | {_g(r['ref_mean_abs'])} |")
    scales = [r for r in rows if "k_scale" in r]
    if scales:
        lines += ["", "Scales used (amax/448):", ""]
        for r in scales:
            lines.append(f"- layer {r['layer']} {r['granularity']}: k_scale "
                         f"{[round(x, 6) for x in r['k_scale']]}, v_scale "
                         f"{[round(x, 6) for x in r['v_scale']]}")
    for note in record["metrics"].get("notes", []):
        lines.append(f"- note: {note}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FP8 vs FP16 KV-cache attention error on real KV")
    ap.add_argument("--impl", default="reference", help="'reference' or a registered kernel name")
    ap.add_argument("--granularities", nargs="+", default=["kv_head"],
                    help=f"fp8 scale granularities to compare, from {list(GRANULARITIES)} or 'all'")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="default: all captured")
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args(argv)

    manifest = read_manifest(args.data_dir)
    if manifest is None:
        print(f"No captured KV in {args.data_dir}. Run scripts/capture_kv.py on the GPU first.")
        return 1
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    grans = list(GRANULARITIES) if args.granularities == ["all"] else args.granularities
    impl, supported = get_impl(args.impl)
    layers = args.layers if args.layers is not None else manifest["layers"]
    rows, notes = [], []
    with GpuMonitor() as mon:
        for layer in layers:
            for g in grans:
                if g not in supported:
                    notes.append(f"{args.impl} does not support '{g}' scales; skipped")
                    continue
                rows.append(measure_layer(layer, impl, g, args.block_size, device, args.data_dir))
    config = {"impl": args.impl, "granularities": grans, "block_size": args.block_size,
              "layers": layers, "device": device, "model": manifest.get("model"),
              "data_files": manifest.get("files")}
    path = save_result("quant_error", config, {"rows": rows, "notes": sorted(set(notes))},
                       mon.result if device == "cuda" else None, metadata=collect_metadata())
    md = summary_markdown(json.loads(path.read_text()))
    [md_path] = write_summary("quant_error", md)
    print(md)
    print(f"saved: {path}\nsummary: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
