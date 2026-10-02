"""Captured real KV: transformers cache access, statistics, file format, and loading into our
paged layout. No model or GPU needed here (scripts/capture_kv.py does the capture).

Files (tests/data/, gitignored):
  kv_layer{L:02d}.pt   dict: keys/values (list of [n_i, num_kv_heads, head_dim] fp16, post-RoPE,
                       as stored in the cache, including the decode step's own token), queries
                       (list of [num_q_heads, head_dim] fp16, post-RoPE query of that decode step),
                       prompt names, sm_scale, model config facts, self-check errors.
  kv_manifest.json     model, layers captured, prompt names and token counts.

Statistics, per (layer, K/V, KV head), over all captured tokens:
  amax, amax excluding each prompt's first token (attention-sink/BOS outliers), rms;
  channel dominance: the channel with the largest amax, its ratio to the median channel amax,
  and the share of total energy (sum of squares) in the top-4 channels; "dominated" if the
  ratio >= DOMINANCE_RATIO or the top-4 share >= DOMINANCE_TOP4;
  FP8 underflow: fraction of nonzero elements that would be e4m3 subnormal (|x| < 2^-6 * scale)
  with a per-head scale vs a per-tensor (per layer) scale, scale = amax / 448.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Sequence

import torch

from bench.common.metadata import REPO_ROOT
from kvcache.cache.fp8 import FP8_MAX
from kvcache.cache.paged import PagedKVCache, build_paged_kv_cache

DATA_DIR = REPO_ROOT / "tests" / "data"
MANIFEST = "kv_manifest.json"
DOMINANCE_RATIO = 8.0
DOMINANCE_TOP4 = 0.5
SUBNORMAL_FACTOR = 2.0**-6  # e4m3 smallest normal = 2^-6 (in units of the scale)


def layer_file(layer: int) -> str:
    return f"kv_layer{layer:02d}.pt"


# --------------------------------------------------------------------------- transformers cache

def cache_layer_kv(cache: Any, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(keys, values) [batch, num_kv_heads, seq, head_dim] for one layer, across transformers
    Cache API versions: ``cache.layers[i].keys`` (>=4.56), ``cache.key_cache[i]`` (older
    DynamicCache), ``cache[i]`` (tuple-style indexing), or a legacy tuple of tuples."""
    layers = getattr(cache, "layers", None)
    if layers is not None and hasattr(layers[layer], "keys") and torch.is_tensor(layers[layer].keys):
        return layers[layer].keys, layers[layer].values
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer], cache.value_cache[layer]
    item = cache[layer]
    return item[0], item[1]


# --------------------------------------------------------------------------- statistics

def head_stats(chunks: Sequence[torch.Tensor], tensor_amax: float | None = None) -> list[dict[str, Any]]:
    """Per-head stats for a list of per-prompt tensors [n_i, num_heads, head_dim]."""
    x = torch.cat([c.float() for c in chunks])                      # [N, H, d]
    rest = [c.float()[1:] for c in chunks if c.shape[0] > 1]
    x_rest = torch.cat(rest) if rest else x[:0]
    absx = x.abs()
    t_amax = float(absx.max()) if tensor_amax is None else tensor_amax
    out = []
    for h in range(x.shape[1]):
        a = absx[:, h]                                               # [N, d]
        amax = float(a.max())
        ch_amax = a.amax(dim=0)                                      # [d]
        energy = (x[:, h] ** 2).sum(dim=0)                           # [d]
        top_ch = int(ch_amax.argmax())
        med = float(ch_amax.median())
        ratio = float(ch_amax.max()) / med if med > 0 else float("inf")
        top4 = float(energy.topk(min(4, energy.numel())).values.sum() / energy.sum()) \
            if float(energy.sum()) > 0 else 0.0
        nz = a[a > 0]

        def subnormal(scale_amax: float) -> float:
            thr = SUBNORMAL_FACTOR * scale_amax / FP8_MAX
            return float((nz < thr).float().mean()) if nz.numel() else 0.0

        out.append({
            "head": h, "amax": amax,
            "amax_excl_first": float(x_rest[:, h].abs().max()) if x_rest.numel() else None,
            "rms": float(x[:, h].pow(2).mean().sqrt()),
            "top_channel": top_ch, "top_median_ratio": ratio, "top4_energy": top4,
            "dominated": bool(ratio >= DOMINANCE_RATIO or top4 >= DOMINANCE_TOP4),
            "subnormal_frac_head_scale": subnormal(amax),
            "subnormal_frac_tensor_scale": subnormal(t_amax),
        })
    return out


def layer_stats(layer: int, keys: Sequence[torch.Tensor],
                values: Sequence[torch.Tensor]) -> list[dict[str, Any]]:
    """Rows for one layer: K and V, per KV head (tensor scale = this layer's K or V amax)."""
    rows = []
    for kind, chunks in (("K", keys), ("V", values)):
        t_amax = max(float(c.float().abs().max()) for c in chunks)
        for r in head_stats(chunks, t_amax):
            rows.append({"layer": layer, "kind": kind, **r})
    return rows


# --------------------------------------------------------------------------- markdown

def _f(v: Any, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}g}" if abs(v) < 1e4 else f"{v:.0f}"
    return str(v)


def stats_markdown(record: dict[str, Any]) -> str:
    """kv_stats.md from a saved ``kv_stats`` result record (generated, never hand-edited)."""
    cfg, met = record["config"], record["metrics"]
    meta = record.get("metadata", {})
    rows: list[dict[str, Any]] = met["rows"]
    heads = sorted({r["head"] for r in rows})
    layers = sorted({r["layer"] for r in rows})
    saved = cfg.get("saved_layers", [])
    git = meta.get("git", {})
    lines = [
        "# KV-cache statistics (real model KV)",
        "",
        f"- Model `{cfg['model']}` · {cfg['num_layers']} layers · {cfg['num_q_heads']} query heads · "
        f"{cfg['num_kv_heads']} KV heads · head_dim {cfg['head_dim']}",
        f"- {len(cfg['prompts'])} prompts, {cfg['total_tokens']} tokens in total "
        f"(per prompt {min(cfg['prompt_tokens'])}–{max(cfg['prompt_tokens'])}); keys post-RoPE as "
        "stored in the cache, including the decode step's token",
        f"- git `{str(git.get('sha'))[:10]}` (dirty: {git.get('dirty')}) · "
        f"{meta.get('gpu', {}).get('name')} · torch {meta.get('packages', {}).get('torch')} · "
        f"transformers {cfg.get('transformers_version')}",
        f"- Dominated = top/median channel amax >= {DOMINANCE_RATIO:g} or top-4 channels hold >= "
        f"{100 * DOMINANCE_TOP4:g}% of the energy. Subnormal % = nonzero elements below the e4m3 "
        "normal range (|x| < 2^-6 · amax/448) with a per-head vs per-tensor (per-layer) scale.",
        "",
        "## All layers: amax per KV head",
        "",
        "| layer | " + " | ".join(f"K h{h}" for h in heads) + " | "
        + " | ".join(f"V h{h}" for h in heads)
        + " | K max top/median ch | V max top/median ch | dominated heads (K / V) |",
        "|" + "---|" * (1 + 2 * len(heads) + 3),
    ]
    for L in layers:
        lr = [r for r in rows if r["layer"] == L]
        k = {r["head"]: r for r in lr if r["kind"] == "K"}
        v = {r["head"]: r for r in lr if r["kind"] == "V"}
        kd = sum(r["dominated"] for r in k.values())
        vd = sum(r["dominated"] for r in v.values())
        lines.append(
            f"| {L}{' *' if L in saved else ''} | "
            + " | ".join(_f(k[h]["amax"]) for h in heads) + " | "
            + " | ".join(_f(v[h]["amax"]) for h in heads) + " | "
            + f"{_f(max(r['top_median_ratio'] for r in k.values()))} | "
            + f"{_f(max(r['top_median_ratio'] for r in v.values()))} | {kd} / {vd} |")
    lines += ["", "\\* = layer saved to tests/data.", "",
              "## Saved layers: detail per head", "",
              "| layer | kv | head | amax | amax excl. 1st token | rms | top ch | top/median ch | "
              "top-4 energy % | dominated | subnormal % (head scale) | subnormal % (tensor scale) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["layer"] not in saved:
            continue
        lines.append(
            f"| {r['layer']} | {r['kind']} | {r['head']} | {_f(r['amax'])} | "
            f"{_f(r['amax_excl_first'])} | {_f(r['rms'])} | {r['top_channel']} | "
            f"{_f(r['top_median_ratio'])} | {100 * r['top4_energy']:.1f} | "
            f"{'yes' if r['dominated'] else 'no'} | {100 * r['subnormal_frac_head_scale']:.3f} | "
            f"{100 * r['subnormal_frac_tensor_scale']:.3f} |")
    checks = met.get("capture_check") or {}
    if checks:
        lines += ["", "## Capture self-check",
                  "", "Reference paged attention on the captured q/K/V vs the model's own attention "
                  "output (input of o_proj) at the decode step, fp16 model output:", "",
                  "| layer | max abs err (worst prompt) | median over prompts |", "|---|---|---|"]
        for L, errs in sorted(checks.items(), key=lambda kv: int(kv[0])):
            lines.append(f"| {L} | {_f(max(errs))} | {_f(statistics.median(errs))} |")
    files = met.get("files") or {}
    if files:
        total = sum(files.values())
        lines += ["", f"Files in tests/data: " + ", ".join(
            f"{n} ({s / 2**20:.1f} MiB)" for n, s in sorted(files.items()))
                  + f"; total {total / 2**20:.1f} MiB."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- files

def save_layer(data_dir: Path, layer: int, payload: dict[str, Any]) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / layer_file(layer)
    torch.save(payload, path)
    return path


def write_manifest(data_dir: Path, manifest: dict[str, Any]) -> Path:
    path = data_dir / MANIFEST
    path.write_text(json.dumps(manifest, indent=2))
    return path


def read_manifest(data_dir: Path = DATA_DIR) -> dict[str, Any] | None:
    path = Path(data_dir) / MANIFEST
    return json.loads(path.read_text()) if path.exists() else None


def load_layer(layer: int, data_dir: Path = DATA_DIR) -> dict[str, Any]:
    return torch.load(Path(data_dir) / layer_file(layer), map_location="cpu", weights_only=True)


def load_captured_paged(layer: int, block_size: int = 16, *, kv_dtype: str = "fp16",
                        scale_granularity: str = "kv_head", device: str | torch.device = "cpu",
                        prompts: Sequence[int] | None = None, data_dir: Path = DATA_DIR,
                        seed: int = 0) -> tuple[PagedKVCache, torch.Tensor, dict[str, Any]]:
    """Captured layer -> (paged cache in our layout, q [batch, num_q_heads, head_dim] fp16,
    payload). ``prompts`` selects a subset by index (default: all)."""
    payload = load_layer(layer, data_dir)
    idx = list(range(len(payload["keys"]))) if prompts is None else list(prompts)
    keys = [payload["keys"][i].to(device) for i in idx]
    values = [payload["values"][i].to(device) for i in idx]
    q = torch.stack([payload["queries"][i] for i in idx]).to(device)
    cache = build_paged_kv_cache(keys, values, block_size, kv_dtype=kv_dtype,  # type: ignore[arg-type]
                                 scale_granularity=scale_granularity,  # type: ignore[arg-type]
                                 seed=seed, device=device)
    return cache, q, payload
