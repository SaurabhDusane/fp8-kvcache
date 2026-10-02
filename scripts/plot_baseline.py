#!/usr/bin/env python
"""Plot the baseline sweep from saved JSON: TTFT p50/p99, ITL p50/p99 and goodput vs rate,
one line per KV-cache dtype, error bars = min–max over repeats (marker = median).

Usage:
    python scripts/plot_baseline.py                  # latest sweep
    python scripts/plot_baseline.py --sweep-id 20261002-101500

Writes bench/results/summary/baseline_<trace>.png (one figure per trace).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `bench` importable

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from bench.common.results import load_results, results_root  # noqa: E402
from bench.serving.sweep import aggregate_runs  # noqa: E402

# Categorical slots 1-2 of the validated default palette (light surface #fcfcfb): pass the
# lightness, chroma, CVD (protan dE 24.7) and contrast checks. Color follows the entity.
SERIES_COLORS = {"auto": "#2a78d6", "fp8": "#eb6834"}
EXTRA_COLORS = ["#1baf7a", "#eda100"]  # slots 3-4, only if more dtypes are swept
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LABELS = {"auto": "auto (FP16 KV)", "fp8": "fp8 KV"}


def color_for(dtype: str, order: Sequence[str]) -> str:
    if dtype in SERIES_COLORS:
        return SERIES_COLORS[dtype]
    others = [d for d in order if d not in SERIES_COLORS]
    return EXTRA_COLORS[others.index(dtype) % len(EXTRA_COLORS)]


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)


def _errorbar(ax: Any, rows: list[dict[str, Any]], key: str, color: str, style: str,
              dodge: float = 1.0) -> None:
    pts = [r for r in rows if r[key] is not None]
    if not pts:
        return
    x = [r["rate"] * dodge for r in pts]  # small multiplicative offset per dtype (log x axis)
    y = [r[key] for r in pts]
    lo = [r[key] - r[f"{key}_min"] for r in pts]
    hi = [r[f"{key}_max"] - r[key] for r in pts]
    ax.errorbar(x, y, yerr=[lo, hi], color=color, linestyle=style, linewidth=1.5,
                marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.0,
                elinewidth=1.0, capsize=2.5)


def plot_trace(trace: str, rows: list[dict[str, Any]], out_path: Path,
               slo_ttft_ms: float | None = None, slo_itl_ms: float | None = None) -> Path:
    dtypes = sorted({r["kv_cache_dtype"] for r in rows}, key=lambda d: (d != "auto", d))
    unit = "sessions/s" if trace == "multi_turn" else "req/s"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=SURFACE)
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    # Label only the 2x and 5x minor ticks of each decade on log axes (avoids crowding).
    minor_25 = matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}" if v > 0 and f"{v:.0e}"[0] in "25" else "")
    panels = (("TTFT (ms)", ("ttft_p50_ms", "ttft_p99_ms"), True),
              ("ITL (ms)", ("itl_p50_ms", "itl_p99_ms"), True),
              ("Goodput (req/s meeting SLO)", ("goodput_rps",), False))
    for ax, (title, keys, logy) in zip(axes, panels):
        _style(ax)
        for i, d in enumerate(dtypes):
            dr = sorted((r for r in rows if r["kv_cache_dtype"] == d), key=lambda r: r["rate"])
            # Dodge dtypes by +-3% in x so overlapping series and error bars stay visible.
            dodge = 1.03 ** (i - (len(dtypes) - 1) / 2)
            for key in keys:
                _errorbar(ax, dr, key, color_for(d, dtypes),
                          "--" if key.endswith("p99_ms") else "-", dodge)
        ax.set_xscale("log", base=2)
        rates = sorted({r["rate"] for r in rows})
        ax.set_xticks(rates)
        ax.xaxis.set_major_formatter(plain)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(plain)
            ax.yaxis.set_minor_formatter(minor_25)
            ax.tick_params(axis="y", which="minor", labelsize=7, labelcolor=MUTED)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel(f"request rate ({unit})", color=INK_2, fontsize=9)
    for ax, slo, lab in ((axes[0], slo_ttft_ms, "SLO"), (axes[1], slo_itl_ms, "SLO (p90 per request)")):
        if slo:
            ax.axhline(slo, color=MUTED, linewidth=0.8, linestyle=":")
            ax.annotate(lab, xy=(0.01, slo), xycoords=("axes fraction", "data"), color=MUTED,
                        fontsize=7, va="bottom")
    handles = [Line2D([], [], color=color_for(d, dtypes), marker="o", linewidth=1.5,
                      label=LABELS.get(d, d)) for d in dtypes]
    handles += [Line2D([], [], color=INK_2, linestyle="-", label="p50"),
                Line2D([], [], color=INK_2, linestyle="--", label="p99")]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, fontsize=8,
               labelcolor=INK_2)
    fig.suptitle(f"{trace}: KV cache dtype vs load (median of repeats, bars = min–max)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def latest_sweep_id() -> str:
    sweeps = load_results("baseline_sweep_*")
    if sweeps.empty:
        raise SystemExit("no baseline_sweep results found; run scripts/run_baseline_sweep.py")
    return str(sweeps.iloc[-1]["config.sweep_id"])


def plot_sweep(sweep_id: str) -> list[Path]:
    runs = load_results("serving_*")
    recs = ([r for r in runs["_raw"] if r["config"].get("sweep_id") == sweep_id]
            if not runs.empty else [])
    if not recs:
        raise SystemExit(f"no serving runs saved for sweep {sweep_id}")
    rows = aggregate_runs(recs)
    slo_ttft = recs[0]["config"].get("slo_ttft_ms")
    slo_itl = recs[0]["config"].get("slo_itl_ms")
    out_dir = results_root() / "summary"
    paths = []
    for trace in sorted({r["trace"] for r in rows}):
        paths.append(plot_trace(trace, [r for r in rows if r["trace"] == trace],
                                out_dir / f"baseline_{trace}.png", slo_ttft, slo_itl))
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep-id", default=None, help="default: latest sweep")
    args = ap.parse_args(argv)
    sweep_id = args.sweep_id or latest_sweep_id()
    for p in plot_sweep(sweep_id):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
