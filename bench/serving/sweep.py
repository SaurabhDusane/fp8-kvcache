"""Baseline-sweep logic shared by scripts/run_baseline_sweep.py and scripts/plot_baseline.py.

Pure functions (no GPU, no server): vLLM log parsing, the rate plan that extends a rate list
until goodput collapses, request-count scaling, aggregation of saved runs over repeats, and the
baseline.md markdown. Everything is computed from saved result JSON (CLAUDE.md rule 7).
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from bench.common.gpu_monitor import flag_throttled

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# vLLM startup log lines (V1 engine; V0 equivalents where they differ). Last match wins.
_LOG_PATTERNS: dict[str, re.Pattern[str]] = {
    "kv_cache_tokens": re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens"),
    "max_concurrency": re.compile(
        r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)\s*x"),
    "kv_cache_memory_gib": re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB"),
    "gpu_blocks": re.compile(r"# GPU blocks:\s*([\d,]+)"),
    "vllm_version": re.compile(r"vLLM API server version\s+(\S+)"),
}
_LINE_KEYS = {"kv_cache_tokens": "GPU KV cache size", "max_concurrency": "Maximum concurrency"}


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_vllm_log(text: str) -> dict[str, Any]:
    """Extract KV-cache capacity facts from a vLLM server log. Missing values are None.

    ``lines`` keeps the exact matched lines so the summary can quote them verbatim.
    """
    text = strip_ansi(text)
    info: dict[str, Any] = {"kv_cache_tokens": None, "max_concurrency": None,
                            "max_concurrency_tokens_per_request": None,
                            "kv_cache_memory_gib": None, "gpu_blocks": None,
                            "vllm_version": None, "lines": {}}
    for key, pat in _LOG_PATTERNS.items():
        matches = list(pat.finditer(text))
        if not matches:
            continue
        m = matches[-1]
        if key == "max_concurrency":
            info["max_concurrency_tokens_per_request"] = int(_num(m.group(1)))
            info["max_concurrency"] = float(m.group(2))
        elif key in ("kv_cache_tokens", "gpu_blocks"):
            info[key] = int(_num(m.group(1)))
        elif key == "kv_cache_memory_gib":
            info[key] = float(m.group(1))
        else:
            info[key] = m.group(1)
        if key in _LINE_KEYS:
            start = text.rfind("\n", 0, m.start()) + 1
            end = text.find("\n", m.end())
            info["lines"][key] = text[start:end if end != -1 else None].strip()
    return info


def extract_startup_error(text: str, max_lines: int = 40) -> str:
    """The most useful part of a failed server log: the last traceback, else ERROR lines,
    else the tail."""
    lines = strip_ansi(text).rstrip().splitlines()
    if not lines:
        return "(empty log)"
    tb = [i for i, l in enumerate(lines) if l.lstrip().startswith("Traceback")]
    if tb:
        return "\n".join(lines[tb[-1]:][-max_lines:])
    errs = [l for l in lines if "ERROR" in l or "Error" in l]
    if errs:
        return "\n".join(errs[-max_lines:])
    return "\n".join(lines[-20:])


# --------------------------------------------------------------------------- rate plan

@dataclass
class RatePlan:
    """Walks a rate list in ascending order and extends it until goodput collapses.

    A point is *collapsed* when its median SLO attainment (fraction of requests meeting the
    SLO) is below ``collapse_attainment``. After the first collapsed point the plan runs
    ``points_after_collapse`` more points (to show the collapse is not noise) and stops.
    If the list ends without a collapse and ``extend`` is set, it keeps multiplying the last
    rate by ``extend_factor`` up to ``max_rate``. Every decision is logged in ``notes``.
    """

    base_rates: Sequence[float]
    collapse_attainment: float = 0.5
    points_after_collapse: int = 1
    extend: bool = True
    extend_factor: float = 2.0
    max_rate: float = 64.0
    results: list[tuple[float, float | None]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    _queue: list[float] = field(default_factory=list, init=False)
    _collapsed: int = field(default=0, init=False)
    _pending: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        rates = sorted(set(float(r) for r in self.base_rates))
        if not rates or rates[0] <= 0:
            raise ValueError("rates must be positive")
        self._queue = rates

    @property
    def collapsed_at(self) -> float | None:
        for rate, att in self.results:
            if att is None or att < self.collapse_attainment:
                return rate
        return None

    def next_rate(self) -> float | None:
        if self._pending is not None:
            raise RuntimeError("record() the previous rate before asking for the next")
        if self._collapsed > self.points_after_collapse:
            self.notes.append(f"stop: {self._collapsed} collapsed point(s)")
            return None
        if self._queue:
            self._pending = self._queue.pop(0)
            return self._pending
        if self._collapsed:
            self.notes.append("stop: rate list exhausted after collapse")
            return None
        if not self.extend:
            self.notes.append("stop: rate list exhausted, no collapse (extension disabled)")
            return None
        last = self.results[-1][0] if self.results else max(self.base_rates)
        nxt = last * self.extend_factor
        if nxt > self.max_rate:
            self.notes.append(f"stop: no collapse up to max rate {self.max_rate:g}")
            return None
        self.notes.append(f"extend: no collapse by {last:g}/s, trying {nxt:g}/s")
        self._pending = nxt
        return nxt

    def record(self, rate: float, attainment_median: float | None) -> None:
        if self._pending is None or not math.isclose(rate, self._pending):
            raise RuntimeError(f"record({rate}) does not match the pending rate {self._pending}")
        self._pending = None
        self.results.append((rate, attainment_median))
        if attainment_median is None or attainment_median < self.collapse_attainment:
            self._collapsed += 1
            self.notes.append(f"collapsed at {rate:g}/s (SLO attainment {attainment_median})")


def num_requests_for(rate: float, trace: str, duration_s: float, min_requests: int,
                     max_requests: int, turns_per_session: float = 4.5) -> int:
    """Requests for ~``duration_s`` of arrivals. For multi_turn the rate is sessions/s, and a
    session averages ``turns_per_session`` requests (3-6 turns)."""
    per_s = rate * (turns_per_session if trace == "multi_turn" else 1.0)
    return int(min(max(round(per_s * duration_s), min_requests), max_requests))


def seed_for(base_seed: int, rate: float, repeat: int) -> int:
    """Same (rate, repeat) -> same seed in every config (comparable workloads); different
    rates/repeats -> different prompts, so the prefix cache can't carry over between runs."""
    return base_seed + int(round(rate * 1000)) * 10 + repeat


def parse_rate_spec(specs: Iterable[str], traces: Sequence[str],
                    defaults: dict[str, list[float]]) -> dict[str, list[float]]:
    """``["chat=1,2,4", "multi_turn=0.5,1"]`` -> per-trace rate lists (others use defaults)."""
    out = {t: list(defaults[t]) for t in traces}
    for spec in specs:
        name, _, vals = spec.partition("=")
        if name not in out or not vals:
            raise ValueError(f"bad --rates {spec!r}; expected TRACE=r1,r2 with TRACE in {list(traces)}")
        out[name] = [float(v) for v in vals.split(",") if v]
    return out


# --------------------------------------------------------------------------- aggregation

def _stats(vals: Sequence[float | None]) -> tuple[float | None, float | None, float | None]:
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, None
    return statistics.median(v), min(v), max(v)


def aggregate_runs(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Saved serving-run records (full JSON dicts) -> one row per (trace, kv dtype, rate).

    Each metric gets median/min/max over repeats. A run is throttled if its median SM clock is
    >15% below the highest SM clock of the whole sweep (CLAUDE.md rule 4).
    """
    sm_med = [((r.get("gpu") or {}).get("summary") or {}).get("sm_clock_mhz", {}).get("median")
              for r in records]
    sm_max = [((r.get("gpu") or {}).get("summary") or {}).get("sm_clock_mhz", {}).get("max")
              for r in records]
    throttled = flag_throttled(sm_med, sm_max)
    groups: dict[tuple[str, str, float], list[tuple[dict[str, Any], bool | None]]] = {}
    for rec, thr in zip(records, throttled):
        c = rec["config"]
        key = (c["trace"], c["kv_cache_dtype"], float(c["rate"]))
        groups.setdefault(key, []).append((rec, thr))

    rows = []
    for (trace, dtype, rate), items in sorted(groups.items()):
        ms = [rec["metrics"] for rec, _ in items]

        def col(fn: Any) -> tuple[float | None, float | None, float | None]:
            return _stats([fn(m) for m in ms])

        row: dict[str, Any] = {"trace": trace, "kv_cache_dtype": dtype, "rate": rate,
                               "repeats": len(items),
                               "num_requests": items[0][0]["metrics"].get("num_requests"),
                               "failed_requests": sum(m.get("failed", 0) for m in ms),
                               "throttled_repeats": sum(1 for _, t in items if t),
                               "trace_source": items[0][0]["config"].get("trace_stats", {}).get("source")}
        for name, fn in {
            "goodput_rps": lambda m: m.get("goodput_rps"),
            "slo_attainment": lambda m: m.get("slo_attainment"),
            "request_throughput_rps": lambda m: m.get("request_throughput_rps"),
            "output_throughput_tps": lambda m: m.get("output_throughput_tps"),
            "ttft_p50_ms": lambda m: m["ttft_ms"].get("median"),
            "ttft_p99_ms": lambda m: m["ttft_ms"].get("p99"),
            "itl_p50_ms": lambda m: m["itl_ms"].get("median"),
            "itl_p99_ms": lambda m: m["itl_ms"].get("p99"),
            "tpot_p50_ms": lambda m: m["tpot_ms"].get("median"),
        }.items():
            med, lo, hi = col(fn)
            row[name], row[f"{name}_min"], row[f"{name}_max"] = med, lo, hi
        rows.append(row)
    return rows


def saturation(rows: Sequence[dict[str, Any]], collapse_attainment: float) -> list[dict[str, Any]]:
    """Per (trace, dtype): peak median goodput and its rate; first rate with median SLO
    attainment below the threshold; highest rate run."""
    out = []
    keys = sorted({(r["trace"], r["kv_cache_dtype"]) for r in rows})
    for trace, dtype in keys:
        pts = sorted((r for r in rows if r["trace"] == trace and r["kv_cache_dtype"] == dtype),
                     key=lambda r: r["rate"])
        good = [r for r in pts if r["goodput_rps"] is not None]
        peak = max(good, key=lambda r: r["goodput_rps"]) if good else None
        collapse = next((r["rate"] for r in pts if r["slo_attainment"] is None
                         or r["slo_attainment"] < collapse_attainment), None)
        out.append({"trace": trace, "kv_cache_dtype": dtype,
                    "peak_goodput_rps": peak["goodput_rps"] if peak else None,
                    "peak_goodput_rate": peak["rate"] if peak else None,
                    "first_collapsed_rate": collapse, "max_rate_run": pts[-1]["rate"]})
    return out


# --------------------------------------------------------------------------- markdown

def _f(v: Any, nd: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def _g(v: float | None) -> str:
    return "n/a" if v is None else f"{v:g}"


def _spread(row: dict[str, Any], key: str, nd: int = 1) -> str:
    med, lo, hi = row[key], row[f"{key}_min"], row[f"{key}_max"]
    if med is None:
        return "n/a"
    if row["repeats"] < 2:
        return _f(med, nd)
    return f"{_f(med, nd)} ({_f(lo, nd)}–{_f(hi, nd)})"


def summary_markdown(sweep: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    """baseline.md from the saved sweep record and aggregated run rows."""
    cfg, met = sweep["config"], sweep["metrics"]
    meta = sweep.get("metadata", {})
    pk = meta.get("packages", {})
    gpu = meta.get("gpu", {})
    git = meta.get("git", {})
    lines = [
        "# Baseline serving sweep: KV cache dtype auto (FP16) vs fp8",
        "",
        f"- Sweep `{cfg['sweep_id']}` · model `{cfg['model']}` · git `{str(git.get('sha'))[:10]}` "
        f"(dirty: {git.get('dirty')}) · {gpu.get('name')} · driver {gpu.get('driver')} · "
        f"vLLM {pk.get('vllm')}",
        f"- Server: `--max-model-len {cfg['max_model_len']} --gpu-memory-utilization "
        f"{cfg['gpu_memory_utilization']}`" + (f" + `{' '.join(cfg['server_args'])}`"
                                                if cfg.get("server_args") else ""),
        f"- Load: {cfg['repeats']} repeats per point, ~{cfg['duration_s']:g} s of arrivals per run "
        f"({cfg['min_requests']}–{cfg['max_requests']} requests), Poisson arrivals "
        f"(multi_turn: session rate), SLO TTFT ≤ {cfg['slo_ttft_ms']:g} ms and p90 ITL ≤ "
        f"{cfg['slo_itl_ms']:g} ms; collapse = median SLO attainment < "
        f"{100 * cfg['collapse_attainment']:g}%",
        "- Cells: median over repeats (min–max). Throttled = run's median SM clock >15% below "
        "the sweep max.",
        "",
        "## Server configs",
        "",
        "| kv-cache-dtype | status | GPU KV cache (tokens) | max concurrency (tokens/req) | "
        "KV cache memory (GiB) | startup (s) |",
        "|---|---|---|---|---|---|",
    ]
    for s in met["servers"]:
        info = s.get("log_info") or {}
        mc = info.get("max_concurrency")
        mc_s = "n/a" if mc is None else f"{mc:g}x ({info.get('max_concurrency_tokens_per_request')})"
        kv = info.get("kv_cache_tokens")
        lines.append(f"| {s['kv_cache_dtype']} | {s['status']} | "
                     f"{'n/a' if kv is None else f'{kv:,}'} | {mc_s} | "
                     f"{_f(info.get('kv_cache_memory_gib'), 2)} | {_f(s.get('startup_s'))} |")
    for s in met["servers"]:
        if s["status"] != "ok":
            lines += ["", f"**{s['kv_cache_dtype']} failed:** see `{s.get('log_path')}`", "",
                      "```", s.get("error", "").strip(), "```"]
        for k, line in ((s.get("log_info") or {}).get("lines") or {}).items():
            lines.append(f"- {s['kv_cache_dtype']} log: `{line}`")

    sat = saturation(rows, cfg["collapse_attainment"])
    if sat:
        lines += ["", "## Saturation", "",
                  "| trace | kv dtype | peak goodput req/s (at rate) | first collapsed rate | "
                  "highest rate run |", "|---|---|---|---|---|"]
        for s in sat:
            fc = s["first_collapsed_rate"]
            lines.append(f"| {s['trace']} | {s['kv_cache_dtype']} | {_f(s['peak_goodput_rps'], 2)} "
                         f"({_g(s['peak_goodput_rate'])}) | {'none' if fc is None else f'{fc:g}'} | "
                         f"{s['max_rate_run']:g} |")

    for trace in sorted({r["trace"] for r in rows}):
        tr = [r for r in rows if r["trace"] == trace]
        src = sorted({str(r["trace_source"]) for r in tr})
        unit = "sessions/s" if trace == "multi_turn" else "req/s"
        lines += ["", f"## {trace} (rate in {unit}; prompts: {', '.join(src)})", "",
                  "| rate | kv dtype | goodput req/s | SLO met % | TTFT p50 ms | TTFT p99 ms | "
                  "ITL p50 ms | ITL p99 ms | out tok/s | reqs | failed | throttled |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(tr, key=lambda r: (r["rate"], r["kv_cache_dtype"])):
            att = r["slo_attainment"]
            lines.append(
                f"| {r['rate']:g} | {r['kv_cache_dtype']} | {_spread(r, 'goodput_rps', 2)} | "
                f"{_f(None if att is None else 100 * att)} | {_spread(r, 'ttft_p50_ms')} | "
                f"{_spread(r, 'ttft_p99_ms')} | {_spread(r, 'itl_p50_ms')} | "
                f"{_spread(r, 'itl_p99_ms')} | {_spread(r, 'output_throughput_tps', 0)} | "
                f"{r['num_requests']} | {r['failed_requests']} | "
                f"{r['throttled_repeats']}/{r['repeats']} |")
    notes = met.get("plan_notes") or {}
    if notes:
        lines += ["", "## Rate plan decisions", ""]
        for key, ns in notes.items():
            lines.append(f"- {key}: " + "; ".join(ns))
    return "\n".join(lines) + "\n"
