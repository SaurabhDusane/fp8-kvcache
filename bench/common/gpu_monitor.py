"""Background GPU-state sampler (SM/mem clock, power, temperature, utilization, memory).

Usage:
    with GpuMonitor() as mon:
        run_benchmark()
    result = mon.result          # MonitorResult: .samples, .summary, .throttled
    print(format_summary(result))

Sampling: a daemon thread calls nvidia-smi every ``interval_s`` (default 200 ms), scheduled
against fixed deadlines. If one call takes longer than the interval, the next one starts right
away, so the achieved interval is max(interval, nvidia-smi latency); the summary reports it.

Throttle rule (CLAUDE.md, benchmarking rule 4): a run is throttled if its median SM clock is
more than 15% below the session max. Within one monitor the "session max" defaults to the
highest SM clock seen in that run; pass ``session_max_sm_mhz`` (or use ``flag_throttled``) to
compare several runs from one session.

If nvidia-smi is missing, the monitor warns and returns empty samples; it never raises.
"""

from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

# (nvidia-smi query field, our column name). Order matches the query string.
QUERY_FIELDS: tuple[tuple[str, str], ...] = (
    ("clocks.sm", "sm_clock_mhz"),
    ("clocks.mem", "mem_clock_mhz"),
    ("power.draw", "power_w"),
    ("power.limit", "power_limit_w"),
    ("temperature.gpu", "temp_c"),
    ("utilization.gpu", "util_pct"),
    ("memory.used", "mem_used_mib"),
)
FIELDS: tuple[str, ...] = tuple(name for _, name in QUERY_FIELDS)
THROTTLE_THRESHOLD = 0.15


def nvidia_smi_command(device: int = 0) -> list[str]:
    query = ",".join(q for q, _ in QUERY_FIELDS)
    return ["nvidia-smi", "-i", str(device), f"--query-gpu={query}", "--format=csv,noheader,nounits"]


def parse_line(line: str) -> dict[str, float | None]:
    """Parse one nvidia-smi CSV line. Unsupported values ("[N/A]", "N/A", "") become None."""
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) != len(FIELDS):
        raise ValueError(f"expected {len(FIELDS)} fields, got {len(parts)}: {line!r}")
    values: dict[str, float | None] = {}
    for name, raw in zip(FIELDS, parts):
        try:
            values[name] = float(raw)
        except ValueError:
            values[name] = None
    return values


def summarize(samples: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    """min/median/max (and count of valid values) per field, ignoring None."""
    out: dict[str, dict[str, float | int | None]] = {}
    for name in FIELDS:
        vals = [s[name] for s in samples if s.get(name) is not None]
        out[name] = {
            "min": min(vals) if vals else None,
            "median": statistics.median(vals) if vals else None,
            "max": max(vals) if vals else None,
            "n": len(vals),
        }
    return out


def is_throttled(median_sm_mhz: float | None, session_max_sm_mhz: float | None,
                 threshold: float = THROTTLE_THRESHOLD) -> bool | None:
    """True if median is more than `threshold` below the session max; None if unknown."""
    if median_sm_mhz is None or not session_max_sm_mhz:
        return None
    return median_sm_mhz < (1.0 - threshold) * session_max_sm_mhz


@dataclass
class MonitorResult:
    samples: list[dict[str, Any]] = field(default_factory=list)  # each has "t" (s since start)
    interval_s: float = 0.2
    duration_s: float = 0.0
    errors: int = 0
    warnings: list[str] = field(default_factory=list)
    session_max_sm_mhz: float | None = None  # override for cross-run comparison

    @property
    def summary(self) -> dict[str, dict[str, float | int | None]]:
        return summarize(self.samples)

    @property
    def achieved_interval_s(self) -> float | None:
        """Median gap between samples (robust to the extra sample taken at stop())."""
        if len(self.samples) < 2:
            return None
        ts = [s["t"] for s in self.samples]
        return statistics.median(b - a for a, b in zip(ts, ts[1:]))

    def throttle(self, session_max_sm_mhz: float | None = None) -> dict[str, Any]:
        sm = self.summary["sm_clock_mhz"]
        ref = session_max_sm_mhz or self.session_max_sm_mhz or sm["max"]
        return {
            "throttled": is_throttled(sm["median"], ref),
            "median_sm_mhz": sm["median"],
            "session_max_sm_mhz": ref,
            "threshold": THROTTLE_THRESHOLD,
        }

    @property
    def throttled(self) -> bool | None:
        return self.throttle()["throttled"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "summary": self.summary,
            "throttle": self.throttle(),
            "interval_s": self.interval_s,
            "achieved_interval_s": self.achieved_interval_s,
            "duration_s": self.duration_s,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class GpuMonitor:
    """Context manager that samples GPU state in a background thread.

    ``query_fn`` returns one raw nvidia-smi CSV line per call; it is injectable for tests.
    """

    def __init__(self, interval_s: float = 0.2, device: int = 0,
                 query_fn: Callable[[], str] | None = None) -> None:
        self.interval_s = interval_s
        self.device = device
        self._query_fn = query_fn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self.result = MonitorResult(interval_s=interval_s)

    def _default_query(self) -> str:
        out = subprocess.run(nvidia_smi_command(self.device), capture_output=True, text=True,
                             timeout=5, check=True)
        return out.stdout.strip().splitlines()[0]

    def _sample_once(self) -> None:
        t = time.perf_counter() - self._t0
        try:
            values = parse_line(self._query_fn())  # type: ignore[misc]
        except Exception as exc:
            self.result.errors += 1
            if self.result.errors == 1:  # report the first failure, don't spam
                msg = f"gpu_monitor: sample failed ({type(exc).__name__}: {exc})"
                self.result.warnings.append(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            return
        self.result.samples.append({"t": round(t, 4), **values})

    def _loop(self) -> None:
        next_t = time.perf_counter()
        while True:
            self._sample_once()
            next_t += self.interval_s
            delay = next_t - time.perf_counter()
            if delay < 0:  # sampling slower than the interval: don't try to catch up
                next_t = time.perf_counter()
                delay = 0.0
            if self._stop.wait(delay):
                break

    def start(self) -> "GpuMonitor":
        self.result = MonitorResult(interval_s=self.interval_s)
        self._stop.clear()
        if self._query_fn is None:
            if shutil.which("nvidia-smi") is None:
                msg = "gpu_monitor: nvidia-smi not found; returning empty GPU samples"
                self.result.warnings.append(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                return self
            self._query_fn = self._default_query
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> MonitorResult:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=10)
            self._thread = None
            self._sample_once()  # final sample at the end of the measured region
            self.result.duration_s = round(time.perf_counter() - self._t0, 4)
        return self.result

    def __enter__(self) -> "GpuMonitor":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def flag_throttled(medians_sm_mhz: Iterable[float | None],
                   maxes_sm_mhz: Iterable[float | None] | None = None,
                   threshold: float = THROTTLE_THRESHOLD) -> list[bool | None]:
    """Flag each run of a session. Session max = highest per-run max (or median if no maxes)."""
    medians = list(medians_sm_mhz)
    maxes = list(maxes_sm_mhz) if maxes_sm_mhz is not None else medians
    known = [m for m in maxes if m is not None]
    session_max = max(known) if known else None
    return [is_throttled(m, session_max, threshold) for m in medians]


def format_summary(result: MonitorResult) -> str:
    """Compact, paste-friendly text table."""
    if not result.samples:
        lines = ["GPU monitor: no samples"]
        lines += [f"  warning: {w}" for w in result.warnings]
        return "\n".join(lines)
    ach = result.achieved_interval_s
    lines = [
        f"GPU monitor: {len(result.samples)} samples over {result.duration_s:.2f} s "
        f"(target interval {result.interval_s * 1000:.0f} ms, achieved "
        f"{'n/a' if ach is None else f'{ach * 1000:.0f} ms'}, errors {result.errors})",
        f"  {'field':<15}{'min':>10}{'median':>10}{'max':>10}",
    ]

    def fmt(v: float | int | None) -> str:
        return "n/a" if v is None else f"{v:.1f}"

    for name, s in result.summary.items():
        lines.append(f"  {name:<15}{fmt(s['min']):>10}{fmt(s['median']):>10}{fmt(s['max']):>10}")
    th = result.throttle()
    lines.append(
        f"  throttled: {th['throttled']} (median SM {fmt(th['median_sm_mhz'])} MHz vs session max "
        f"{fmt(th['session_max_sm_mhz'])} MHz, threshold {th['threshold']:.0%})"
    )
    lines += [f"  warning: {w}" for w in result.warnings]
    return "\n".join(lines)
