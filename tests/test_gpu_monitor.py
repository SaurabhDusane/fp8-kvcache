"""GPU monitor: parsing, summary, throttle logic, and the sampling thread with mocked nvidia-smi."""

from __future__ import annotations

import itertools
import time

import pytest

from bench.common import gpu_monitor as gm
from bench.common.gpu_monitor import (
    FIELDS, GpuMonitor, MonitorResult, flag_throttled, format_summary, is_throttled,
    nvidia_smi_command, parse_line, summarize,
)

# Realistic lines as printed by
#   nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,
#              utilization.gpu,memory.used --format=csv,noheader,nounits
LINES = [
    "2610, 9001, 141.52, 150.00, 71, 100, 4321",
    "2580, 9001, 148.07, 150.00, 74, 100, 4321",
    "2100, 9001, 149.90, 150.00, 83, 99, 4325",
]


def test_command_matches_requested_query() -> None:
    cmd = nvidia_smi_command(device=0)
    assert cmd[0] == "nvidia-smi"
    assert ("--query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,"
            "utilization.gpu,memory.used") in cmd
    assert "--format=csv,noheader,nounits" in cmd


def test_parse_line() -> None:
    s = parse_line(LINES[0])
    assert s == {"sm_clock_mhz": 2610.0, "mem_clock_mhz": 9001.0, "power_w": 141.52,
                 "power_limit_w": 150.0, "temp_c": 71.0, "util_pct": 100.0, "mem_used_mib": 4321.0}


def test_parse_line_not_available_fields() -> None:
    # Laptop GPUs under WSL often report [N/A] for power fields.
    s = parse_line("1800, 8000, [N/A], [N/A], 60, 5, 1000")
    assert s["power_w"] is None and s["power_limit_w"] is None and s["sm_clock_mhz"] == 1800.0


def test_parse_line_wrong_field_count() -> None:
    with pytest.raises(ValueError):
        parse_line("1800, 8000")


def test_summarize_ignores_none() -> None:
    samples = [parse_line(l) for l in LINES] + [parse_line("[N/A], 9001, [N/A], 150, 80, 100, 1")]
    s = summarize(samples)
    assert s["sm_clock_mhz"] == {"min": 2100.0, "median": 2580.0, "max": 2610.0, "n": 3}
    assert s["power_w"]["n"] == 3 and s["temp_c"]["n"] == 4
    assert summarize([])["sm_clock_mhz"] == {"min": None, "median": None, "max": None, "n": 0}


@pytest.mark.parametrize("median, session_max, expected", [
    (2610, 2610, False),
    (2219, 2610, False),   # 2219 >= 0.85 * 2610 = 2218.5
    (2218, 2610, True),    # more than 15% below
    (1000, 2610, True),
    (None, 2610, None),
    (2000, None, None),
    (2000, 0, None),
])
def test_is_throttled(median, session_max, expected) -> None:
    assert is_throttled(median, session_max) is expected


def test_flag_throttled_uses_session_max() -> None:
    # Session max is 2610 (from run 0's max); run 2's median is >15% below it.
    flags = flag_throttled([2580, 2400, 2000], maxes_sm_mhz=[2610, 2500, 2300])
    assert flags == [False, False, True]
    assert flag_throttled([None, 2000]) == [None, False]
    assert flag_throttled([]) == []


def test_result_throttle_override() -> None:
    r = MonitorResult(samples=[{"t": 0.0, **parse_line(l)} for l in LINES])
    assert r.throttled is False  # median 2580 vs own max 2610
    assert r.throttle(session_max_sm_mhz=3100)["throttled"] is True  # 2580 < 2635


def test_monitor_samples_with_mocked_nvidia_smi() -> None:
    feed = itertools.cycle(LINES)
    with GpuMonitor(interval_s=0.01, query_fn=lambda: next(feed)) as mon:
        time.sleep(0.15)
    r = mon.result
    assert len(r.samples) >= 5
    assert all(set(FIELDS) <= set(s) for s in r.samples)
    ts = [s["t"] for s in r.samples]
    assert ts == sorted(ts) and r.duration_s >= 0.15
    assert r.summary["sm_clock_mhz"]["max"] == 2610.0
    assert r.errors == 0 and r.throttled in (True, False)
    d = r.to_dict()
    assert set(d) >= {"samples", "summary", "throttle", "achieved_interval_s", "warnings"}
    assert "throttled:" in format_summary(r)


def test_monitor_survives_query_errors() -> None:
    calls = itertools.count()

    def flaky() -> str:
        if next(calls) % 2:
            raise RuntimeError("nvidia-smi hiccup")
        return LINES[0]

    with pytest.warns(RuntimeWarning, match="sample failed"):
        with GpuMonitor(interval_s=0.01, query_fn=flaky) as mon:
            time.sleep(0.08)
    assert mon.result.errors >= 1 and len(mon.result.samples) >= 1
    assert len(mon.result.warnings) == 1  # first failure reported once


def test_monitor_without_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gm.shutil, "which", lambda name: None)
    with pytest.warns(RuntimeWarning, match="nvidia-smi not found"):
        with GpuMonitor(interval_s=0.01) as mon:
            time.sleep(0.02)
    r = mon.result
    assert r.samples == [] and r.throttled is None
    assert "no samples" in format_summary(r)


def test_monitor_real_subprocess_path_with_fake_nvidia_smi(
        tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A fake `nvidia-smi` executable on PATH exercises the default subprocess query, including
    # the -i/--query-gpu/--format arguments being passed through.
    fake = tmp_path / "nvidia-smi"
    fake.write_text('#!/bin/sh\n'
                    'case "$*" in *"--query-gpu=clocks.sm,clocks.mem"*"nounits"*) ;; '
                    '*) echo "bad args: $*" >&2; exit 2;; esac\n'
                    'echo "2400, 9001, 120.5, 150.00, 70, 97, 2048"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    with GpuMonitor(interval_s=0.02) as mon:
        time.sleep(0.1)
    r = mon.result
    assert r.errors == 0 and len(r.samples) >= 2
    assert r.summary["sm_clock_mhz"]["median"] == 2400.0
    assert r.summary["util_pct"]["max"] == 97.0


def test_window_median_and_elapsed() -> None:
    r = MonitorResult(samples=[{"t": 0.0, "sm_clock_mhz": 2600.0}, {"t": 0.2, "sm_clock_mhz": 2000.0},
                               {"t": 0.4, "sm_clock_mhz": 1800.0}, {"t": 0.6, "sm_clock_mhz": None}])
    assert r.window_median("sm_clock_mhz", 0.1, 0.5) == 1900.0
    assert r.window_median("sm_clock_mhz", 0.55, 1.0) is None
    with GpuMonitor(interval_s=0.01, query_fn=lambda: LINES[0]) as mon:
        time.sleep(0.03)
        assert 0.03 <= mon.elapsed() < 5
