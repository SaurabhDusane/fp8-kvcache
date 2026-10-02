"""Baseline sweep: vLLM log parsing, rate plan, aggregation/summary, server lifecycle with a fake
`vllm` executable, the full sweep end to end on CPU, and plotting from fixtures."""

from __future__ import annotations

import json
import os
import socket
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from bench.common.results import save_result
from bench.serving.sweep import (
    RatePlan, aggregate_runs, extract_startup_error, num_requests_for, parse_rate_spec,
    parse_vllm_log, saturation, seed_for, summary_markdown,
)
from bench.serving.vllm_server import ServerStartError, VllmServer, build_vllm_cmd

REPO = Path(__file__).resolve().parents[1]

# Representative vLLM V1 startup log (ANSI colors as vLLM prints them to a terminal/file).
V1_LOG = (
    "INFO 10-02 10:00:01 [api_server.py:1880] vLLM API server version 0.11.0\n"
    "INFO 10-02 10:00:30 [gpu_worker.py:276] Available KV cache memory: 7.53 GiB\n"
    "\x1b[1;36m(EngineCore_DP0 pid=123)\x1b[0;0m INFO 10-02 10:00:31 [kv_cache_utils.py:1087] "
    "GPU KV cache size: 282,048 tokens\n"
    "\x1b[1;36m(EngineCore_DP0 pid=123)\x1b[0;0m INFO 10-02 10:00:31 [kv_cache_utils.py:1091] "
    "Maximum concurrency for 4,096 tokens per request: 68.86x\n"
    "INFO 10-02 10:00:45 [api_server.py:1950] Starting vLLM API server 0 on http://0.0.0.0:8000\n"
)


# --------------------------------------------------------------------------- log parsing

def test_parse_vllm_v1_log() -> None:
    info = parse_vllm_log(V1_LOG)
    assert info["kv_cache_tokens"] == 282048
    assert info["max_concurrency"] == 68.86
    assert info["max_concurrency_tokens_per_request"] == 4096
    assert info["kv_cache_memory_gib"] == 7.53
    assert info["vllm_version"] == "0.11.0"
    assert info["lines"]["kv_cache_tokens"].endswith("GPU KV cache size: 282,048 tokens")
    assert "\x1b" not in info["lines"]["max_concurrency"]


def test_parse_vllm_v0_style_and_last_match_wins() -> None:
    log = ("INFO executor_base.py:110] # GPU blocks: 17,628, # CPU blocks: 9,362\n"
           "INFO Maximum concurrency for 4096 tokens per request: 68.86x\n"
           "INFO Maximum concurrency for 4096 tokens per request: 70.00x\n")
    info = parse_vllm_log(log)
    assert info["gpu_blocks"] == 17628 and info["max_concurrency"] == 70.0
    assert info["kv_cache_tokens"] is None


def test_parse_empty_log() -> None:
    info = parse_vllm_log("")
    assert info["kv_cache_tokens"] is None and info["max_concurrency"] is None and info["lines"] == {}


def test_extract_startup_error() -> None:
    tb_log = "INFO starting\nTraceback (most recent call last):\n  File x\nValueError: No available memory for the cache blocks.\n"
    assert extract_startup_error(tb_log).startswith("Traceback")
    assert "No available memory" in extract_startup_error(tb_log)
    err_log = "INFO a\nERROR 10-02 engine core failed: CUDA out of memory\nINFO b\n"
    assert extract_startup_error(err_log) == "ERROR 10-02 engine core failed: CUDA out of memory"
    assert extract_startup_error("just\nsome\nlines") == "just\nsome\nlines"
    assert extract_startup_error("") == "(empty log)"


# --------------------------------------------------------------------------- rate plan

def _drive(plan: RatePlan, attainment) -> list[float]:
    seen = []
    while (r := plan.next_rate()) is not None:
        seen.append(r)
        plan.record(r, attainment(r))
    return seen


def test_rate_plan_stops_one_point_after_collapse() -> None:
    plan = RatePlan([4, 1, 2, 8, 16, 32])
    seen = _drive(plan, lambda r: 0.95 if r < 8 else 0.1)
    assert seen == [1, 2, 4, 8, 16]  # sorted; 8 collapses, 16 confirms, 32 skipped
    assert plan.collapsed_at == 8 and any("stop" in n for n in plan.notes)


def test_rate_plan_extends_until_collapse() -> None:
    plan = RatePlan([1, 2], points_after_collapse=0)
    seen = _drive(plan, lambda r: 1.0 if r < 16 else 0.2)
    assert seen == [1, 2, 4, 8, 16]
    assert any("extend" in n for n in plan.notes)


def test_rate_plan_max_rate_and_no_extend() -> None:
    plan = RatePlan([1, 2], max_rate=6)
    assert _drive(plan, lambda r: 1.0) == [1, 2, 4]
    assert "no collapse up to max rate 6" in plan.notes[-1]
    plan2 = RatePlan([1, 2], extend=False)
    assert _drive(plan2, lambda r: 1.0) == [1, 2]
    assert "extension disabled" in plan2.notes[-1]


def test_rate_plan_none_attainment_counts_as_collapse_and_misuse_errors() -> None:
    plan = RatePlan([1, 2, 4], points_after_collapse=0)
    assert _drive(plan, lambda r: None) == [1]
    p = RatePlan([1])
    p.next_rate()
    with pytest.raises(RuntimeError):
        p.next_rate()
    with pytest.raises(RuntimeError):
        p.record(5, 1.0)
    with pytest.raises(ValueError):
        RatePlan([0, 1])


def test_request_count_seed_and_rate_spec() -> None:
    assert num_requests_for(2, "chat", 45, 30, 300) == 90
    assert num_requests_for(0.25, "chat", 45, 30, 300) == 30
    assert num_requests_for(16, "chat", 45, 30, 300) == 300
    assert num_requests_for(1, "multi_turn", 40, 10, 500) == 180
    assert seed_for(0, 2.0, 1) == seed_for(0, 2.0, 1) != seed_for(0, 2.0, 2)
    assert seed_for(0, 0.5, 0) != seed_for(0, 1.0, 0)
    defaults = {"chat": [1.0], "multi_turn": [0.5]}
    assert parse_rate_spec(["chat=1,3.5"], ["chat", "multi_turn"], defaults) == {
        "chat": [1.0, 3.5], "multi_turn": [0.5]}
    with pytest.raises(ValueError):
        parse_rate_spec(["nope=1"], ["chat"], {"chat": [1.0]})


# --------------------------------------------------------------------------- aggregation

def _run_record(dtype: str, trace: str, rate: float, rep: int, goodput: float, att: float,
                ttft50: float, sm_med: float = 2500.0, sm_max: float = 2600.0) -> dict:
    return {
        "config": {"sweep_id": "S1", "kv_cache_dtype": dtype, "trace": trace, "rate": rate,
                   "repeat": rep, "trace_stats": {"source": "sharegpt"},
                   "slo_ttft_ms": 500.0, "slo_itl_ms": 50.0},
        "metrics": {"goodput_rps": goodput, "slo_attainment": att, "request_throughput_rps": rate,
                    "output_throughput_tps": 100 * rate, "num_requests": 40, "failed": 0,
                    "ttft_ms": {"median": ttft50, "p99": 4 * ttft50},
                    "itl_ms": {"median": 12.0 + rep, "p99": 30.0 + rep},
                    "tpot_ms": {"median": 13.0}},
        "gpu": {"summary": {"sm_clock_mhz": {"median": sm_med, "max": sm_max}}},
    }


def _fixture_runs() -> list[dict]:
    recs = []
    for dtype, k in (("auto", 1.0), ("fp8", 1.5)):
        for rate in (1.0, 2.0, 4.0):
            for rep in range(3):
                att = 1.0 if rate * 1.0 / k < 3 else 0.3
                recs.append(_run_record(dtype, "chat", rate, rep, goodput=rate * att + 0.01 * rep,
                                        att=att, ttft50=50 * rate + rep,
                                        sm_med=2000.0 if (dtype, rate, rep) == ("fp8", 4.0, 2) else 2500.0))
    return recs


def test_aggregate_runs() -> None:
    rows = aggregate_runs(_fixture_runs())
    assert len(rows) == 6
    r = next(x for x in rows if x["kv_cache_dtype"] == "auto" and x["rate"] == 2.0)
    assert r["repeats"] == 3
    assert r["goodput_rps"] == pytest.approx(2.01) and r["goodput_rps_min"] == pytest.approx(2.0)
    assert r["goodput_rps_max"] == pytest.approx(2.02)
    assert r["ttft_p50_ms"] == 101 and r["ttft_p50_ms_min"] == 100 and r["ttft_p50_ms_max"] == 102
    assert r["itl_p99_ms"] == 31.0 and r["trace_source"] == "sharegpt"
    hot = next(x for x in rows if x["kv_cache_dtype"] == "fp8" and x["rate"] == 4.0)
    assert hot["throttled_repeats"] == 1  # 2000 < 0.85 * 2600


def test_saturation() -> None:
    sat = {(s["trace"], s["kv_cache_dtype"]): s for s in saturation(aggregate_runs(_fixture_runs()), 0.5)}
    assert sat[("chat", "auto")]["first_collapsed_rate"] == 4.0
    assert sat[("chat", "auto")]["peak_goodput_rate"] == 2.0
    assert sat[("chat", "fp8")]["first_collapsed_rate"] is None  # 4/1.5 < 3
    assert sat[("chat", "fp8")]["peak_goodput_rate"] == 4.0


def _sweep_record(servers: list[dict]) -> dict:
    return {"config": {"sweep_id": "S1", "model": "Qwen/Qwen2.5-1.5B-Instruct", "max_model_len": 4096,
                       "gpu_memory_utilization": 0.85, "server_args": [], "repeats": 3,
                       "duration_s": 45.0, "min_requests": 30, "max_requests": 300,
                       "slo_ttft_ms": 500.0, "slo_itl_ms": 50.0, "collapse_attainment": 0.5},
            "metrics": {"servers": servers, "plan_notes": {"auto/chat": ["collapsed at 4/s"]}},
            "metadata": {"git": {"sha": "abc1234567890", "dirty": False},
                         "gpu": {"name": "FakeGPU", "driver": "1"}, "packages": {"vllm": "0.11.0"}}}


def test_summary_markdown() -> None:
    servers = [{"kv_cache_dtype": "auto", "status": "ok", "startup_s": 61.2,
                "log_info": parse_vllm_log(V1_LOG)},
               {"kv_cache_dtype": "fp8", "status": "failed_to_start", "log_path": "x.log",
                "error": "server exited with code 1\nValueError: boom", "log_info": parse_vllm_log("")}]
    md = summary_markdown(_sweep_record(servers), aggregate_runs(_fixture_runs()))
    assert "| auto | ok | 282,048 | 68.86x (4096) | 7.53 | 61.2 |" in md
    assert "| fp8 | failed_to_start | n/a | n/a | n/a | n/a |" in md
    assert "**fp8 failed:**" in md and "ValueError: boom" in md
    assert "auto log: `" in md and "GPU KV cache size: 282,048 tokens`" in md
    assert "| chat | auto | 2.01 (2) | 4 | 4 |" in md
    assert "## chat (rate in req/s; prompts: sharegpt)" in md
    assert "| 2 | auto | 2.01 (2.00–2.02) | 100.0 | 101.0 (100.0–102.0) |" in md
    assert "| 4 | fp8 |" in md and "| 1/3 |" in md
    assert "auto/chat: collapsed at 4/s" in md


# --------------------------------------------------------------------------- server lifecycle

FAKE_VLLM = textwrap.dedent('''\
    #!{python}
    """Fake `vllm serve MODEL --kv-cache-dtype D ... --port P` for tests."""
    import os, signal, sys, threading, time
    sys.path.insert(0, {repo!r})
    argv = sys.argv[1:]
    assert argv[0] == "serve", argv
    model = argv[1]
    opt = lambda k: argv[argv.index(k) + 1]
    port, dtype = int(opt("--port")), opt("--kv-cache-dtype")
    print("INFO vLLM API server version 0.0.fake", flush=True)
    if dtype in os.environ.get("FAKE_VLLM_FAIL", "").split(","):
        print("Traceback (most recent call last):", flush=True)
        print('  File "engine.py", line 1', flush=True)
        print(f"ValueError: fake failure for kv-cache-dtype={{dtype}}", flush=True)
        sys.exit(1)
    if os.environ.get("FAKE_VLLM_HANG"):
        time.sleep(600)
    tokens = 100000 if dtype == "auto" else 200000
    print(f"\\x1b[1;36m(EngineCore_DP0 pid=1)\\x1b[0;0m INFO GPU KV cache size: {{tokens:,}} tokens", flush=True)
    print(f"INFO Maximum concurrency for 4,096 tokens per request: {{tokens / 4096:.2f}}x", flush=True)
    from bench.serving.fake_server import FakeServer, FakeServerConfig
    srv = FakeServer(FakeServerConfig(model=model, ttft_s=0.005, itl_s=0.001), port=port).start()
    stop = threading.Event()
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.set())
    stop.wait()
    srv.stop()
    print("INFO shutdown complete", flush=True)
''')


@pytest.fixture
def fake_vllm(tmp_path: Path) -> str:
    path = tmp_path / "vllm"
    path.write_text(FAKE_VLLM.format(python=sys.executable, repo=str(REPO)))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_build_vllm_cmd_exact() -> None:
    assert build_vllm_cmd("M", "fp8", port=8000, max_model_len=4096, gpu_memory_utilization=0.85,
                          extra_args=["--seed", "0"]) == [
        "vllm", "serve", "M", "--kv-cache-dtype", "fp8", "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85", "--port", "8000", "--seed", "0"]


def test_server_start_parse_stop(fake_vllm: str, tmp_path: Path) -> None:
    port = free_port()
    cmd = build_vllm_cmd("m", "auto", port=port, max_model_len=4096, gpu_memory_utilization=0.85,
                         vllm_bin=fake_vllm)
    srv = VllmServer(cmd, port, tmp_path / "logs" / "auto.log", startup_timeout_s=60)
    assert srv.start() > 0 and srv.alive()
    info = parse_vllm_log(srv.log_text())
    assert info["kv_cache_tokens"] == 100000 and info["max_concurrency"] == pytest.approx(24.41)
    assert srv.log_text().startswith("$ ")
    assert srv.reset_prefix_cache() is False  # endpoint absent outside vLLM dev mode
    assert srv.stop() == 0 and not srv.alive()
    assert "shutdown complete" in srv.log_text()


def test_server_start_failure_reports_traceback(fake_vllm: str, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_VLLM_FAIL", "fp8")
    port = free_port()
    cmd = build_vllm_cmd("m", "fp8", port=port, max_model_len=4096, gpu_memory_utilization=0.85,
                         vllm_bin=fake_vllm)
    srv = VllmServer(cmd, port, tmp_path / "fp8.log", startup_timeout_s=60)
    with pytest.raises(ServerStartError) as ei:
        srv.start()
    assert ei.value.returncode == 1
    assert ei.value.log_excerpt.startswith("Traceback")
    assert "fake failure for kv-cache-dtype=fp8" in ei.value.log_excerpt


def test_server_startup_timeout_kills_process(fake_vllm: str, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_VLLM_HANG", "1")
    port = free_port()
    srv = VllmServer(build_vllm_cmd("m", "auto", port=port, max_model_len=4096,
                                    gpu_memory_utilization=0.85, vllm_bin=fake_vllm),
                     port, tmp_path / "hang.log", startup_timeout_s=3)
    with pytest.raises(ServerStartError, match="not healthy after 3 s"):
        srv.start()
    assert srv.proc is None


def test_missing_binary(tmp_path: Path) -> None:
    srv = VllmServer(["/nonexistent/vllm", "serve"], free_port(), tmp_path / "x.log")
    with pytest.raises(ServerStartError, match="could not launch"):
        srv.start()


# --------------------------------------------------------------------------- end to end

def test_sweep_end_to_end_with_fake_vllm(fake_vllm: str, tmp_path: Path, monkeypatch, capsys) -> None:
    """auto runs; fp8 fails to start -> reported, sweep continues, summary + plots generated."""
    from scripts import plot_baseline, run_baseline_sweep

    monkeypatch.setenv("KVCACHE_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("FAKE_VLLM_FAIL", "fp8")
    rc = run_baseline_sweep.main([
        "--model", "fake/model", "--tokenizer", "stub", "--vllm-bin", fake_vllm,
        "--port", str(free_port()), "--traces", "fixed", "multi_turn",
        "--rates", "fixed=4,8", "--rates", "multi_turn=2", "--repeats", "2",
        "--duration-s", "1", "--min-requests", "3", "--max-requests", "6", "--no-extend",
        "--slo-ttft-ms", "1000", "--slo-itl-ms", "100", "--startup-timeout-s", "60"])
    assert rc == 1  # fp8 failed
    out = capsys.readouterr().out
    assert "FAILED TO START" in out and "fake failure for kv-cache-dtype=fp8" in out

    results = tmp_path / "results"
    md = (results / "summary" / "baseline.md").read_text()
    assert "| auto | ok | 100,000 | 24.41x (4096) |" in md
    assert "| fp8 | failed_to_start |" in md and "fake failure for kv-cache-dtype=fp8" in md
    assert "## fixed (rate in req/s" in md and "## multi_turn (rate in sessions/s" in md
    assert "| 4 | auto |" in md and "| 8 | auto |" in md
    runs = list((results / "raw").glob("*/serving_*.json"))
    assert len(runs) == 2 * 2 + 1 * 2  # fixed: 2 rates x 2 repeats; multi_turn: 1 x 2
    rec = json.loads(runs[0].read_text())
    assert rec["config"]["kv_cache_dtype"] == "auto" and rec["config"]["server_log_info"]["kv_cache_tokens"] == 100000
    seeds = {(json.loads(p.read_text())["config"]["rate"], json.loads(p.read_text())["config"]["repeat"]):
             json.loads(p.read_text())["config"]["seed"] for p in runs}
    assert len(set(seeds.values())) == len(seeds)  # distinct prompts per (rate, repeat)
    logs = list((results / "raw").glob("*/vllm_logs/*.log"))
    assert sorted(p.name.split("_", 1)[1] for p in logs) == ["auto.log", "fp8.log"]

    sweep_id = json.loads(next((results / "raw").glob("*/baseline_sweep_*.json")).read_text())["config"]["sweep_id"]
    md_path_before = (results / "summary" / "baseline.md").read_text()
    assert run_baseline_sweep.main(["--summarize-only", sweep_id]) == 0
    assert (results / "summary" / "baseline.md").read_text() == md_path_before  # regenerable

    assert plot_baseline.main([]) == 0
    for trace in ("fixed", "multi_turn"):
        png = results / "summary" / f"baseline_{trace}.png"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and png.stat().st_size > 10_000


# --------------------------------------------------------------------------- plotting

def test_plot_from_fixture_records(tmp_path: Path, monkeypatch) -> None:
    from scripts import plot_baseline

    monkeypatch.setenv("KVCACHE_RESULTS_DIR", str(tmp_path))
    meta = {"git": {"sha": "x", "dirty": False}, "gpu": {}, "packages": {}}
    for r in _fixture_runs():
        save_result("serving_chat", r["config"], r["metrics"], None, metadata=meta)
    save_result("baseline_sweep", _sweep_record([])["config"], {"servers": []}, None, metadata=meta)
    assert plot_baseline.latest_sweep_id() == "S1"
    [png] = plot_baseline.plot_sweep("S1")
    assert png == tmp_path / "summary" / "baseline_chat.png"
    assert png.read_bytes()[:4] == b"\x89PNG" and png.stat().st_size > 10_000
    with pytest.raises(SystemExit):
        plot_baseline.plot_sweep("missing")
