"""Load generator end to end against the fake OpenAI-compatible streaming server (CPU only).

Timing assertions use injected server delays with generous upper bounds for a loaded machine.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from bench.serving import run as run_cli
from bench.serving.fake_server import FakeServer, FakeServerConfig
from bench.serving.loadgen import arrival_times, run_trace
from bench.serving.metrics import compute_metrics
from bench.serving.tokenization import StubTokenizer
from bench.serving.traces import build_fixed, build_trace


@pytest.fixture
def server():
    srv = FakeServer(FakeServerConfig(ttft_s=0.05, itl_s=0.01)).start()
    yield srv
    srv.stop()


@pytest.fixture
def fast_server():
    srv = FakeServer(FakeServerConfig(ttft_s=0.005, itl_s=0.001)).start()
    yield srv
    srv.stop()


def _fixed(n: int, out: int = 20, seed: int = 0):
    return build_fixed(StubTokenizer(), n, np.random.default_rng(seed), 4096,
                       input_len=30, output_len=out)


def test_metrics_match_injected_delays(server) -> None:
    recs = asyncio.run(run_trace(_fixed(8), server.base_url, "fake-model", max_concurrency=4))
    m = compute_metrics(recs, slo_ttft_ms=1000, slo_itl_ms=1000)
    assert m["completed"] == 8 and m["failed"] == 0
    assert all(r.output_tokens == 20 and len(r.chunk_times) == 20 for r in recs)
    assert m["output_len_mismatches"] == 0
    # Injected: TTFT 50 ms (role-only chunk at t=0 must NOT count), ITL 10 ms.
    assert 49 <= m["ttft_ms"]["median"] < 120
    assert 9.5 <= m["itl_ms"]["median"] < 25
    assert 9.5 <= m["tpot_ms"]["median"] < 25
    assert m["e2e_ms"]["min"] >= 50 + 19 * 10 - 1
    assert m["goodput_rps"] == pytest.approx(m["request_throughput_rps"])
    assert all(r.server_prompt_tokens == 30 for r in recs)


def test_request_body(server) -> None:
    asyncio.run(run_trace(_fixed(2, out=5), server.base_url, "fake-model"))
    body = server.state.requests[0]
    assert body["model"] == "fake-model" and body["stream"] is True
    assert body["max_tokens"] == 5 and body["ignore_eos"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["messages"][0]["role"] == "user"


def test_closed_loop_concurrency_cap(server) -> None:
    recs = asyncio.run(run_trace(_fixed(10, out=5), server.base_url, "fake-model",
                                 max_concurrency=2))
    assert server.state.peak_active == 2
    assert compute_metrics(recs, 1e9, 1e9)["completed"] == 10


def test_unbounded_release_all_at_once(server) -> None:
    asyncio.run(run_trace(_fixed(6, out=5), server.base_url, "fake-model"))
    assert server.state.peak_active == 6


def test_poisson_send_times_follow_arrivals(server) -> None:
    rate, n = 20.0, 12
    recs = asyncio.run(run_trace(_fixed(n, out=3), server.base_url, "fake-model", rate=rate, seed=5))
    expected = arrival_times(n, rate, seed=5)
    sends = [r.send_time for r in sorted(recs, key=lambda r: r.request_id)]
    assert [r.scheduled_time for r in sorted(recs, key=lambda r: r.request_id)] == expected
    assert all(0 <= s - e < 0.05 for s, e in zip(sends, expected))
    assert compute_metrics(recs, 1e9, 1e9)["send_lag_ms"]["max"] < 50


def test_multi_turn_sequencing(fast_server) -> None:
    server = fast_server
    trace = build_trace("multi_turn", StubTokenizer(), 9, seed=1, think_time_mean_s=0.05)
    recs = asyncio.run(run_trace(trace, server.base_url, "fake-model"))
    by_session: dict[int, list] = {}
    for r in recs:
        by_session.setdefault(r.session_id, []).append(r)
    for sid, rs in by_session.items():
        rs.sort(key=lambda r: r.turn)
        turns = trace.sessions[sid].turns
        for k in range(1, len(rs)):
            # Turn k goes out only after turn k-1 finished plus its think time.
            assert rs[k].send_time >= rs[k - 1].end_time + turns[k].think_time_s - 1e-3
    # Server saw each session's growing conversation.
    sent = [b["messages"] for b in server.state.requests]
    for s in trace.sessions:
        for t in s.turns:
            assert t.messages in sent
    assert compute_metrics(recs, 1e9, 1e9)["completed"] == 9


def test_errors_recorded_and_excluded_from_goodput() -> None:
    with FakeServer(FakeServerConfig(ttft_s=0.01, itl_s=0.001, fail_every=3)) as srv:
        recs = asyncio.run(run_trace(_fixed(9, out=4), srv.base_url, "fake-model", max_concurrency=1))
    m = compute_metrics(recs, 1e9, 1e9)
    assert m["failed"] == 3 and m["completed"] == 6
    assert m["errors"] and m["errors"][0].startswith("HTTP 500")
    assert m["goodput_rps"] == pytest.approx(m["request_throughput_rps"])


def test_ignore_eos_off_is_detected() -> None:
    with FakeServer(FakeServerConfig(ttft_s=0.01, itl_s=0.001, eos_after=3)) as srv:
        recs = asyncio.run(run_trace(_fixed(2, out=10), srv.base_url, "fake-model", ignore_eos=False))
    assert compute_metrics(recs, 1e9, 1e9)["output_len_mismatches"] == 2


def test_connection_error_recorded() -> None:
    recs = asyncio.run(run_trace(_fixed(2), "http://127.0.0.1:9", "m", timeout_s=2))
    assert all(not r.ok and r.error for r in recs)


def test_warmup_excluded(server) -> None:
    warm = list(_fixed(3, out=2, seed=9).turns())
    recs = asyncio.run(run_trace(_fixed(4, out=2), server.base_url, "fake-model", warmup_turns=warm))
    assert sum(r.warmup for r in recs) == 3 and len(server.state.requests) == 7
    m = compute_metrics(recs, 1e9, 1e9)
    assert m["num_requests"] == 4
    assert min(r.send_time for r in recs if not r.warmup) >= 0  # clock restarts after warmup


def test_cli_end_to_end(fast_server, tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("KVCACHE_RESULTS_DIR", str(tmp_path))
    rc = run_cli.main(["--base-url", fast_server.base_url, "--tokenizer", "stub", "--trace", "multi_turn",
                       "--rate", "50", "--num-requests", "8", "--think-time-s", "0.01",
                       "--slo-ttft-ms", "500", "--slo-itl-ms", "50", "--warmup-requests", "2",
                       "--no-gpu-monitor"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "== serving multi_turn | fake-model | Poisson 50/s ==" in out and "TTFT" in out
    [path] = list((tmp_path / "raw").glob("*/serving_multi_turn_*.json"))
    rec = json.loads(path.read_text())
    assert rec["config"]["trace"] == "multi_turn" and rec["config"]["model"] == "fake-model"
    assert rec["config"]["trace_stats"]["num_requests"] == 8
    m = rec["metrics"]
    assert m["completed"] == 8 and m["warmup_requests"] == 2
    assert len(m["records"]) == 10 and "chunk_times" in m["records"][-1]
    assert rec["gpu"]["samples"] == []


def test_cli_unreachable_server() -> None:
    with pytest.raises(SystemExit, match="not reachable"):
        run_cli.main(["--base-url", "http://127.0.0.1:9", "--tokenizer", "stub", "--trace", "fixed",
                      "--no-save", "--no-gpu-monitor"])


def test_cli_rejects_unknown_model(server) -> None:
    with pytest.raises(SystemExit, match="not served"):
        run_cli.main(["--base-url", server.base_url, "--model", "other", "--tokenizer", "stub",
                      "--trace", "fixed", "--no-save", "--no-gpu-monitor"])
