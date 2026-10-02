"""Serving metric math on hand-built records (exact), arrival process, and traces (stub tokenizer)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bench.serving.loadgen import arrival_times, build_payload
from bench.serving.metrics import RequestRecord, compute_metrics, dist_ms, format_table
from bench.serving.tokenization import StubTokenizer, prompt_tokens
from bench.serving.traces import build_chat, build_trace, make_text


def rec(i: int, send: float, chunks: list[float], out: int | None = None, error: str | None = None,
        expected: int | None = None) -> RequestRecord:
    r = RequestRecord(request_id=i, session_id=i, turn=0, prompt_tokens=100,
                      expected_output_tokens=expected if expected is not None else len(chunks),
                      scheduled_time=send, send_time=send, error=error)
    r.chunk_times = list(chunks)
    r.first_token_time = chunks[0] if chunks else None
    r.end_time = chunks[-1] if chunks else send + 0.5
    r.output_tokens = out if out is not None else (len(chunks) or None)
    return r


def test_record_definitions() -> None:
    r = rec(0, 1.0, [1.2, 1.25, 1.35, 1.40])
    assert r.ttft == pytest.approx(0.2)
    assert r.itls == pytest.approx([0.05, 0.10, 0.05])
    assert r.e2e == pytest.approx(0.4)
    assert r.tpot == pytest.approx((0.4 - 0.2) / 3)
    assert r.p90_itl() == pytest.approx(np.percentile([0.05, 0.10, 0.05], 90))
    one = rec(1, 0.0, [0.1])
    assert one.tpot is None and one.itls == [] and one.p90_itl() is None


def test_slo_boundaries() -> None:
    r = rec(0, 0.0, [0.5, 0.55, 0.60])  # TTFT 500 ms, ITLs 50 ms
    assert r.meets_slo(500, 50)
    assert not r.meets_slo(499.9, 50)
    assert not r.meets_slo(500, 49.9)
    assert rec(1, 0.0, [0.1]).meets_slo(500, 1)  # no ITLs -> ITL condition vacuous
    assert not rec(2, 0.0, [], error="HTTP 500").meets_slo(1e9, 1e9)


def test_compute_metrics_exact() -> None:
    records = [
        rec(0, 0.0, [0.1, 0.2, 0.3]),            # TTFT 100, ITL 100,100, E2E 300 -> fails ITL SLO 50
        rec(1, 0.5, [0.55, 0.57, 0.59, 0.61]),   # TTFT 50, ITL 20x3, E2E 110 -> good
        rec(2, 1.0, [1.02, 1.04]),               # TTFT 20, ITL 20, E2E 40, ends at 1.04 -> good
        rec(3, 1.5, [], error="HTTP 500: boom"),  # failed
    ]
    warm = rec(99, -5.0, [-4.0, -3.0])
    warm.warmup = True
    m = compute_metrics(records + [warm], slo_ttft_ms=100, slo_itl_ms=50)
    assert m["num_requests"] == 4 and m["completed"] == 3 and m["failed"] == 1
    assert m["errors"] == ["HTTP 500: boom"]
    assert m["duration_s"] == pytest.approx(1.04)  # first send 0.0 -> last completion 1.04
    assert m["request_throughput_rps"] == pytest.approx(3 / 1.04)
    assert m["total_output_tokens"] == 9
    assert m["output_throughput_tps"] == pytest.approx(9 / 1.04)
    assert m["input_throughput_tps"] == pytest.approx(300 / 1.04)
    assert m["goodput_rps"] == pytest.approx(2 / 1.04)
    assert m["slo_attainment"] == pytest.approx(2 / 4)
    assert m["ttft_ms"]["mean"] == pytest.approx((100 + 50 + 20) / 3)
    assert m["ttft_ms"]["median"] == pytest.approx(50)
    assert m["itl_ms"]["n"] == 6 and m["itl_ms"]["max"] == pytest.approx(100)
    assert m["e2e_ms"]["median"] == pytest.approx(110)
    assert m["tpot_ms"]["n"] == 3 and m["tpot_ms"]["min"] == pytest.approx(20)
    assert m["output_len_mismatches"] == 0
    assert "goodput" in format_table(m)


def test_output_length_mismatch_detected() -> None:
    r = rec(0, 0.0, [0.1, 0.2], out=2, expected=128)
    m = compute_metrics([r], 1e9, 1e9)
    assert m["output_len_mismatches"] == 1
    assert "ignore_eos" in format_table(m)


def test_all_failed_and_empty() -> None:
    m = compute_metrics([rec(0, 0.0, [], error="x")], 1, 1)
    assert m["completed"] == 0 and m["request_throughput_rps"] is None
    assert m["ttft_ms"]["n"] == 0
    assert dist_ms([]) ["median"] is None
    format_table(m)


def test_arrival_times() -> None:
    a = arrival_times(20_000, rate=4.0, seed=1)
    assert a[0] == 0.0 and all(y >= x for x, y in zip(a, a[1:]))
    assert np.mean(np.diff(a)) == pytest.approx(0.25, rel=0.03)
    assert np.std(np.diff(a)) == pytest.approx(0.25, rel=0.05)  # exponential: std == mean
    assert arrival_times(5, 4.0, seed=1) == arrival_times(5, 4.0, seed=1)
    assert arrival_times(5, 4.0, seed=1) != arrival_times(5, 4.0, seed=2)
    assert arrival_times(3, math.inf, 0) == [0.0, 0.0, 0.0]
    assert arrival_times(0, 1.0, 0) == []
    with pytest.raises(ValueError):
        arrival_times(3, 0.0, 0)


def test_payload_forces_output_length() -> None:
    from bench.serving.traces import Turn

    p = build_payload("m", Turn(messages=[{"role": "user", "content": "hi"}], max_tokens=7,
                                prompt_tokens=1))
    assert p["max_tokens"] == 7 and p["ignore_eos"] is True and p["stream"] is True
    assert p["stream_options"] == {"include_usage": True}


# --------------------------------------------------------------------------- traces

@pytest.fixture
def tok() -> StubTokenizer:
    return StubTokenizer()


def test_make_text_exact_with_stub(tok) -> None:
    rng = np.random.default_rng(0)
    for n in (1, 7, 300):
        assert len(tok.encode(make_text(tok, n, rng))) == n
    assert make_text(tok, 0, rng) == ""


def _flat(trace):
    return [(t.messages, t.max_tokens, t.prompt_tokens, t.think_time_s) for t in trace.turns()]


@pytest.mark.parametrize("name", ["chat", "long_context", "multi_turn", "fixed"])
def test_traces_deterministic_and_sized(name: str) -> None:
    kw = {"sharegpt": "off"} if name == "chat" else {}
    a = build_trace(name, StubTokenizer(), 25, seed=3, **kw)
    b = build_trace(name, StubTokenizer(), 25, seed=3, **kw)
    c = build_trace(name, StubTokenizer(), 25, seed=4, **kw)
    assert a.num_requests == 25
    assert _flat(a) == _flat(b) and _flat(a) != _flat(c)
    tok = StubTokenizer()
    for t in a.turns():
        assert t.prompt_tokens == prompt_tokens(tok, t.messages)
        assert t.prompt_tokens + t.max_tokens <= 4096 and t.max_tokens >= 1
    assert a.stats()["num_requests"] == 25 and a.meta["seed"] == 3


def test_long_context_ranges(tok) -> None:
    tr = build_trace("long_context", tok, 40, seed=0)
    for t in tr.turns():
        assert 2000 <= t.prompt_tokens <= 3500 and 32 <= t.max_tokens <= 128
    assert build_trace("long_context", tok, 5, seed=0, max_model_len=3000,
                       prompt_range=(2000, 2900)).num_requests == 5


def test_multi_turn_structure(tok) -> None:
    tr = build_trace("multi_turn", tok, 60, seed=0, think_time_mean_s=2.0)
    for s in tr.sessions[:-1]:  # last session may be cut short by num_requests
        assert 3 <= len(s.turns) <= 6
    for s in tr.sessions:
        assert s.turns[0].think_time_s == 0.0
        assert all(t.think_time_s > 0 for t in s.turns[1:])
        for prev, cur in zip(s.turns, s.turns[1:]):
            # Full conversation resent: previous prompt + assistant reply + new user message.
            assert cur.messages[:len(prev.messages)] == prev.messages
            assert len(cur.messages) == len(prev.messages) + 2
            reply = cur.messages[len(prev.messages)]
            assert reply["role"] == "assistant" and len(tok.encode(reply["content"])) == prev.max_tokens
            assert cur.prompt_tokens > prev.prompt_tokens
    thinks = [t.think_time_s for t in tr.turns() if t.think_time_s > 0]
    assert 1.0 < np.mean(thinks) < 3.5


def test_multi_turn_respects_max_model_len(tok) -> None:
    tr = build_trace("multi_turn", tok, 30, seed=0, max_model_len=500)
    assert all(t.prompt_tokens + t.max_tokens <= 500 for t in tr.turns())
    assert tr.meta["sessions_truncated_by_max_model_len"] > 0


def test_chat_sharegpt_path_and_filter(tok) -> None:
    words = lambda n: " ".join(["w"] * n)  # noqa: E731
    pairs = [(words(10), words(20)), (words(2), words(20)),      # prompt too short
             (words(1500), words(10)), (words(900), words(1500)),  # too long / total > 2048
             (words(30), words(40)), (words(50), words(3))]       # output too short
    tr = build_chat(tok, 2, np.random.default_rng(0), 4096, pair_loader=lambda: pairs)
    assert tr.source == "sharegpt" and tr.meta["sharegpt_pairs_available"] == 6
    assert sorted((t.prompt_tokens, t.max_tokens) for t in tr.turns()) == [(10, 20), (30, 40)]


def test_chat_falls_back_and_records_reason(tok) -> None:
    def broken():
        raise OSError("offline")

    tr = build_chat(tok, 10, np.random.default_rng(0), 4096, pair_loader=broken)
    assert tr.source == "synthetic" and "offline" in tr.meta["fallback_reason"]
    tr2 = build_chat(tok, 10, np.random.default_rng(0), 4096, pair_loader=lambda: [("a b c d e", "x y z w v")])
    assert tr2.source == "synthetic" and "only 1" in tr2.meta["fallback_reason"]
    off = build_trace("chat", tok, 5, sharegpt="off")
    assert off.source == "synthetic" and "disabled" in off.meta["fallback_reason"]
    for t in tr.turns():
        assert 4 <= t.prompt_tokens <= 1024 and t.prompt_tokens + t.max_tokens <= 2048


class _HFLike(StubTokenizer):
    """Mimics a transformers tokenizer with a chat template (adds 3 template tokens per message
    plus 2 for the generation prompt)."""

    chat_template = "fake"

    def __init__(self, as_dict: bool) -> None:
        super().__init__()
        self.as_dict = as_dict

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        ids = []
        for m in messages:
            ids += [0, 0, 0] + self.encode(m["content"])
        ids += [0, 0] if add_generation_prompt else []
        return {"input_ids": ids} if self.as_dict else ids


@pytest.mark.parametrize("as_dict", [False, True])
def test_prompt_tokens_uses_chat_template(as_dict: bool) -> None:
    tok = _HFLike(as_dict)
    msgs = [{"role": "user", "content": "a b c"}, {"role": "assistant", "content": "d e"}]
    assert prompt_tokens(tok, msgs) == 3 + 3 + 3 + 2 + 2
    tr = build_trace("long_context", tok, 3, seed=0)  # template overhead subtracted from target
    assert all(2000 <= t.prompt_tokens <= 3500 for t in tr.turns())
