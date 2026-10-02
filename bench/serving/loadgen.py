"""Async load generator: streams chat completions and records per-request timings.

Arrival model: sessions start according to the arrival process; a session's turns run
sequentially (turn k+1 is sent ``think_time_s`` after turn k completes). Single-turn traces have
one turn per session, so for them "session" = "request".

  rate=R (finite)       Poisson: session start gaps ~ Exponential(mean 1/R), seeded.
  rate=inf              all sessions are released at t=0.
  max_concurrency=C     at most C sessions in flight (a semaphore around each session).
                        rate=inf + C is closed-loop: C users, each starting its next session
                        as soon as the previous one finishes.

For multi-turn traces the concurrency limit counts users (sessions), so a slot stays held during
think time. Timing uses ``time.perf_counter`` relative to the start of the measured phase.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any, Sequence

import httpx
import numpy as np

from bench.serving.metrics import RequestRecord
from bench.serving.traces import Trace, Turn


def arrival_times(n: int, rate: float, seed: int) -> list[float]:
    """Start offsets (s) for ``n`` sessions: Poisson process at ``rate``/s, or all 0 if inf."""
    if n <= 0:
        return []
    if math.isinf(rate):
        return [0.0] * n
    if rate <= 0:
        raise ValueError("rate must be > 0 (use inf for no pacing)")
    gaps = np.random.default_rng(seed).exponential(1.0 / rate, size=n)
    gaps[0] = 0.0  # first session starts immediately
    return np.cumsum(gaps).tolist()


def build_payload(model: str, turn: Turn, ignore_eos: bool = True) -> dict[str, Any]:
    return {
        "model": model,
        "messages": turn.messages,
        "max_tokens": turn.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # vLLM extension (what the OpenAI SDK sends via extra_body): keep generating past EOS so
        # every request produces exactly max_tokens tokens.
        "ignore_eos": ignore_eos,
    }


async def stream_request(client: httpx.AsyncClient, url: str, payload: dict[str, Any],
                         rec: RequestRecord, clock: Any) -> RequestRecord:
    """Send one streaming chat completion; fill ``rec`` with timings, token counts, errors."""
    rec.send_time = clock()
    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                rec.error = f"HTTP {resp.status_code}: {body[:200]}"
                rec.end_time = clock()
                return rec
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                now = clock()
                chunk = json.loads(data)
                if chunk.get("error"):
                    rec.error = f"stream error: {str(chunk['error'])[:200]}"
                    break
                for choice in chunk.get("choices") or []:
                    if (choice.get("delta") or {}).get("content"):
                        if rec.first_token_time is None:
                            rec.first_token_time = now
                        rec.chunk_times.append(now)
                        break
                usage = chunk.get("usage")
                if usage:
                    rec.output_tokens = usage.get("completion_tokens")
                    rec.server_prompt_tokens = usage.get("prompt_tokens")
    except Exception as exc:  # connection errors, timeouts, bad JSON
        rec.error = f"{type(exc).__name__}: {exc}"[:300]
    rec.end_time = clock()
    if rec.output_tokens is None and rec.chunk_times:
        rec.output_tokens = len(rec.chunk_times)
    if rec.error is None and rec.first_token_time is None:
        rec.error = "no content received"
    return rec


async def run_trace(trace: Trace, base_url: str, model: str, *, rate: float = math.inf,
                    max_concurrency: int | None = None, seed: int = 0,
                    timeout_s: float = 600.0, warmup_turns: Sequence[Turn] = (),
                    ignore_eos: bool = True) -> list[RequestRecord]:
    """Run warmup turns (sequentially, flagged ``warmup``), then the trace. Returns all records."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    timeout = httpx.Timeout(timeout_s, connect=30.0)
    records: list[RequestRecord] = []
    t0 = time.perf_counter()

    def clock() -> float:
        return time.perf_counter() - t0

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for i, turn in enumerate(warmup_turns):
            rec = RequestRecord(request_id=-1 - i, session_id=-1, turn=0,
                                prompt_tokens=turn.prompt_tokens,
                                expected_output_tokens=turn.max_tokens, warmup=True)
            await stream_request(client, url, build_payload(model, turn, ignore_eos), rec, clock)
            rec.scheduled_time = rec.send_time
            records.append(rec)

        t0 = time.perf_counter()  # measured phase starts here; warmup times are discarded
        starts = arrival_times(len(trace.sessions), rate, seed)
        sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        ids: dict[tuple[int, int], int] = {}
        for s in trace.sessions:
            for k in range(len(s.turns)):
                ids[(s.session_id, k)] = len(ids)

        async def run_session(session_idx: int) -> None:
            session = trace.sessions[session_idx]
            delay = starts[session_idx] - clock()
            if delay > 0:
                await asyncio.sleep(delay)
            # scheduled_time is when the request *should* go out, so send lag measures client
            # lateness only: the Poisson arrival, or (with a cap) the moment a slot freed up.
            scheduled = starts[session_idx]
            if sem is not None:
                await sem.acquire()
                scheduled = max(scheduled, clock())
            try:
                for k, turn in enumerate(session.turns):
                    if k > 0:
                        if turn.think_time_s > 0:
                            await asyncio.sleep(turn.think_time_s)
                        scheduled = clock()
                    rec = RequestRecord(request_id=ids[(session.session_id, k)],
                                        session_id=session.session_id, turn=k,
                                        prompt_tokens=turn.prompt_tokens,
                                        expected_output_tokens=turn.max_tokens,
                                        scheduled_time=scheduled)
                    records.append(rec)
                    await stream_request(client, url, build_payload(model, turn, ignore_eos),
                                         rec, clock)
            finally:
                if sem is not None:
                    sem.release()

        await asyncio.gather(*(run_session(i) for i in range(len(trace.sessions))))
    return sorted(records, key=lambda r: (not r.warmup, r.request_id))
