"""Per-request records and serving metrics (pure functions; no I/O).

Definitions (times in seconds on the client's monotonic clock; reported in ms):
  TTFT   first content chunk arrival - send time
  ITL    gaps between consecutive content-chunk arrivals (pooled over requests). One chunk is
         normally one token in vLLM; this matches ``vllm bench serve``'s ITL definition.
  TPOT   (E2E - TTFT) / (output_tokens - 1), per request with > 1 output token
  E2E    last chunk arrival (end of stream) - send time
  throughput  completed requests / duration and output tokens / duration, where duration runs
              from the first send to the last completion
  goodput     requests meeting the SLO / duration. SLO: TTFT <= slo_ttft_ms AND the request's
              own p90 ITL <= slo_itl_ms (vacuously true with < 2 content chunks). Failed requests
              never meet it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

PERCENTILES = (50, 90, 99)


@dataclass
class RequestRecord:
    request_id: int
    session_id: int
    turn: int
    prompt_tokens: int                  # measured client-side with the tokenizer
    expected_output_tokens: int         # max_tokens sent (ignore_eos forces this length)
    scheduled_time: float = 0.0         # when the arrival process wanted it sent
    send_time: float = 0.0
    first_token_time: float | None = None
    chunk_times: list[float] = field(default_factory=list)  # arrivals of content chunks
    end_time: float | None = None
    output_tokens: int | None = None    # server usage.completion_tokens, else content chunks
    server_prompt_tokens: int | None = None
    error: str | None = None
    warmup: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and self.first_token_time is not None

    @property
    def ttft(self) -> float | None:
        return None if self.first_token_time is None else self.first_token_time - self.send_time

    @property
    def itls(self) -> list[float]:
        return [b - a for a, b in zip(self.chunk_times, self.chunk_times[1:])]

    @property
    def e2e(self) -> float | None:
        return None if self.end_time is None else self.end_time - self.send_time

    @property
    def tpot(self) -> float | None:
        n = self.output_tokens
        if not self.ok or self.e2e is None or n is None or n < 2:
            return None
        return (self.e2e - self.ttft) / (n - 1)  # type: ignore[operator]

    def p90_itl(self) -> float | None:
        itls = self.itls
        return float(np.percentile(itls, 90)) if itls else None

    def meets_slo(self, slo_ttft_ms: float, slo_itl_ms: float) -> bool:
        if not self.ok:
            return False
        p90 = self.p90_itl()
        eps = 1e-6  # ms; a value exactly at the threshold (up to float rounding) meets it
        return (self.ttft * 1e3 <= slo_ttft_ms + eps  # type: ignore[operator]
                and (p90 is None or p90 * 1e3 <= slo_itl_ms + eps))

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.update(ttft=self.ttft, e2e=self.e2e, tpot=self.tpot)
        return d


def dist_ms(values_s: Sequence[float]) -> dict[str, float | int | None]:
    """mean/median/p90/p99/min/max in milliseconds."""
    if not len(values_s):
        return {"n": 0, "mean": None, "median": None, "p90": None, "p99": None,
                "min": None, "max": None}
    a = np.asarray(values_s, dtype=float) * 1e3
    out: dict[str, float | int | None] = {"n": int(a.size), "mean": float(a.mean()),
                                          "median": float(np.median(a))}
    for p in PERCENTILES[1:]:
        out[f"p{p}"] = float(np.percentile(a, p))
    out["min"], out["max"] = float(a.min()), float(a.max())
    return out


def compute_metrics(records: Sequence[RequestRecord], slo_ttft_ms: float,
                    slo_itl_ms: float) -> dict[str, Any]:
    recs = [r for r in records if not r.warmup]
    ok = [r for r in recs if r.ok]
    failed = [r for r in recs if not r.ok]
    if ok:
        start = min(r.send_time for r in recs)
        end = max(r.end_time for r in ok if r.end_time is not None)
        duration = max(end - start, 1e-9)
    else:
        duration = 0.0
    out_tok = sum(r.output_tokens or 0 for r in ok)
    in_tok = sum(r.prompt_tokens for r in ok)
    good = [r for r in ok if r.meets_slo(slo_ttft_ms, slo_itl_ms)]
    lag = [r.send_time - r.scheduled_time for r in recs]
    mismatched = [r.request_id for r in ok
                  if r.output_tokens is not None and r.output_tokens != r.expected_output_tokens]

    def rate(x: float) -> float | None:
        return x / duration if duration > 0 else None

    return {
        "num_requests": len(recs),
        "completed": len(ok),
        "failed": len(failed),
        "errors": sorted({r.error for r in failed if r.error})[:10],
        "duration_s": duration,
        "request_throughput_rps": rate(len(ok)),
        "output_throughput_tps": rate(out_tok),
        "input_throughput_tps": rate(in_tok),
        "total_output_tokens": out_tok,
        "total_input_tokens": in_tok,
        "goodput_rps": rate(len(good)),
        "slo_attainment": len(good) / len(recs) if recs else None,
        "slo": {"ttft_ms": slo_ttft_ms, "p90_itl_ms": slo_itl_ms},
        "ttft_ms": dist_ms([r.ttft for r in ok]),  # type: ignore[misc]
        "itl_ms": dist_ms([x for r in ok for x in r.itls]),
        "tpot_ms": dist_ms([r.tpot for r in ok if r.tpot is not None]),  # type: ignore[misc]
        "e2e_ms": dist_ms([r.e2e for r in ok]),  # type: ignore[misc]
        "send_lag_ms": dist_ms(lag),  # client-side scheduling delay; should be ~0
        "output_len_mismatches": len(mismatched),
        "server_prompt_tokens_total": sum(r.server_prompt_tokens or 0 for r in ok),
    }


def format_table(m: dict[str, Any], title: str = "") -> str:
    """Compact, paste-friendly summary."""
    def f(v: Any, nd: int = 1) -> str:
        return "n/a" if v is None else f"{v:.{nd}f}"

    lines = [title] if title else []
    lines += [
        f"requests {m['completed']}/{m['num_requests']} ok, {m['failed']} failed, "
        f"duration {f(m['duration_s'], 2)} s",
        f"throughput {f(m['request_throughput_rps'], 3)} req/s, "
        f"{f(m['output_throughput_tps'])} out tok/s, {f(m['input_throughput_tps'])} in tok/s",
        f"goodput {f(m['goodput_rps'], 3)} req/s (SLO TTFT<={m['slo']['ttft_ms']:g} ms & "
        f"p90 ITL<={m['slo']['p90_itl_ms']:g} ms; attainment "
        f"{f(None if m['slo_attainment'] is None else 100 * m['slo_attainment'])}%)",
        f"{'metric (ms)':<12}{'mean':>9}{'median':>9}{'p90':>9}{'p99':>9}{'max':>9}{'n':>7}",
    ]
    for key, label in (("ttft_ms", "TTFT"), ("itl_ms", "ITL"), ("tpot_ms", "TPOT"),
                       ("e2e_ms", "E2E"), ("send_lag_ms", "send lag")):
        d = m[key]
        lines.append(f"{label:<12}{f(d['mean']):>9}{f(d['median']):>9}{f(d['p90']):>9}"
                     f"{f(d['p99']):>9}{f(d['max']):>9}{d['n']:>7}")
    if m["output_len_mismatches"]:
        lines.append(f"WARNING: {m['output_len_mismatches']} requests returned a different number "
                     "of tokens than max_tokens (ignore_eos not honoured?)")
    for e in m["errors"]:
        lines.append(f"error: {e}")
    return "\n".join(lines)
