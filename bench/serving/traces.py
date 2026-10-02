"""Deterministic request traces for the serving load generator.

Every trace is a list of sessions; a session is a list of turns sent sequentially (single-turn
traces have one turn per session). All randomness comes from ``numpy.random.default_rng(seed)``,
and all lengths are measured with the served model's tokenizer (chat template included).

Traces:
  chat          first (human, gpt) exchange of ShareGPT conversations, via HF ``datasets``
                (cached by HF). Output length = tokenized length of the gpt reply. Falls back to
                lognormal synthetic lengths if ShareGPT can't be loaded; ``trace.source`` says
                which was used and why. Filter (as in vLLM's ShareGPT benchmark): prompt <= 1024,
                prompt + output <= 2048 tokens, both >= 4.
  long_context  2000-3500-token prompts (template included), 32-128-token outputs; every request
                fits ``max_model_len`` (default 4096).
  multi_turn    sessions of 3-6 turns; each turn resends the whole conversation so far (growing
                shared prefix) plus a new user message, after an exponential think time.
                Assistant turns in the history are deterministic synthetic text with the same
                token length the server was asked to generate, so prompt lengths are reproducible.
  fixed         fixed input/output lengths with random-word prompts; matches
                ``vllm bench serve --dataset-name random`` for apples-to-apples checks.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator

import numpy as np

from bench.serving.tokenization import Message, prompt_tokens, text_tokens

TRACES = ("chat", "long_context", "multi_turn", "fixed")
SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"

# Plain words that are single tokens in common BPE vocabularies (with a leading space).
_WORDS = (
    "the of and to in is was for on that with as by at from it his an were are which this be "
    "has had not but they or one their first new after also all two been have more who would "
    "its time she when other year years over into most some only city during can her about there "
    "between such made used up world these through then later many three may under state well "
    "known part people both while where since film being would school before area him team work "
    "number system water house music life game place found river family group name light paper "
    "table window market garden summer winter morning evening story letter number simple strong"
).split()


@dataclass
class Turn:
    messages: list[Message]
    max_tokens: int
    prompt_tokens: int          # measured with the tokenizer, chat template included
    think_time_s: float = 0.0   # wait after the previous turn completes (0 for first turns)


@dataclass
class Session:
    session_id: int
    turns: list[Turn]


@dataclass
class Trace:
    name: str
    sessions: list[Session]
    source: str                         # "sharegpt" | "synthetic"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_requests(self) -> int:
        return sum(len(s.turns) for s in self.sessions)

    def turns(self) -> Iterator[Turn]:
        for s in self.sessions:
            yield from s.turns

    def stats(self) -> dict[str, Any]:
        def st(vals: list[float]) -> dict[str, float]:
            return {"mean": statistics.fmean(vals), "median": statistics.median(vals),
                    "min": min(vals), "max": max(vals)} if vals else {}

        turns = list(self.turns())
        return {
            "name": self.name, "source": self.source, "num_sessions": len(self.sessions),
            "num_requests": len(turns),
            "prompt_tokens": st([t.prompt_tokens for t in turns]),
            "output_tokens": st([t.max_tokens for t in turns]),
            "turns_per_session": st([len(s.turns) for s in self.sessions]),
            **self.meta,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"stats": self.stats(), "sessions": [asdict(s) for s in self.sessions]}


# --------------------------------------------------------------------------- text helpers

def make_text(tok: Any, n_tokens: int, rng: np.random.Generator) -> str:
    """Random-word text that tokenizes to ~``n_tokens`` tokens (exact for word-level tokenizers).

    Generates words, encodes, truncates to ``n_tokens`` ids and decodes. Decode/encode may not
    round-trip exactly for BPE tokenizers; callers record measured lengths, not targets.
    """
    if n_tokens <= 0:
        return ""
    words: list[str] = []
    ids: list[int] = []
    while len(ids) < n_tokens:
        words += list(rng.choice(_WORDS, size=n_tokens - len(ids) + 8))
        ids = tok.encode(" ".join(words), add_special_tokens=False)
    return tok.decode(ids[:n_tokens]).strip()


def _template_overhead(tok: Any) -> int:
    return prompt_tokens(tok, [{"role": "user", "content": ""}])


def _lognormal_int(rng: np.random.Generator, median: float, sigma: float, lo: int, hi: int) -> int:
    return int(np.clip(round(rng.lognormal(np.log(median), sigma)), lo, hi))


def _single_turn_sessions(turns: list[Turn]) -> list[Session]:
    return [Session(session_id=i, turns=[t]) for i, t in enumerate(turns)]


# --------------------------------------------------------------------------- ShareGPT

_SHAREGPT_CACHE: dict[str, Any] = {}


def load_sharegpt_pairs() -> list[tuple[str, str]]:
    """(first human message, first gpt reply) pairs, loaded once per process (a failure is
    cached too, so sweeps don't retry a large download on every run)."""
    if "pairs" not in _SHAREGPT_CACHE and "error" not in _SHAREGPT_CACHE:
        try:
            _SHAREGPT_CACHE["pairs"] = _load_sharegpt_pairs_uncached()
        except Exception as exc:
            _SHAREGPT_CACHE["error"] = exc
    if "error" in _SHAREGPT_CACHE:
        raise _SHAREGPT_CACHE["error"]
    return _SHAREGPT_CACHE["pairs"]


def _load_sharegpt_pairs_uncached() -> list[tuple[str, str]]:
    """Tries ``datasets`` (HF-cached), then a direct hub download of the JSON (also HF-cached).
    Raises if neither works."""
    errors = []
    try:
        import datasets

        ds = datasets.load_dataset(SHAREGPT_REPO, data_files=SHAREGPT_FILE, split="train")
        rows: Any = (r["conversations"] for r in ds)
    except Exception as exc:  # schema/offline/missing package
        errors.append(f"datasets: {type(exc).__name__}: {exc}")
        try:
            import json

            from huggingface_hub import hf_hub_download

            path = hf_hub_download(SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset")
            with open(path) as f:
                rows = (r.get("conversations", []) for r in json.load(f))
        except Exception as exc2:
            errors.append(f"hub: {type(exc2).__name__}: {exc2}")
            raise RuntimeError("; ".join(errors)) from exc2
    pairs = []
    for conv in rows:
        if (len(conv) >= 2 and conv[0].get("from") == "human" and conv[1].get("from") == "gpt"
                and conv[0].get("value") and conv[1].get("value")):
            pairs.append((conv[0]["value"], conv[1]["value"]))
    if not pairs:
        raise RuntimeError("ShareGPT loaded but contained no usable conversations")
    return pairs


def build_chat(tok: Any, num_requests: int, rng: np.random.Generator, max_model_len: int,
               sharegpt: str = "auto",
               pair_loader: Callable[[], list[tuple[str, str]]] = load_sharegpt_pairs) -> Trace:
    max_prompt, max_total = 1024, min(2048, max_model_len)
    reason = "disabled (--sharegpt off)"
    if sharegpt != "off":
        try:
            pairs = pair_loader()
            order = rng.permutation(len(pairs))
            turns: list[Turn] = []
            for idx in order:
                prompt, reply = pairs[int(idx)]
                msgs = [{"role": "user", "content": prompt}]
                pt = prompt_tokens(tok, msgs)
                out = text_tokens(tok, reply)
                if pt < 4 or out < 4 or pt > max_prompt or pt + out > max_total:
                    continue
                turns.append(Turn(messages=msgs, max_tokens=out, prompt_tokens=pt))
                if len(turns) == num_requests:
                    break
            if len(turns) < num_requests:
                raise RuntimeError(f"only {len(turns)} ShareGPT conversations pass the filter")
            return Trace("chat", _single_turn_sessions(turns), "sharegpt",
                         {"sharegpt_pairs_available": len(pairs)})
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"[:500]
    # Lognormal fallback with roughly ShareGPT-like medians.
    turns = []
    overhead = _template_overhead(tok)
    while len(turns) < num_requests:
        p = _lognormal_int(rng, 150, 1.0, 4, max_prompt)
        o = _lognormal_int(rng, 150, 0.9, 4, 1024)
        if p + o > max_total:
            continue
        msgs = [{"role": "user", "content": make_text(tok, max(1, p - overhead), rng)}]
        turns.append(Turn(messages=msgs, max_tokens=o, prompt_tokens=prompt_tokens(tok, msgs)))
    return Trace("chat", _single_turn_sessions(turns), "synthetic",
                 {"fallback_reason": reason, "synthetic": "lognormal(median 150, sigma 1.0/0.9)"})


# --------------------------------------------------------------------------- other traces

def build_long_context(tok: Any, num_requests: int, rng: np.random.Generator,
                       max_model_len: int, prompt_range: tuple[int, int] = (2000, 3500),
                       output_range: tuple[int, int] = (32, 128)) -> Trace:
    overhead = _template_overhead(tok)
    turns = []
    for _ in range(num_requests):
        target = int(rng.integers(prompt_range[0], prompt_range[1] + 1))
        out = int(rng.integers(output_range[0], output_range[1] + 1))
        msgs = [{"role": "user", "content": make_text(tok, target - overhead, rng)}]
        pt = prompt_tokens(tok, msgs)
        if pt + out > max_model_len:  # BPE drift: trim the output rather than overflow
            out = max_model_len - pt
            if out < 1:
                raise ValueError(f"prompt of {pt} tokens does not fit max_model_len={max_model_len}")
        turns.append(Turn(messages=msgs, max_tokens=out, prompt_tokens=pt))
    return Trace("long_context", _single_turn_sessions(turns), "synthetic",
                 {"prompt_range": list(prompt_range), "output_range": list(output_range)})


def build_multi_turn(tok: Any, num_requests: int, rng: np.random.Generator, max_model_len: int,
                     turns_range: tuple[int, int] = (3, 6), think_time_mean_s: float = 1.0,
                     user_median: int = 80, reply_median: int = 120) -> Trace:
    sessions: list[Session] = []
    total = 0
    truncated = 0
    while total < num_requests:
        n_turns = int(rng.integers(turns_range[0], turns_range[1] + 1))
        history: list[Message] = []
        turns: list[Turn] = []
        for t in range(min(n_turns, num_requests - total)):
            user = {"role": "user", "content": make_text(
                tok, _lognormal_int(rng, user_median, 0.6, 8, 400), rng)}
            reply_len = _lognormal_int(rng, reply_median, 0.6, 16, 400)
            msgs = history + [user]
            pt = prompt_tokens(tok, msgs)
            if pt + reply_len > max_model_len:
                truncated += 1
                break
            think = 0.0 if t == 0 else float(rng.exponential(think_time_mean_s))
            turns.append(Turn(messages=msgs, max_tokens=reply_len, prompt_tokens=pt,
                              think_time_s=think))
            # Stand-in for the server's reply: same token length, deterministic text.
            history = msgs + [{"role": "assistant", "content": make_text(tok, reply_len, rng)}]
        if turns:
            sessions.append(Session(session_id=len(sessions), turns=turns))
            total += len(turns)
    return Trace("multi_turn", sessions, "synthetic",
                 {"turns_range": list(turns_range), "think_time_mean_s": think_time_mean_s,
                  "sessions_truncated_by_max_model_len": truncated})


def build_fixed(tok: Any, num_requests: int, rng: np.random.Generator, max_model_len: int,
                input_len: int = 512, output_len: int = 128) -> Trace:
    """``input_len`` is the user-content length (the chat template is added on top, as with
    ``vllm bench serve --dataset-name random --backend openai-chat``)."""
    turns = []
    for _ in range(num_requests):
        msgs = [{"role": "user", "content": make_text(tok, input_len, rng)}]
        pt = prompt_tokens(tok, msgs)
        if pt + output_len > max_model_len:
            raise ValueError(f"{pt}+{output_len} tokens exceeds max_model_len={max_model_len}")
        turns.append(Turn(messages=msgs, max_tokens=output_len, prompt_tokens=pt))
    return Trace("fixed", _single_turn_sessions(turns), "synthetic",
                 {"input_len": input_len, "output_len": output_len})


def build_trace(name: str, tok: Any, num_requests: int, seed: int = 0,
                max_model_len: int = 4096, **opts: Any) -> Trace:
    rng = np.random.default_rng(seed)
    builders: dict[str, Callable[..., Trace]] = {
        "chat": build_chat, "long_context": build_long_context,
        "multi_turn": build_multi_turn, "fixed": build_fixed,
    }
    if name not in builders:
        raise ValueError(f"unknown trace {name!r}; choose from {TRACES}")
    trace = builders[name](tok, num_requests, rng, max_model_len, **opts)
    trace.meta.update({"seed": seed, "max_model_len": max_model_len})
    return trace
