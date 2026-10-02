"""Tokenizer loading (lazy) and prompt-length measurement.

Lengths are measured with the served model's tokenizer, including its chat template, so the
recorded prompt_tokens match what the server prefills. Tests use ``StubTokenizer``.
"""

from __future__ import annotations

import functools
from typing import Any, Protocol, Sequence

Message = dict[str, str]


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class StubTokenizer:
    """Whitespace tokenizer for CPU tests: one token per word, exact round trip.

    No chat template: a prompt's length is the sum of its messages' word counts.
    """

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._words: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out = []
        for w in text.split():
            if w not in self._ids:
                self._ids[w] = len(self._words)
                self._words.append(w)
            out.append(self._ids[w])
        return out

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self._words[i] for i in ids)


@functools.lru_cache(maxsize=4)
def load_tokenizer(name: str) -> Any:
    """``"stub"`` -> StubTokenizer; anything else -> transformers AutoTokenizer (imported lazily)."""
    if name == "stub":
        return StubTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def text_tokens(tok: Any, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def prompt_tokens(tok: Any, messages: Sequence[Message]) -> int:
    """Tokens the server prefills for ``messages``: chat template + generation prompt if the
    tokenizer has one, else the sum of message token counts."""
    if getattr(tok, "chat_template", None) and hasattr(tok, "apply_chat_template"):
        ids = tok.apply_chat_template(list(messages), tokenize=True, add_generation_prompt=True)
        if isinstance(ids, dict) or hasattr(ids, "keys"):  # newer transformers return a BatchEncoding
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids)
    return sum(text_tokens(tok, m["content"]) for m in messages)
