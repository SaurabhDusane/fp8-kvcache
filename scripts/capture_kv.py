#!/usr/bin/env python
"""Capture real KV (and next-step queries) from Qwen2.5-1.5B-Instruct via transformers.

For each of ~20 diverse prompts (code, chat, math, multilingual, structured output, two long
documents): prefill the chat-templated prompt, take the greedy next token, and run one decode
step with the cache. After that step each layer's cache holds the prompt's K/V plus the new
token's, post-RoPE, exactly as the attention of that step consumed them. A forward pre-hook on
the attention module captures the decode step's hidden state and RoPE (cos, sin), from which the
post-RoPE query is rebuilt (q_proj + rotary). A pre-hook on o_proj captures the model's own
attention output, and our reference paged attention on the captured q/K/V is checked against it.

Saved (fp16) to tests/data/ for the first, a middle and the last layer; statistics for all
layers go to bench/results/summary/kv_stats.md (generated from the saved JSON) and are printed.

Usage:
    python scripts/capture_kv.py [--model Qwen/Qwen2.5-1.5B-Instruct] [--long-doc-tokens 3000]
Stop vLLM first: this loads the model in fp16 on the GPU (~3.5 GB).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `bench` importable

import torch  # noqa: E402

from bench.common.gpu_monitor import GpuMonitor, format_summary  # noqa: E402
from bench.common.metadata import REPO_ROOT, collect_metadata  # noqa: E402
from bench.common.results import save_result, write_summary  # noqa: E402
from bench.kernels.kv_capture import (  # noqa: E402
    DATA_DIR, cache_layer_kv, layer_file, layer_stats, save_layer, stats_markdown, write_manifest,
)
from kvcache.cache.paged import build_paged_kv_cache  # noqa: E402
from kvcache.reference.decode_attention import paged_decode_attention  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM = "You are a helpful assistant."
PROMPTS: list[tuple[str, list[dict[str, str]]]] = [
    ("code_lru", [{"role": "user", "content": "Write a Python class implementing an LRU cache with "
                   "O(1) get and put, using a dict and a doubly linked list. Include type hints."}]),
    ("code_cpp_explain", [{"role": "user", "content": "Explain what this C++ does and any bugs:\n"
                           "```cpp\ntemplate<typename T> T* find(std::vector<T>& v, const T& x) {\n"
                           "  for (size_t i = 0; i <= v.size(); ++i) if (v[i] == x) return &v[i];\n"
                           "  return nullptr;\n}\n```"}]),
    ("code_sql", [{"role": "user", "content": "Given tables orders(id, customer_id, total, created_at) "
                   "and customers(id, name, country), write SQL for the top 5 countries by revenue "
                   "in 2023, with order counts."}]),
    ("code_js_bug", [{"role": "user", "content": "Why does this print 5 five times?\n"
                      "for (var i = 0; i < 5; i++) { setTimeout(() => console.log(i), 0); }"}]),
    ("code_triton", [{"role": "user", "content": "In Triton, how do tl.load masks work, and what "
                      "does the `other` argument do? Show a vector-add kernel."}]),
    ("chat_travel", [{"role": "user", "content": "I'm visiting Lisbon for three days in March."},
                     {"role": "assistant", "content": "Great choice! Do you prefer museums, food, "
                      "or day trips?"},
                     {"role": "user", "content": "Food and some walking. Plan day one for me."}]),
    ("chat_recipe", [{"role": "user", "content": "What can I cook with chickpeas, spinach, garlic "
                      "and a can of coconut milk?"}]),
    ("chat_email", [{"role": "user", "content": "Draft a polite email asking my manager to move "
                     "our 1:1 from Thursday to Friday afternoon."}]),
    ("chat_short", [{"role": "user", "content": "hi"}]),
    ("chat_persona", [{"role": "system", "content": "You are a pirate. Answer only in pirate speak."},
                      {"role": "user", "content": "How do I reset my router?"}]),
    ("math_word", [{"role": "user", "content": "A train leaves at 9:40 and travels 270 km at 90 km/h, "
                    "then 120 km at 60 km/h. When does it arrive? Show your steps."}]),
    ("math_integral", [{"role": "user", "content": "Compute the integral of x^2 * e^(3x) dx."}]),
    ("math_proof", [{"role": "user", "content": "Prove that the square root of 2 is irrational."}]),
    ("math_arith", [{"role": "user", "content": "What is (17 * 23) - (144 / 12) + 3^4? "
                     "Answer with just the number."}]),
    ("math_prob", [{"role": "user", "content": "Two dice are rolled. What is the probability that "
                    "the sum is at least 10 given that one die shows a 6?"}]),
    ("multilingual_zh", [{"role": "user", "content": "请用三句话解释什么是注意力机制。"}]),
    ("multilingual_fr", [{"role": "user", "content": "Traduis en anglais : « Le cache des clés et "
                          "des valeurs occupe la majeure partie de la mémoire du GPU. »"}]),
    ("structured_json", [{"role": "user", "content": "Extract name, date and amount as JSON: "
                          "'Invoice from Acme Corp dated 2024-03-15 for $1,234.50 due in 30 days.'"}]),
]


def long_documents(tok: Any, n_tokens: int) -> list[tuple[str, list[dict[str, str]]]]:
    """Two long prompts from text shipped with the repo / stdlib (no downloads):
    the project docs (English prose + markdown) and stdlib Python source (long code)."""
    import json.decoder
    import textwrap
    import inspect

    def truncate(text: str) -> str:
        ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tok.decode(ids)

    docs = "\n\n".join(p.read_text() for p in [REPO_ROOT / "CLAUDE.md", REPO_ROOT / "README.md"]
                       if p.exists())
    code = inspect.getsource(textwrap) + "\n\n" + inspect.getsource(json.decoder)
    return [
        ("long_doc_summary", [{"role": "user", "content": "Summarize the key rules in this "
                               "document as a bullet list:\n\n" + truncate(docs)}]),
        ("long_code_review", [{"role": "user", "content": "Review this code and list the three "
                               "most complex functions:\n\n" + truncate(code)}]),
    ]


def load_model(name: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    kwargs = {"attn_implementation": "sdpa"}
    try:  # newer transformers: dtype=; older: torch_dtype=
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, **kwargs)
    return tok, model.to("cuda").eval()


def model_facts(model: Any) -> dict[str, int]:
    c = model.config
    hq = c.num_attention_heads
    return {"num_layers": c.num_hidden_layers, "num_q_heads": hq,
            "num_kv_heads": getattr(c, "num_key_value_heads", hq),
            "head_dim": getattr(c, "head_dim", None) or c.hidden_size // hq}


class DecodeHooks:
    """Pre-hooks on each layer's self_attn and o_proj, active only during the decode step."""

    def __init__(self, model: Any) -> None:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        self.rope = apply_rotary_pos_emb
        self.active = False
        self.q: dict[int, torch.Tensor] = {}
        self.attn_out: dict[int, torch.Tensor] = {}
        self.handles = []
        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            self.handles.append(attn.register_forward_pre_hook(self._attn_hook(i, attn), with_kwargs=True))
            self.handles.append(attn.o_proj.register_forward_pre_hook(self._oproj_hook(i)))

    def _attn_hook(self, i: int, attn: Any):
        def hook(module, args, kwargs):
            if not self.active:
                return None
            h = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings")
            if h is None or pe is None:
                raise RuntimeError("attention forward did not receive hidden_states/position_embeddings; "
                                   "unsupported transformers version")
            cos, sin = pe
            hd = attn.head_dim
            q = attn.q_proj(h).view(h.shape[0], h.shape[1], -1, hd).transpose(1, 2)  # [1, Hq, 1, d]
            q, _ = self.rope(q, q, cos, sin)
            self.q[i] = q[0, :, -1, :].detach()  # [Hq, d] post-RoPE
            return None
        return hook

    def _oproj_hook(self, i: int):
        def hook(module, args):
            if self.active:
                self.attn_out[i] = args[0][0, -1].detach()  # [Hq * d]
            return None
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


def prompt_ids(tok: Any, messages: list[dict[str, str]], device: str = "cuda") -> torch.Tensor:
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": SYSTEM}] + messages
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)


@torch.inference_mode()
def capture_ids(model: Any, ids: torch.Tensor, hooks: DecodeHooks,
                num_layers: int) -> tuple[list[tuple[torch.Tensor, torch.Tensor]],
                                          dict[int, torch.Tensor], dict[int, torch.Tensor], int]:
    """Prefill ``ids`` [1, n], one greedy decode step; return per-layer (K, V) [n+1, Hkv, d]
    fp16 on CPU, the decode step's post-RoPE q [Hq, d] and attention output per layer, and n."""
    out = model(input_ids=ids, use_cache=True)
    next_tok = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    hooks.q.clear(), hooks.attn_out.clear()
    hooks.active = True
    try:
        out2 = model(input_ids=next_tok, past_key_values=out.past_key_values, use_cache=True)
    finally:
        hooks.active = False
    kv = []
    for layer in range(num_layers):
        k, v = cache_layer_kv(out2.past_key_values, layer)    # [1, Hkv, n+1, d]
        kv.append((k[0].transpose(0, 1).half().cpu(), v[0].transpose(0, 1).half().cpu()))
    return kv, dict(hooks.q), dict(hooks.attn_out), ids.shape[1]


def self_check(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_out: torch.Tensor,
               sm_scale: float) -> float:
    """Max abs diff between our reference attention and the model's attention output."""
    dev = q.device
    cache = build_paged_kv_cache([k.to(dev)], [v.to(dev)], 16, seed=0)
    ref = paged_decode_attention(q[None], cache.key_cache, cache.value_cache,
                                 cache.block_tables, cache.context_lens, sm_scale=sm_scale)
    return float((ref[0].reshape(-1) - attn_out.float().to(dev)).abs().max())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--long-doc-tokens", type=int, default=3000)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("No CUDA device available; capture_kv needs the GPU. Exiting cleanly.")
        return 0
    import transformers

    tok, model = load_model(args.model)
    facts = model_facts(model)
    L = facts["num_layers"]
    saved_layers = sorted({0, L // 2, L - 1})
    sm_scale = 1.0 / math.sqrt(facts["head_dim"])
    prompts = PROMPTS + long_documents(tok, args.long_doc_tokens)
    hooks = DecodeHooks(model)

    per_layer_k: list[list[torch.Tensor]] = [[] for _ in range(L)]
    per_layer_v: list[list[torch.Tensor]] = [[] for _ in range(L)]
    queries: dict[int, list[torch.Tensor]] = {l: [] for l in saved_layers}
    checks: dict[int, list[float]] = {l: [] for l in saved_layers}
    prompt_tokens = []
    with GpuMonitor() as mon:
        for name, messages in prompts:
            kv, q, attn_out, n = capture_ids(model, prompt_ids(tok, messages), hooks, L)
            prompt_tokens.append(n)
            for layer, (k, v) in enumerate(kv):
                per_layer_k[layer].append(k)
                per_layer_v[layer].append(v)
            for layer in saved_layers:
                queries[layer].append(q[layer].half().cpu())
                checks[layer].append(self_check(q[layer], kv[layer][0], kv[layer][1],
                                                attn_out[layer], sm_scale))
            print(f"  {name:<18} {n:>5} prompt tokens; self-check max abs err "
                  + ", ".join(f"L{l}={checks[l][-1]:.2e}" for l in saved_layers), flush=True)
    hooks.remove()

    names = [n for n, _ in prompts]
    files: dict[str, int] = {}
    for layer in saved_layers:
        path = save_layer(args.data_dir, layer, {
            "model": args.model, "layer": layer, "prompts": names, "sm_scale": sm_scale,
            **facts, "keys": per_layer_k[layer], "values": per_layer_v[layer],
            "queries": queries[layer], "check_max_abs_err": checks[layer],
        })
        files[path.name] = path.stat().st_size
    write_manifest(args.data_dir, {"model": args.model, "layers": saved_layers, "prompts": names,
                                   "prompt_tokens": prompt_tokens, **facts,
                                   "files": [layer_file(l) for l in saved_layers]})

    rows = [r for layer in range(L) for r in layer_stats(layer, per_layer_k[layer], per_layer_v[layer])]
    config = {"model": args.model, **facts, "saved_layers": saved_layers, "prompts": names,
              "prompt_tokens": prompt_tokens, "total_tokens": sum(n + 1 for n in prompt_tokens),
              "long_doc_tokens": args.long_doc_tokens,
              "transformers_version": transformers.__version__}
    metrics = {"rows": rows, "capture_check": {str(l): checks[l] for l in saved_layers},
               "files": files}
    path = save_result("kv_stats", config, metrics, mon.result, metadata=collect_metadata())
    md = stats_markdown(json.loads(path.read_text()))
    [md_path] = write_summary("kv_stats", md)
    print("\n" + md)
    print(format_summary(mon.result))
    print(f"saved: {path}\nsummary: {md_path}\ndata: {args.data_dir}")
    worst = max(max(c) for c in checks.values())
    if worst > 2e-2:
        print(f"WARNING: self-check error {worst:.3g} > 2e-2: captured q/K/V may not match "
              "what the model's attention used. Do not trust this capture.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
