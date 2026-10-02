"""KV capture plumbing on CPU: transformers cache access, statistics, markdown, file format and
loading into the paged layout (tiny synthetic tensors; no model download)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bench.common.results import save_result
from bench.kernels.kv_capture import (
    FP8_MAX, cache_layer_kv, head_stats, layer_stats, load_captured_paged, load_layer,
    read_manifest, save_layer, stats_markdown, write_manifest,
)
from kvcache.reference.decode_attention import dense_decode_attention_sdpa, paged_decode_attention


def test_cache_layer_kv_across_api_versions() -> None:
    k = [torch.full((1, 2, 3, 4), float(i)) for i in range(2)]
    v = [t + 10 for t in k]
    new = SimpleNamespace(layers=[SimpleNamespace(keys=k[i], values=v[i]) for i in range(2)])
    old = SimpleNamespace(key_cache=k, value_cache=v)
    legacy = tuple((k[i], v[i]) for i in range(2))
    for cache in (new, old, legacy):
        kk, vv = cache_layer_kv(cache, 1)
        assert torch.equal(kk, k[1]) and torch.equal(vv, v[1])


def _synthetic(n_prompts=3, n=20, h=2, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    chunks = [torch.randn(n, h, d, generator=g) for _ in range(n_prompts)]
    for c in chunks:
        c[:, 0, 7] *= 50.0     # head 0: one dominant channel
        c[0, 1, 3] = 200.0     # head 1: first-token outlier only
    return chunks


def test_head_stats_dominance_and_first_token() -> None:
    chunks = _synthetic()
    s0, s1 = head_stats(chunks)
    x = torch.cat(chunks)
    assert s0["amax"] == pytest.approx(float(x[:, 0].abs().max()))
    assert s0["top_channel"] == 7 and s0["top_median_ratio"] > 8 and s0["dominated"]
    assert s0["top4_energy"] > 0.9
    assert s1["amax"] == 200.0 and s1["top_channel"] == 3
    assert s1["amax_excl_first"] < 10  # outlier only in each prompt's first token
    assert s1["rms"] == pytest.approx(float(x[:, 1].pow(2).mean().sqrt()))


def test_head_stats_subnormal_fractions() -> None:
    # Head amax 448 -> scale 1 -> subnormal below 2^-6. 4 of 8 nonzero values are below.
    vals = torch.tensor([448.0, 1.0, 0.5, 0.1, 2**-7, 2**-8, 2**-10, 2**-12, 0.0])
    t = vals.view(-1, 1, 1)
    [s] = head_stats([t])
    assert s["subnormal_frac_head_scale"] == pytest.approx(4 / 8)
    # With a 64x larger tensor-level scale, the threshold is 1.0 -> 6 of 8 (0.5, 0.1 and the 4).
    [s2] = head_stats([t], tensor_amax=448.0 * 64)
    assert s2["subnormal_frac_tensor_scale"] == pytest.approx(6 / 8)
    assert FP8_MAX == 448.0


def test_layer_stats_rows_and_tensor_scale() -> None:
    ks, vs = _synthetic(), _synthetic(seed=1)
    rows = layer_stats(5, ks, vs)
    assert [(r["kind"], r["head"]) for r in rows] == [("K", 0), ("K", 1), ("V", 0), ("V", 1)]
    assert all(r["layer"] == 5 for r in rows)
    # The head with the smaller amax underflows at least as much with the per-tensor scale.
    k1 = rows[1] if rows[1]["amax"] < rows[0]["amax"] else rows[0]
    assert k1["subnormal_frac_tensor_scale"] >= k1["subnormal_frac_head_scale"]


def _write_capture(data_dir: Path, layers=(0, 1, 2), lens=(1, 17, 40), hq=4, hkv=2, d=16):
    g = torch.Generator().manual_seed(3)
    files = []
    for layer in layers:
        keys = [torch.randn(n, hkv, d, generator=g).half() for n in lens]
        values = [torch.randn(n, hkv, d, generator=g).half() for n in lens]
        queries = [torch.randn(hq, d, generator=g).half() for _ in lens]
        p = save_layer(data_dir, layer, {"model": "fake", "layer": layer, "prompts": ["a", "b", "c"],
                                         "sm_scale": d ** -0.5, "num_layers": 3, "num_q_heads": hq,
                                         "num_kv_heads": hkv, "head_dim": d, "keys": keys,
                                         "values": values, "queries": queries,
                                         "check_max_abs_err": [1e-3] * len(lens)})
        files.append(p.name)
    write_manifest(data_dir, {"model": "fake", "layers": list(layers), "files": files,
                              "prompts": ["a", "b", "c"], "prompt_tokens": [n - 1 for n in lens]})
    return files


def test_files_roundtrip_and_paged_loading(tmp_path: Path) -> None:
    files = _write_capture(tmp_path)
    assert read_manifest(tmp_path)["files"] == files and read_manifest(tmp_path / "nope") is None
    payload = load_layer(1, tmp_path)
    assert payload["keys"][1].dtype == torch.float16 and payload["keys"][1].shape == (17, 2, 16)
    cache, q, _ = load_captured_paged(1, 16, data_dir=tmp_path)
    assert q.shape == (3, 4, 16) and cache.context_lens.tolist() == [1, 17, 40]
    out = paged_decode_attention(q, cache.key_cache, cache.value_cache, cache.block_tables,
                                 cache.context_lens, sm_scale=payload["sm_scale"])
    ref = dense_decode_attention_sdpa(q, payload["keys"], payload["values"], sm_scale=payload["sm_scale"])
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    c8, q8, _ = load_captured_paged(1, 16, kv_dtype="fp8", prompts=[2], data_dir=tmp_path)
    assert c8.is_fp8 and c8.k_scale.shape == (2,) and q8.shape == (1, 4, 16)


def test_stats_markdown_from_saved_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KVCACHE_RESULTS_DIR", str(tmp_path))
    rows = [r for L in range(3) for r in layer_stats(L, _synthetic(seed=L), _synthetic(seed=L + 9))]
    cfg = {"model": "fake/model", "num_layers": 3, "num_q_heads": 4, "num_kv_heads": 2,
           "head_dim": 16, "saved_layers": [0, 2], "prompts": ["a", "b", "c"],
           "prompt_tokens": [19, 19, 19], "total_tokens": 60, "transformers_version": "x"}
    meta = {"git": {"sha": "abcdef123456", "dirty": False}, "gpu": {"name": "G"}, "packages": {"torch": "t"}}
    path = save_result("kv_stats", cfg, {"rows": rows, "capture_check": {"0": [1e-3, 2e-3], "2": [5e-4]},
                                         "files": {"kv_layer00.pt": 3 * 2**20}}, None, metadata=meta)
    md = stats_markdown(json.loads(path.read_text()))
    assert "| layer | K h0 | K h1 | V h0 | V h1 |" in md
    assert "| 0 * |" in md and "| 1 |" in md and "| 2 * |" in md
    detail = [l for l in md.splitlines() if l.startswith("| 0 | K | 0 |")]
    assert detail and "| 7 |" in detail[0] and "| yes |" in detail[0]
    assert not any(l.startswith("| 1 | K |") for l in md.splitlines())  # unsaved layer: no detail
    assert "| 0 | 0.002 | 0.0015 |" in md
    assert "kv_layer00.pt (3.0 MiB)" in md and "abcdef1234" in md


def test_capture_script_without_cuda_exits_cleanly(capsys) -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA available; this checks the CPU path")
    from scripts import capture_kv

    assert capture_kv.main([]) == 0
    assert "No CUDA device available" in capsys.readouterr().out
    names = [n for n, _ in capture_kv.PROMPTS]
    assert len(names) == len(set(names)) == 18  # + 2 long documents = 20 prompts


# --------------------------------------------------------------------------- real captured data

@pytest.mark.gpu
def test_captured_kv_reference_matches_sdpa(captured_kv_dir) -> None:
    data_dir, manifest = captured_kv_dir
    for layer in manifest["layers"]:
        cache, q, payload = load_captured_paged(layer, 16, device="cuda", data_dir=data_dir)
        out = paged_decode_attention(q, cache.key_cache, cache.value_cache, cache.block_tables,
                                     cache.context_lens, sm_scale=payload["sm_scale"])
        ref = dense_decode_attention_sdpa(q, [k.cuda() for k in payload["keys"]],
                                          [v.cuda() for v in payload["values"]],
                                          sm_scale=payload["sm_scale"])
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
        assert max(payload["check_max_abs_err"]) < 2e-2  # capture matched the model's attention


@pytest.mark.gpu
def test_captured_kv_fp8_cache_builds(captured_kv_dir) -> None:
    data_dir, manifest = captured_kv_dir
    cache, q, payload = load_captured_paged(manifest["layers"][-1], 16, kv_dtype="fp8",
                                            device="cuda", data_dir=data_dir)
    out = paged_decode_attention(q, cache.key_cache, cache.value_cache, cache.block_tables,
                                 cache.context_lens, sm_scale=payload["sm_scale"],
                                 k_scale=cache.k_scale, v_scale=cache.v_scale)
    assert torch.isfinite(out).all() and cache.k_scale.shape == (payload["num_kv_heads"],)


# --------------------------------------------------------------------------- tiny real model

def test_capture_path_on_tiny_random_qwen2() -> None:
    """The capture path against the installed transformers with a tiny random Qwen2 (no
    download): cache API, post-RoPE query rebuild and the attention self-check."""
    transformers = pytest.importorskip("transformers")
    from scripts import capture_kv

    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=256)
    model = transformers.Qwen2ForCausalLM(cfg).eval().float()
    with torch.no_grad():  # random q/k biases so RoPE and bias handling matter
        for layer in model.model.layers:
            layer.self_attn.q_proj.bias.normal_()
            layer.self_attn.k_proj.bias.normal_()
    facts = capture_kv.model_facts(model)
    assert facts == {"num_layers": 3, "num_q_heads": 4, "num_kv_heads": 2, "head_dim": 16}
    hooks = capture_kv.DecodeHooks(model)
    ids = torch.randint(0, 128, (1, 37), generator=torch.Generator().manual_seed(1))
    kv, q, attn_out, n = capture_kv.capture_ids(model, ids, hooks, 3)
    hooks.remove()
    assert n == 37 and len(kv) == 3 and set(q) == set(attn_out) == {0, 1, 2}
    for layer in range(3):
        k, v = kv[layer]
        assert k.shape == (38, 2, 16) and k.dtype == torch.float16  # prompt + decode token
        assert q[layer].shape == (4, 16)
        err = capture_kv.self_check(q[layer].float(), k, v, attn_out[layer], 16 ** -0.5)
        assert err < 2e-2, f"layer {layer}: {err}"  # fp16-stored K/V vs fp32 model
    # Negative control: skipping RoPE on q must break the check (keys are post-RoPE).
    hooks2 = capture_kv.DecodeHooks(model)
    hooks2.rope = lambda q, k, cos, sin: (q, k)
    _, q_bad, attn2, _ = capture_kv.capture_ids(model, ids, hooks2, 3)
    hooks2.remove()
    k, v = kv[1]
    assert capture_kv.self_check(q_bad[1].float(), k, v, attn2[1], 16 ** -0.5) > 0.05
