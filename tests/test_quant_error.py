"""quant_error.py on CPU with a synthetic capture (no model, no GPU)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bench.kernels import quant_error as qe
from tests.test_kv_capture import _write_capture


def test_compare_outputs_exact() -> None:
    ref = torch.tensor([[[1.0, 0.0], [0.0, 2.0]], [[3.0, 4.0], [0.0, 1.0]]])
    m = qe.compare_outputs(ref, ref.clone())
    assert m["max_abs"] == 0 and m["mean_abs"] == 0 and m["cosine"] == pytest.approx(1.0)
    test = ref.clone()
    test[1, 0, 0] += 0.5
    m = qe.compare_outputs(ref, test)
    assert m["max_abs"] == 0.5 and m["mean_abs"] == pytest.approx(0.5 / 8)
    assert m["max_abs_rel"] == pytest.approx(0.5 / 4)
    assert m["min_prompt_index"] == 1 and m["min_prompt_cosine"] < 1.0 <= m["cosine"] + 0.01
    assert m["ref_mean_abs"] == pytest.approx(ref.abs().mean().item())


def test_quant_error_cli_on_synthetic_capture(tmp_path: Path, monkeypatch, capsys) -> None:
    data = tmp_path / "data"
    _write_capture(data, layers=(0, 2), lens=(1, 17, 40), hq=4, hkv=2, d=16)
    monkeypatch.setenv("KVCACHE_RESULTS_DIR", str(tmp_path / "results"))
    assert qe.main(["--data-dir", str(data), "--device", "cpu", "--granularities", "all"]) == 0
    out = capsys.readouterr().out
    md = (tmp_path / "results" / "summary" / "quant_error.md").read_text()
    assert out.startswith("# FP8 KV-cache quantization error on real KV")
    rows = [l for l in md.splitlines() if l.startswith("| 0 |") or l.startswith("| 2 |")]
    assert len(rows) == 2 * 4  # 2 layers x 4 granularities
    assert "Reference implementation: pure quantization error." in md
    assert "layer 0 kv_head: k_scale [" in md
    [raw] = list((tmp_path / "results" / "raw").glob("*/quant_error_*.json"))
    import json

    rec = json.loads(raw.read_text())
    for r in rec["metrics"]["rows"]:
        assert 0 < r["max_abs"] < 0.5 and r["mean_abs"] <= r["max_abs"]
        assert r["cosine"] > 0.99 and r["num_prompts"] == 3 and r["tokens"] == 58


def test_quant_error_without_capture(tmp_path: Path, capsys) -> None:
    assert qe.main(["--data-dir", str(tmp_path)]) == 1
    assert "Run scripts/capture_kv.py" in capsys.readouterr().out


def test_quant_error_kernel_impl_selection(monkeypatch) -> None:
    from kvcache import kernels as K

    with pytest.raises(KeyError):
        qe.get_impl("no_such_kernel")
    K.register_kernel("zz_fp16_only")(lambda *a, **k: None)
    try:
        with pytest.raises(SystemExit, match="does not support fp8"):
            qe.get_impl("zz_fp16_only")
    finally:
        K.unregister_kernel("zz_fp16_only")
    fn, grans = qe.get_impl("reference")
    assert "kv_head" in grans and callable(fn)
