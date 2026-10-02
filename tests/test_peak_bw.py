"""peak_bw: Triton bandwidth kernels under the interpreter, byte accounting, summary generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.common.results import save_result
from scripts import peak_bw as pb

torch = pytest.importorskip("torch")
pytest.importorskip("triton")


@pytest.fixture
def interp_kernels():
    from bench.kernels import bw_kernels

    return bw_kernels


@pytest.mark.interpreter
@pytest.mark.parametrize("n, block", [(1000, 64), (4096, 64), (1, 64), (300, 128)])
def test_triton_copy_interpreter(interp_kernels, n: int, block: int) -> None:
    g = torch.Generator().manual_seed(0)
    src = torch.randn(n, generator=g).to(torch.float16)
    dst = torch.full_like(src, -7.0)
    interp_kernels.triton_copy(src, dst, block=block)
    assert torch.equal(dst, src)  # bit-exact, including the masked tail


@pytest.mark.interpreter
@pytest.mark.parametrize("n, block, programs", [
    (1000, 64, 4),     # grid-stride: 16 blocks over 4 programs, masked tail
    (4096, 64, 64),    # one block per program
    (100, 64, 8),      # fewer blocks (2) than partial slots (8)
    (5000, 128, 3),    # uneven blocks per program
])
def test_triton_reduce_interpreter(interp_kernels, n: int, block: int, programs: int) -> None:
    # Small integers are exact in fp16 and their sums exact in fp32 -> exact comparison, so a
    # dropped or double-counted element/block cannot hide inside a tolerance.
    g = torch.Generator().manual_seed(1)
    src = torch.randint(0, 8, (n,), generator=g).to(torch.float16)
    partials = torch.full((programs,), float("nan"), dtype=torch.float32)
    used = interp_kernels.triton_reduce_sum(src, partials, block=block)
    assert used.numel() == min(programs, -(-n // block))
    assert used.sum().item() == src.to(torch.float64).sum().item()


@pytest.mark.interpreter
def test_triton_reduce_interpreter_random(interp_kernels) -> None:
    src = torch.rand(10_000, generator=torch.Generator().manual_seed(2)).to(torch.float16)
    partials = torch.empty(7, dtype=torch.float32)
    got = interp_kernels.triton_reduce_sum(src, partials, block=256).double().sum().item()
    assert got == pytest.approx(src.double().sum().item(), rel=1e-6)


def test_bytes_moved() -> None:
    size = 256 * pb.MIB
    assert pb.bytes_moved("torch_copy", size) == 2 * size
    assert pb.bytes_moved("triton_copy", size) == 2 * size
    assert pb.bytes_moved("triton_reduce", size, n_programs=232) == size + 4 * 232
    with pytest.raises(ValueError):
        pb.bytes_moved("nope", size)


def test_gbps_and_required_bytes() -> None:
    assert pb.gbps(10**9, 1000.0) == pytest.approx(1.0)
    assert pb.gbps(512 * pb.MIB, 2.0) == pytest.approx(268.435456)
    assert pb.required_bytes(1024 * pb.MIB, 768 * pb.MIB) == (2048 + 768) * pb.MIB


def test_method_order_interleaves() -> None:
    orders = [pb.method_order(r) for r in range(3)]
    assert all(sorted(o) == sorted(pb.METHODS) for o in orders)
    assert len({o[0] for o in orders}) == 3


def _fake_measurements() -> list[dict]:
    ms = []
    for size_mib, base in ((256, 400.0), (1024, 410.0)):
        size = size_mib * pb.MIB
        for rnd in range(5):
            for method, f in (("torch_copy", 0.9), ("triton_copy", 0.92), ("triton_reduce", 1.0)):
                nb = pb.bytes_moved(method, size, 232)
                bw = base * f + rnd  # 5 distinct values per (method, size)
                ms.append({"method": method, "size_bytes": size, "round": rnd,
                           "ms_median": nb / (bw * 1e9) * 1e3, "bytes_moved": nb, "gbps": bw,
                           # round 3 unknown (no samples), round 4 of the reduce throttled
                           "throttled": None if rnd == 3 else (rnd == 4 and method == "triton_reduce")})
    return ms


def test_aggregate_and_peaks() -> None:
    rows = pb.aggregate(_fake_measurements(), spec_gbps=432.0)
    assert [(r["method"], r["size_mib"]) for r in rows] == [
        ("torch_copy", 256), ("triton_copy", 256), ("triton_reduce", 256),
        ("torch_copy", 1024), ("triton_copy", 1024), ("triton_reduce", 1024)]
    red = rows[-1]
    assert red["gbps_median"] == 412.0 and red["gbps_min"] == 410.0 and red["gbps_max"] == 414.0
    assert red["pct_of_spec"] == pytest.approx(100 * 412 / 432)
    assert red["throttled_repeats"] == 1 and red["unknown_throttle_repeats"] == 1
    assert red["bytes_moved"] == 1024 * pb.MIB + 4 * 232
    p = pb.peaks(rows)
    assert p["read_peak_gbps"] == 412.0 and p["read_peak_size_mib"] == 1024
    assert p["copy_peak_method"] == "triton_copy"


def test_summary_from_saved_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("KVCACHE_RESULTS_DIR", str(tmp_path))
    rows = pb.aggregate(_fake_measurements(), spec_gbps=432.0)
    config = {"sizes_mib": [256, 1024], "skipped_sizes_mib": [512], "repeats": 5, "warmup_ms": 25,
              "rep_ms": 200, "dtype": "float16", "reduce_programs": 232, "spec_gbps": 432.0,
              "spec_source": pb.SPEC_SOURCE, "nvidia_smi_max_mem_clock_mhz": 9001.0}
    meta = {"git": {"sha": "abcdef1234567", "dirty": False},
            "gpu": {"name": "FakeGPU", "driver": "1.0"}, "packages": {}}
    path = save_result("peak_bw", config, {"rows": rows, **pb.peaks(rows)}, None, metadata=meta)

    assert pb.main(["--from-json", str(path)]) == 0
    md = (tmp_path / "summary" / "peak_bw.md").read_text()
    assert "| triton_reduce | 1024 | 412.0 | 410.0–414.0 | 95.4 |" in md
    assert "Read-only peak (decode ceiling): 412.0 GB/s** (triton_reduce, 1024 MiB)" in md
    assert "Skipped sizes (insufficient free VRAM): [512] MiB" in md
    assert "UNVERIFIED" in md and "FakeGPU" in md and "abcdef1234" in md
    out = capsys.readouterr().out
    assert "saved:" in out and "GPU monitor: no samples" in out


def test_run_without_cuda_exits_cleanly(capsys) -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA available; this checks the CPU path")
    assert pb.main(["--sizes-mib", "1", "--repeats", "1"]) == 0
    assert "No CUDA device available" in capsys.readouterr().out


@pytest.mark.gpu
def test_bw_kernels_on_gpu() -> None:
    from bench.kernels import bw_kernels as k

    src = torch.rand(3 * k.COPY_BLOCK + 17, device="cuda", dtype=torch.float16)
    dst = torch.empty_like(src)
    k.triton_copy(src, dst)
    assert torch.equal(dst, src)
    partials = torch.empty(16, device="cuda", dtype=torch.float32)
    got = k.triton_reduce_sum(src, partials).double().sum().item()
    assert got == pytest.approx(src.double().sum().item(), rel=1e-6)
