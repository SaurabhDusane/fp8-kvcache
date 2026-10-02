#!/usr/bin/env python
"""Print the local GPU / software environment and run a quick FP8 sanity check.

Usage:
    python scripts/env_check.py

Exit codes:
    0  all checks passed, or no CUDA GPU is present (reported clearly, not an error)
    1  CUDA is present but a check failed (e.g. FP8 round-trip mismatch)

GPU packages are only inspected, never imported at module load, so this runs anywhere.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
import shutil
import subprocess
import sys
from typing import Callable

# Distribution names to try for each package (first hit wins).
PACKAGES: dict[str, tuple[str, ...]] = {
    "torch": ("torch",),
    "triton": ("triton", "pytorch-triton"),
    "vllm": ("vllm",),
    "flashinfer": ("flashinfer-python", "flashinfer"),
}

# E4M3 has 3 mantissa bits: round-to-nearest relative error <= 2**-4 for normal values.
FP8_E4M3_MAX = 448.0
FP8_E4M3_REL_TOL = 2.0**-4


def row(key: str, value: object) -> None:
    print(f"  {key:<22} {value}")


def section(title: str) -> None:
    print(f"\n== {title} ==")


def package_version(name: str) -> str:
    """Version from installed metadata; avoids importing heavy packages like vllm."""
    for dist in PACKAGES[name]:
        try:
            return md.version(dist)
        except md.PackageNotFoundError:
            continue
    # Fallback for source/dev installs without matching dist metadata.
    if importlib.util.find_spec(name) is None:
        return "not installed"
    try:
        return str(getattr(importlib.import_module(name), "__version__", "installed (version unknown)"))
    except Exception as exc:  # broken install
        return f"installed but import failed ({type(exc).__name__}: {exc})"


def run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def nvcc_info() -> tuple[str, str]:
    path = shutil.which("nvcc")
    if path is None:
        for candidate in ("/usr/local/cuda/bin/nvcc",):
            if shutil.which(candidate):
                path = candidate
                break
    if path is None:
        return "not found (not on PATH, not in /usr/local/cuda/bin)", "n/a"
    out = run([path, "--version"])
    release = "unknown"
    if out:
        for line in out.splitlines():
            if "release" in line:
                release = line.strip()
                break
    return path, release


def driver_version() -> str:
    out = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if not out:
        return "unavailable (nvidia-smi not found or failed)"
    return out.splitlines()[0].strip()


def fp8_sanity_check(torch_mod, device: str = "cuda") -> tuple[bool, list[str]]:
    """float8_e4m3fn tensor on `device`, cast back to fp16, verify against expected values.

    `device` is a parameter only so the logic can be unit-tested on CPU.
    """
    torch = torch_mod
    lines: list[str] = []
    if not hasattr(torch, "float8_e4m3fn"):
        return False, ["torch has no float8_e4m3fn dtype"]
    dev = torch.device(device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    ok = True

    # 1) Values exactly representable in e4m3 must round-trip bit-exactly.
    exact = torch.tensor(
        [0.0, 1.0, -2.0, 0.5, 0.015625, 1.75, -3.5, 240.0, FP8_E4M3_MAX, -FP8_E4M3_MAX],
        dtype=torch.float16,
        device=dev,
    )
    x8 = exact.to(torch.float8_e4m3fn)
    back = x8.to(torch.float16)
    sync()
    exact_ok = (x8.dtype == torch.float8_e4m3fn and x8.element_size() == 1
                and x8.device.type == dev.type and torch.equal(back, exact))
    lines.append(f"exact round-trip (10 values): {'OK' if exact_ok else 'FAIL'}")
    if not exact_ok:
        lines.append(f"  expected {exact.tolist()}\n  got      {back.tolist()}")
    ok &= exact_ok

    # 2) Random normal-range values: relative error within e4m3 rounding bound.
    g = torch.Generator(device=dev).manual_seed(0)
    r = torch.randn(1 << 16, generator=g, device=dev, dtype=torch.float32) * 16
    mag = r.abs().clamp(2.0**-6, FP8_E4M3_MAX)  # stay in the normal range
    r = (torch.sign(r) * mag).to(torch.float16)
    rb = r.to(torch.float8_e4m3fn).to(torch.float16)
    rel = ((rb.float() - r.float()).abs() / r.float().abs()).max().item()
    rand_ok = rel <= FP8_E4M3_REL_TOL + 1e-6
    lines.append(f"random round-trip (65536 values): max rel err {rel:.4f} "
                 f"(bound {FP8_E4M3_REL_TOL:.4f}) {'OK' if rand_ok else 'FAIL'}")
    ok &= rand_ok

    # 3) Saturation semantics: e4m3fn has no inf; document what this build does.
    big = torch.tensor([1000.0], dtype=torch.float16, device=dev).to(torch.float8_e4m3fn)
    lines.append(f"cast of 1000.0 -> {big.to(torch.float16).item()} "
                 "(info only; quantizers must clamp to +/-448 before casting)")
    return bool(ok), lines


def main() -> int:
    print("fp8-kvcache environment check")
    section("Python")
    row("python", sys.version.split()[0])
    row("executable", sys.executable)

    section("Packages")
    for name in PACKAGES:
        row(name, package_version(name))

    section("CUDA toolkit")
    path, release = nvcc_info()
    row("nvcc path", path)
    row("nvcc version", release)
    row("driver", driver_version())

    section("GPU")
    try:
        import torch
    except ImportError:
        print("  torch is not installed -> cannot query the GPU.")
        print("\nRESULT: NO CUDA (torch missing). Nothing to check on this machine; exiting cleanly.")
        return 0

    row("torch CUDA runtime", torch.version.cuda or "none (CPU-only torch build)")
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception as exc:
        cuda_ok = False
        row("cuda error", f"{type(exc).__name__}: {exc}")
    if not cuda_ok:
        print("  No CUDA device available (CPU-only torch build, no GPU, or driver not visible).")
        print("\nRESULT: NO CUDA. GPU and FP8 checks skipped; exiting cleanly. "
              "Run this on the GPU machine for a full check.")
        return 0

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    major, minor = props.major, props.minor
    free, total = torch.cuda.mem_get_info(idx)
    row("device count", torch.cuda.device_count())
    row("name", props.name)
    row("compute capability", f"sm_{major}{minor} ({major}.{minor})")
    row("VRAM total", f"{total / 2**30:.2f} GiB")
    row("VRAM free", f"{free / 2**30:.2f} GiB")
    row("SM count", props.multi_processor_count)
    row("native FP8 (sm>=8.9)", "yes" if (major, minor) >= (8, 9) else "no")

    section("FP8 sanity check (float8_e4m3fn)")
    try:
        fp8_ok, lines = fp8_sanity_check(torch)
    except Exception as exc:
        fp8_ok, lines = False, [f"raised {type(exc).__name__}: {exc}"]
    for line in lines:
        print(f"  {line}")

    print(f"\nRESULT: {'OK' if fp8_ok else 'FAIL'} "
          f"({props.name}, sm_{major}{minor}, torch {torch.__version__}, CUDA {torch.version.cuda})")
    return 0 if fp8_ok else 1


if __name__ == "__main__":
    sys.exit(main())
