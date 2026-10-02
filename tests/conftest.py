"""Shared pytest configuration.

Tiers:
- ``gpu``: needs CUDA. Skipped automatically when CUDA (or torch) is unavailable, and when
  running under the Triton interpreter.
- ``interpreter``: Triton kernels run by the interpreter on CPU tensors (small shapes).
  Needs TRITON_INTERPRET=1 *before triton is imported*, because ``@triton.jit`` (including
  triton.language helpers such as ``tl.zeros``) picks interpreted vs compiled at import time.
  On a machine without CUDA this conftest sets it automatically, so plain ``pytest`` runs the
  interpreter tier. On the GPU machine run it explicitly:
      TRITON_INTERPRET=1 pytest -m interpreter
"""

from __future__ import annotations

import functools
import os
import sys
import warnings

import pytest


@functools.lru_cache(maxsize=1)
def cuda_status() -> tuple[bool, str]:
    """Return (available, reason). Cached: CUDA init is slow and must happen once."""
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"
    try:
        if torch.cuda.is_available():
            return True, ""
    except Exception as exc:  # broken driver / WSL passthrough issues
        return False, f"CUDA check raised {type(exc).__name__}: {exc}"
    return False, "CUDA is not available"


def _interpreter_on() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


def _configure_interpreter() -> None:
    if "TRITON_INTERPRET" in os.environ or cuda_status()[0]:
        return
    if "triton" in sys.modules:
        warnings.warn("triton was imported before conftest; interpreter tests may fail")
    os.environ["TRITON_INTERPRET"] = "1"


_configure_interpreter()  # at conftest import, before any test module imports triton


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    available, reason = cuda_status()
    interp = _interpreter_on()
    skip_gpu = None
    if not available:
        skip_gpu = pytest.mark.skip(reason=f"gpu test skipped: {reason}")
    elif interp:
        skip_gpu = pytest.mark.skip(reason="gpu test skipped: running under TRITON_INTERPRET=1")
    skip_interp = pytest.mark.skip(
        reason="interpreter test: run with TRITON_INTERPRET=1 pytest -m interpreter")
    for item in items:
        if "gpu" in item.keywords and skip_gpu is not None:
            item.add_marker(skip_gpu)
        if "interpreter" in item.keywords and not interp:
            item.add_marker(skip_interp)


@pytest.fixture
def captured_kv_dir():
    """tests/data with real captured KV (scripts/capture_kv.py on the GPU machine); skips the
    test if the capture hasn't been run."""
    from bench.kernels.kv_capture import DATA_DIR, read_manifest

    manifest = read_manifest(DATA_DIR)
    if manifest is None or not all((DATA_DIR / f).exists() for f in manifest["files"]):
        pytest.skip("no captured KV in tests/data (run scripts/capture_kv.py on the GPU machine)")
    return DATA_DIR, manifest
