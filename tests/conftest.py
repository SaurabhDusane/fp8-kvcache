"""Shared pytest configuration.

Tests marked ``@pytest.mark.gpu`` are skipped automatically when CUDA is unavailable
(including when torch itself is not installed), so the same suite runs in the CPU-only
cloud sandbox and on the local GPU machine.
"""

from __future__ import annotations

import functools

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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gpu_items = [item for item in items if "gpu" in item.keywords]
    if not gpu_items:
        return
    available, reason = cuda_status()
    if available:
        return
    skip_gpu = pytest.mark.skip(reason=f"gpu test skipped: {reason}")
    for item in gpu_items:
        item.add_marker(skip_gpu)
