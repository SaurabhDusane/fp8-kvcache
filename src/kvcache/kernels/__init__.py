"""Triton decode-attention kernels, one file per version (v0_..., v1_...).

The kernel registry (the shared interface used by tests and benchmarks) is added together
with the kernel test suite. This package must stay importable on a CPU-only machine.
"""
