# Progress

Session log per the protocol in CLAUDE.md: what was done, what was verified on CPU, what is
waiting on a local run, open issues, next step.

## 2026-10-02 · Day 1 · P1 Bootstrap

**Done**
- Repo skeleton per CLAUDE.md Layout (`src/kvcache/{cache,reference,kernels,tiering}`,
  `bench/{common,kernels,serving,results}`, `scripts/`, `tests/`, `docs/`).
- `pyproject.toml`: package `kvcache` (src layout), no torch/triton/vllm/flashinfer deps;
  `dev` extra (light tools) and sandbox-only `cpu-dev` extra (CPU torch + triton).
- pytest markers `slow` and `gpu`; `tests/conftest.py` auto-skips `gpu` tests without CUDA.
- `scripts/setup_local.sh`: editable `--no-deps` install + light tools, with the installed
  GPU packages pinned via a constraints file and verified unchanged afterwards.
- `scripts/env_check.py`: GPU/driver/CUDA/package/nvcc report + FP8 e4m3 round-trip check.

**Verified on CPU (sandbox, torch 2.14.1+cpu, triton 3.8.0)**
- `pytest -q`: CPU tests pass; `gpu` tests skipped with reason "CUDA is not available".
- `env_check.py` exits 0 with a clear "NO CUDA" message.
- `setup_local.sh` run twice in a venv with torch/triton installed (both left unchanged, rerun is
  a no-op) and once in a fresh venv (no torch pulled in; gpu tests then skip as "torch is not installed").

**Waiting on local run**
- `setup_local.sh`, `env_check.py` (GPU section + FP8 check), `pytest -q` (incl. gpu tests).
  Paste env_check output here: TODO.

**Open issues**
- None yet.

**Next step**
- P2: `bench/common/` (metadata, GPU monitor, results writer/loader).
