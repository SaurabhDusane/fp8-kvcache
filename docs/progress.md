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

## 2026-10-02 · Day 1 · P2 Results and GPU monitoring

**Done**
- `bench/common/metadata.py`: git SHA/dirty/branch, timestamps, hostname, GPU name/driver/VRAM
  (nvidia-smi), CUDA runtime (from torch if already imported), torch/triton/vllm/flashinfer
  versions from package metadata. Missing pieces are `"unavailable"`.
- `bench/common/gpu_monitor.py`: `GpuMonitor` context manager polls nvidia-smi (SM/mem clock,
  power, power limit, temp, util, mem used) every 200 ms in a daemon thread; `MonitorResult` with
  raw samples, min/median/max summary, achieved interval, and throttle flag (median SM clock
  >15% below session max; per-run max by default, `flag_throttled` for a whole session).
  No nvidia-smi → warning + empty samples. Query failures are counted, not fatal.
- `bench/common/results.py`: `save_result` → `bench/results/raw/<date>/<name>_<HHMMSS>.json`
  (atomic write, no overwrite within a second), `load_results(pattern)` → flat DataFrame
  (`config.*`, `metrics.*`, `throttled`, full record in `_raw`), `write_summary` → `.md`
  (+ `.csv` for DataFrames) in `bench/results/summary/`, no `tabulate` dependency.
- `scripts/gpu_monitor_demo.py`: 5 s fp16 matmul loop under the monitor; prints and saves.

**Verified on CPU**
- `pytest -q`: 39 passed, 1 skipped (gpu). Monitor tested with mocked query lines, a flaky
  query, a missing nvidia-smi, and a fake `nvidia-smi` executable on PATH.
- `gpu_monitor_demo.py` exits 0 with "No CUDA device available".

**Waiting on local run**
- `gpu_monitor_demo.py` on the GPU: real nvidia-smi sampling rate under WSL, which fields
  report `[N/A]`, and the printed summary. Pasted output: TODO.
- P1 local run (setup_local.sh, env_check.py) also still not reported back.

**Open issues**
- nvidia-smi spawn latency under WSL may exceed 200 ms; the summary's "achieved" interval will
  show it. If it does, switch to a long-running `nvidia-smi -lms 200` stream.

**Next step**
- P3: `scripts/peak_bw.py` (roofline ceiling).
