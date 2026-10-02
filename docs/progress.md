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

## 2026-10-02 · Day 1 · P3 Peak bandwidth

**Done**
- `bench/kernels/bw_kernels.py`: Triton copy kernel and read-only grid-stride sum kernel
  (fp32 accumulation, one partial per program).
- `scripts/peak_bw.py`: torch copy / Triton copy / Triton reduce at 256, 512, 1024 MiB (fp16),
  sizes skipped if free VRAM < 2×size + 768 MiB reserve. Bytes: copy = 2N, reduce = N + 4 B/program.
  `do_bench` (L2 flushed per call), 5 interleaved rounds with rotated method order, GPU monitor
  on; per-call SM-clock median and throttle flag. Correctness check of all three methods before
  timing. Saves JSON (`metrics.read_peak_gbps` is the decode ceiling), writes
  `bench/results/summary/peak_bw.md` from the saved JSON, prints it. `--from-json` rebuilds it.
- Spec constant 432 GB/s (RTX 4080 Laptop, 192-bit GDDR6 @ 18 Gbps) marked UNVERIFIED; the run
  records nvidia-smi's max memory clock for cross-checking.
- New pytest tier `interpreter`: auto-enabled (TRITON_INTERPRET=1) when CUDA is unavailable;
  on the GPU machine run `TRITON_INTERPRET=1 pytest -m interpreter`. gpu tests skip under it.
- `GpuMonitor.elapsed()` and `MonitorResult.window_median()` for per-measurement clocks.

**Verified on CPU**
- `pytest -q`: 55 passed, 2 skipped (gpu). Interpreter tests check both kernels exactly
  (integer-valued fp16 → exact sums), incl. masked tails and grid-stride with uneven blocks.
  A deliberately broken grid stride fails 3 of them.
- Summary generation from a saved JSON (synthetic numbers, test only); script exits 0 without CUDA.

**Waiting on local run**
- `peak_bw.py` on the GPU (plugged in, Best performance). Pasted summary: TODO.
- Earlier: env_check output, gpu_monitor_demo output: TODO.

**Open issues**
- Spec bandwidth unverified (see above).
- Reduce grid = 4 programs/SM (8 warps each) is a guess; if the read-only number looks low vs
  copy, try `--reduce-programs-per-sm 2/6/8`.

**Next step**
- Day 2, P4: serving load generator.

## 2026-10-02 · Day 2 · P4 Serving load generator

**Done**
- `bench/serving/`:
  - `loadgen.py`: httpx async streaming chat completions (`stream_options.include_usage`,
    `ignore_eos`, `temperature 0`); per request: scheduled/send time, first-content-chunk time,
    every content-chunk arrival, end time, client-measured prompt tokens, server usage tokens,
    errors. Arrivals: Poisson session starts (seeded) or all at t=0; optional session
    concurrency cap (rate inf + cap = closed loop). Multi-turn turns are sequential with think time.
    Warmup requests (separate prompts) run first and are excluded.
  - `metrics.py`: TTFT, ITL (pooled chunk gaps, same as `vllm bench serve`), TPOT, E2E,
    req/s, output/input tok/s, goodput (TTFT ≤ X AND per-request p90 ITL ≤ Y), send lag,
    output-length mismatch counter (catches ignore_eos not honoured).
  - `traces.py`: `chat` (ShareGPT via `datasets`, then hub download, else lognormal fallback;
    source + reason recorded), `long_context` (2000–3500 prompt, 32–128 out), `multi_turn`
    (3–6 turns, full conversation resent, deterministic stand-in assistant replies of the
    requested length, exponential think time), `fixed` (for apples-to-apples with
    `vllm bench serve --dataset-name random`). Lengths measured with the model tokenizer incl.
    chat template.
  - `run.py` CLI (`python -m bench.serving.run ...`), GPU monitor on, saves `serving_<trace>`.
  - `fake_server.py`: stdlib asyncio OpenAI-compatible streaming server with configurable
    TTFT/ITL/prefill delay, failure injection, EOS behaviour; records requests and peak concurrency.
- `datasets>=3` pinned in the dev extra and `setup_local.sh` (uv had resolved datasets 2.14,
  which crashes on import with pyarrow ≥ 21).

**Verified on CPU**
- `pytest -q`: all serving tests pass against the fake server: metrics match injected delays
  (TTFT 50 ms / ITL 10 ms), role-only first chunk not counted as first token, request body
  (max_tokens, ignore_eos, include_usage), concurrency cap, Poisson send times, multi-turn
  ordering and growing prefix, HTTP errors, warmup exclusion, CLI end to end + saved JSON.
- Exact metric math on hand-built records; trace determinism and length bounds (stub tokenizer).

**Not verified (needs local run)**
- Real Qwen tokenizer + vLLM: TTFT/ITL vs `vllm bench serve`, ignore_eos honoured, usage chunk.
- ShareGPT loading through `datasets` (not attempted here: ~670 MB download).

**Open issues**
- The concurrency cap counts sessions (users), not requests; for single-turn traces these are
  the same.

**Next step**
- P5: baseline sweep (fp16 vs fp8 KV cache).

## 2026-10-02 · Day 2 · P5 Baseline sweep (FP16 vs FP8 KV cache)

**Done**
- `scripts/run_baseline_sweep.py`: per `--kv-cache-dtype` (auto, fp8) launches
  `vllm serve <model> --kv-cache-dtype D --max-model-len 4096 --gpu-memory-utilization 0.85
  --port 8000` (+ only `--server-arg`s given explicitly), polls `/health`, runs each trace over
  its rate list with 3 repeats per rate, then shuts the server down (SIGINT → SIGTERM → SIGKILL
  on the process group). Server logs saved to `bench/results/raw/<date>/vllm_logs/`; "GPU KV
  cache size", "Maximum concurrency", "Available KV cache memory" parsed into every run's config.
  Start failures / mid-sweep crashes are reported with the log's traceback and the sweep moves
  on; no setting is ever changed. Ctrl-C saves what was measured.
- Rate plan (`bench/serving/sweep.py`): ascending rate list; a point is collapsed when median
  SLO attainment < 50%; one more point after the first collapse, then stop; if the list ends
  without collapse, keep doubling up to `--max-rate` (logged). Requests per run ≈ 45 s of
  arrivals (30–300). Seeds depend on (rate, repeat) only: same workload across configs, new
  prompts per run (no prefix-cache carry-over). Best-effort `/reset_prefix_cache` before runs.
- `bench/results/summary/baseline.md` generated from saved JSON (`--summarize-only <id>` to
  regenerate): server configs + verbatim KV log lines, saturation (peak goodput, first collapsed
  rate), per-trace tables (median and min–max over repeats), throttled repeats, plan decisions.
- `scripts/plot_baseline.py`: per trace, TTFT p50/p99, ITL p50/p99, goodput vs rate; one color
  per dtype (validated palette), error bars = min–max over repeats, SLO lines.

**Verified on CPU**
- 106 passed. vLLM log parsing (V1 lines with ANSI codes, V0 lines), error extraction, rate plan
  (collapse stop, extension, max rate, no-extend), aggregation + saturation + markdown on fixtures.
- Server lifecycle with a fake `vllm` executable: start/parse/stop, crash with traceback,
  startup timeout, missing binary. Full sweep end to end with the fake `vllm` (fp8 forced to
  fail → reported, auto measured, baseline.md + PNGs produced).

**Waiting on local run**
- Full sweep, plots, baseline.md: TODO (paste).
- Not verified: real vLLM log line formats for this vLLM version; SIGINT shutdown frees VRAM
  before the next config starts.

**Next step**
- Write docs/baseline.md from the pasted baseline.md.

## 2026-10-02 · Day 3 · P6 decision + P7 Cache, FP8 utils, reference

**Decisions (Saurabh, P6):** layout **NHD, stride-generic** (logical
`[num_blocks, block_size, num_kv_heads, head_dim]`; consumers use strides; HND available as a
physical layout for experiments). FP8 scales **per KV head**, static, fp32, shape
`[num_kv_heads]` (per-tensor = broadcast of one value).

**Done**
- `src/kvcache/cache/fp8.py`: e4m3fn, scale = amax / 448 (zero group → 1.0, NaN ignored),
  clamp to ±448 before the cast; dequantize in fp32. Granularity inferred from the scale's rank:
  tensor `[]`, kv_head `[H]` (default), block_head `[NB, H]`, token_head `[NB, BS, H]`, all
  supported in quantize/dequantize and the reference so they can be evaluated later.
- `src/kvcache/cache/paged.py`: `allocate_kv_cache` (NHD or HND memory, logical NHD view),
  `assign_blocks` (random non-contiguous blocks, block 0 reserved as null/padding as in vLLM V1),
  `build_paged_kv_cache` (fp16/bf16/fp32/fp8; NaN poison in null, free and tail slots),
  `gather_kv`, `PagedKVCache` with `strides()` for launchers.
- `src/kvcache/reference/decode_attention.py`: `paged_decode_attention(q, k_cache, v_cache,
  block_tables, context_lens, sm_scale, k_scale, v_scale)` in fp32 with GQA (HF repeat_kv
  mapping), fp16 or fp8 + scales; `dense_decode_attention_sdpa` via torch SDPA.

**Verified on CPU**
- 40 CPU tests: paged vs SDPA at atol=rtol=1e-5 over head_dim {64,128} × GQA {1,4,6,8} ×
  block size {16,32} with ragged lengths incl. 1 and non-multiples; NHD vs HND identical; fp8
  reference equals SDPA on the dequantized cache for all granularities; e4m3 round-trip error
  bound (2^-4 relative / 2^-10·scale subnormal) for all granularities; clamping; poison never read.
- Mutation checks: wrong GQA mapping fails 17 tests, wrong block order fails 20.

**Waiting on local run**
- The same tests on CUDA (`-m gpu`, plus the 32k-context `slow` case).

**Next step**
- P8: capture real KV from the model.
