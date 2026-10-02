# CLAUDE.md — fp8-kvcache

## What this project is

Portfolio project: a **quantized, tiered KV-cache for LLM inference**. Two-week scope:

- **Week 1:** Triton paged-attention *decode* kernel that reads an FP8 (e4m3) KV-cache with fused
  dequantization. Benchmarked against a PyTorch reference, FlashInfer, and vLLM's FP8 KV path,
  with memory-bandwidth (roofline) analysis. Plus a serving baseline harness for vLLM.
- **Week 2:** tiered KV offload (GPU → pinned CPU memory) with async prefetch, integrated with
  vLLM, plus the write-up.

The goal is **honest, reproducible measurements against strong baselines**, not novelty claims.
Every headline number must trace back to a result Saurabh produced on the GPU.

## Two environments — read this first

| | Claude Code (you) | Saurabh's machine |
|---|---|---|
| Where | Cloud sandbox | WSL2 Ubuntu 24.04, `~/kvcache` |
| GPU | **None** | RTX 4080 Laptop, 12 GB, sm_89 |
| Job | Write code, CPU tests, docs | Run GPU code, benchmarks, vLLM |

Rules that follow from this:

- **You cannot run GPU code.** Never claim a GPU result, benchmark number, or "it works" for
  GPU paths. Say what you verified on CPU and what still needs a local run.
- **End every task with a `Local run` block** (format below) giving Saurabh exact commands and
  what to paste back.
- **Only use numbers Saurabh pastes.** When writing docs or summaries, quote his pasted output;
  if something is missing, leave a clearly marked TODO instead of estimating.
- Design every GPU script to **print a compact summary table** at the end that's easy to paste
  back, in addition to saving full results to disk.
- Make code testable on CPU wherever possible:
  - Import `vllm` and `flashinfer` lazily, inside the functions that need them.
  - Mark GPU tests `@pytest.mark.gpu` and skip them automatically when CUDA is unavailable.
  - Triton kernels can be checked on CPU with `TRITON_INTERPRET=1` on small shapes. Add an
    interpreter test for each kernel; if a feature (e.g. FP8) is unsupported in the interpreter,
    skip that case with a clear reason rather than deleting it.
  - The serving harness is tested against a small fake OpenAI-compatible streaming server.
  - Don't download models in CPU tests; use tiny synthetic configs.

### `Local run` block format

```
### Local run
cd ~/kvcache && git pull origin main && source .venv/bin/activate
<exact commands>
Expected: <what success looks like>
Paste back: <exactly which output lines / summary table to paste>
```

## Ownership

- **Saurabh owns kernel design:** memory layout, tiling, program/grid mapping, quantization scheme
  and scale granularity. For anything under `src/kvcache/kernels/` or a layout/quantization
  decision: propose options with trade-offs and **stop for a decision** unless the prompt says to
  implement. When editing kernels, comment every non-obvious choice.
- **Claude Code owns:** harnesses, tests, references, integration glue, plotting, scripts, docs.
- Before any kernel optimization: state the hypothesis and expected effect (bytes moved,
  parallelism, occupancy) *before* changing code.

## Saurabh's local GPU environment (fixed facts)

- WSL2 Ubuntu 24.04. Code in `~/kvcache`; venv at `~/kvcache/.venv` managed with **uv**.
- GPU: RTX 4080 Laptop, 12 GB, sm_89 (Ada, FP8-capable). VRAM is shared with the Windows display;
  use `--gpu-memory-utilization 0.85` max for vLLM.
- Already installed: PyTorch 2.13 (+cu130), Triton, vLLM, FlashInfer. CUDA toolkit 13.0 at
  `/usr/local/cuda`.
- **Nsight Compute does not work locally** (ERR_NVGPUCTRPERM under WSL). Use
  `triton.testing.do_bench` + computed bandwidth. ncu profiling happens later on a rented GPU.
- Laptop GPU throttles with heat/power; numbers drift. See benchmarking rules.
- Dev model: `Qwen/Qwen2.5-1.5B-Instruct`. **Read model dims from the HF config; never hardcode.**
- Triton FP8 type for `torch.float8_e4m3fn` is `tl.float8e4nv`.

## Dependencies — protect the local GPU stack

- **`pyproject.toml` must NOT list torch, triton, vllm, or flashinfer as dependencies.** Installing
  this package locally must never replace Saurabh's CUDA builds. List them under an optional
  `cpu-dev` extra (or a separate requirements file) for your sandbox only.
- Provide `scripts/setup_local.sh`: `uv pip install -e . --no-deps` plus the light tools
  (pytest, numpy, pandas, matplotlib, httpx, openai, datasets) without touching the GPU packages.
- In the sandbox, install CPU torch + triton as needed for tests.

## Layout

```
fp8-kvcache/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── src/kvcache/
│   ├── cache/        # paged KV layout, block tables, fp8 quant utils
│   ├── reference/    # PyTorch reference attention (ground truth)
│   ├── kernels/      # Triton kernels, one file per version + registry in __init__.py
│   └── tiering/      # week 2
├── bench/
│   ├── common/       # metadata, GPU monitor, result writer/loader
│   ├── kernels/      # kernel microbenchmarks, quant-error measurement
│   ├── serving/      # load generator, traces, fake server for tests
│   └── results/
│       ├── raw/      # full JSON results (gitignored, local only)
│       └── summary/  # small markdown/CSV summaries + figures (committed from local)
├── scripts/          # setup_local, env_check, peak_bw, capture_kv, sweeps, plots
├── tests/
│   └── data/         # captured real KV tensors (gitignored)
└── docs/             # progress.md, kernel_log.md, baseline.md, kv_stats.md, results
```

## Benchmarking rules (non-negotiable)

1. **Never fabricate, estimate, or "fill in" numbers.**
2. Every GPU run saves JSON to `bench/results/raw/` via `bench/common/results.py`, with: git SHA +
   dirty flag, timestamp, GPU name, driver, torch/triton/vllm/flashinfer versions, full config,
   and GPU-state samples (SM clock, mem clock, power, temp) taken during the run.
3. Warm up before measuring. Kernels: `triton.testing.do_bench`, report median plus p20/p80.
   Serving: discard warmup requests.
4. Repeats: ≥5 per kernel config, ≥3 per serving config. Report median and spread.
   Flag a run as **throttled** if its median SM clock is >15% below the session max.
5. Compare variants **in the same session, interleaved** (A B A B).
6. Bandwidth: achieved GB/s = bytes that must move from DRAM ÷ median time. Count KV bytes for the
   *actual* context lengths (not padded), plus scales, Q, output, block tables, context lens.
   Report **% of measured peak** from the latest `scripts/peak_bw.py` result.
7. Plots and summaries are generated by scripts from saved JSON, never typed in by hand.

## Correctness rules

- Write or extend tests **before** optimizing a kernel. No kernel change lands with failing tests.
- Keep two error measurements separate:
  - **Kernel error** — kernel vs PyTorch reference using the *same* cache (fp16, or fp8
    dequantized), fp32 math. Start at `atol=rtol=1e-2` on fp16 output. A failure is a bug.
  - **Quantization error** — FP8-cache vs FP16-cache attention. Measured and reported
    (max abs, mean abs, cosine similarity), not a pass/fail test.
- Test shapes: head_dim 64 and 128; GQA ratios 1, 4, 6, 8; block sizes 16 and 32; context lengths
  including 1, non-multiples of block size, and up to 32k (GPU, `slow`); batch 1–64 with ragged
  lengths; non-contiguous block tables. Interpreter tests use small shapes only.
- Also test on **real KV captured from the model** (`tests/data/`, local GPU only).
- Seed everything.

## Coding conventions

- Type hints, small modules, no notebook-only logic.
- One file per kernel version (`v0_...`, `v1_...`); keep old versions. All share one interface
  and register in `src/kvcache/kernels/__init__.py`.
- Never modify installed packages. Integrate with vLLM via public extension points or wrappers.

## Git workflow

- Repo: github.com/SaurabhDusane/fp8-kvcache. Target branch: **main**. Commit after each
  completed task with a descriptive message and push. If you can't push to main directly, open a
  PR and say so in your reply so Saurabh can merge it.
- Saurabh's local commits only touch `bench/results/summary/`, `docs/` result sections, and (on
  kernel days) kernel files he writes. Don't rewrite those without asking.

## Session protocol

- **Start:** read this file and `docs/progress.md`.
- **End:** append to `docs/progress.md` — what was done, what was verified on CPU, what is waiting
  on a local run, open issues, next step.
- Kernel experiments go in `docs/kernel_log.md`:
  hypothesis → change → prediction → result (numbers from Saurabh's pasted output) → keep/revert.
