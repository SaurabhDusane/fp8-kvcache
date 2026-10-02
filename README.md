# fp8-kvcache

A quantized, tiered KV-cache for LLM inference.

- **Week 1:** a Triton paged-attention decode kernel that reads an FP8 (e4m3) KV-cache with
  fused dequantization, benchmarked against a PyTorch reference, FlashInfer, and vLLM's FP8 KV
  path, with memory-bandwidth (roofline) analysis. Plus a serving baseline harness for vLLM.
- **Week 2:** tiered KV offload (GPU → pinned CPU memory) with async prefetch, integrated with vLLM.

**Status:** bootstrap. No results yet. All headline numbers will come from runs on an
RTX 4080 Laptop GPU (12 GB, sm_89) and trace back to saved JSON in `bench/results/`.

## Setup

GPU machine (keeps the existing CUDA builds of torch/triton/vllm/flashinfer untouched):

```bash
source .venv/bin/activate
bash scripts/setup_local.sh
python scripts/env_check.py
pytest -q
```

CPU-only development (no GPU; GPU tests auto-skip):

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[cpu-dev]" --extra-index-url https://download.pytorch.org/whl/cpu
pytest -q
```

## Layout

See [CLAUDE.md](CLAUDE.md#layout). Progress log: [docs/progress.md](docs/progress.md).
Kernel experiment log: [docs/kernel_log.md](docs/kernel_log.md).
