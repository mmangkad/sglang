# AllReduce Communicator Benchmark

This directory contains the maintained SGLang communicator benchmark for
allreduce path tuning.

- Main script: `benchmark/kernels/all_reduce/benchmark_all_reduce.py`

## What It Benchmarks

Per tensor shape `[num_tokens, hidden_size]`, it benchmarks the available
communicators on your environment:

- `torch_nccl` (baseline via `torch.distributed.all_reduce` on NCCL group)
- `custom_allreduce` (SGLang custom/aiter allreduce dispatcher)
- `pynccl`
- `torch_symm_mem` (if supported)
- `flashinfer_auto`, `flashinfer_trtllm`, `flashinfer_mnnvl` (if supported)

It also computes a threshold recommendation for FlashInfer standalone
allreduce (`_FI_ALLREDUCE_MAX_SIZE_MB`) for your current `(SM, world_size)`.

## Quick Start

```bash
torchrun --nproc_per_node=8 \
  benchmark/kernels/all_reduce/benchmark_all_reduce.py
```

With explicit sweep and JSON output:

```bash
torchrun --nproc_per_node=8 \
  benchmark/kernels/all_reduce/benchmark_all_reduce.py \
  --sequence-lengths 16 64 128 256 512 1024 2048 4096 \
  --hidden-size 8192 \
  --dtype bf16 \
  --threshold-candidates-mb 0.5 1 2 4 8 16 32 64 \
  --output-json /tmp/sglang_ar_bench.json
```

## Threshold Tuning Notes

- Recommendation target is driven by `--threshold-baseline`:
  - `best_non_flashinfer` (default): compare against the fastest non-FlashInfer
    communicator at each message size.
  - Or pass a communicator name from the result table.
- The script enforces `--threshold-max-regression` (default `1.02`) where
  possible; if no candidate satisfies it, it falls back to the best
  unconstrained candidate.
- It prints a ready-to-apply snippet for
  `python/sglang/srt/distributed/device_communicators/flashinfer_all_reduce.py`.
