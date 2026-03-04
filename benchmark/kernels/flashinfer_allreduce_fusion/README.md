# FlashInfer Fused Collective Benchmark

Maintained benchmark for FlashInfer fused allreduce + residual + RMSNorm in
SGLang.

- Script: `benchmark/kernels/flashinfer_allreduce_fusion/benchmark_fused_collective.py`
- API used: FlashInfer unified API
  - `create_allreduce_fusion_workspace`
  - `allreduce_fusion`

## What It Measures

For each `(seq_len, backend, oneshot/twoshot, residual_mode)` configuration,
it compares:

- Standard path: `torch.distributed.all_reduce` + residual add + RMSNorm
- FlashInfer fused path: `AllReduceFusionPattern.kARResidualRMSNorm`

## Quick Start

```bash
torchrun --nproc_per_node=8 \
  benchmark/kernels/flashinfer_allreduce_fusion/benchmark_fused_collective.py
```

With explicit settings and JSON output:

```bash
torchrun --nproc_per_node=8 \
  benchmark/kernels/flashinfer_allreduce_fusion/benchmark_fused_collective.py \
  --sequence-lengths 128 512 1024 2048 4096 \
  --hidden-dim 8192 \
  --dtype bf16 \
  --backends trtllm mnnvl \
  --trials 50 \
  --ops-per-trial 16 \
  --check-correctness \
  --output-json /tmp/sglang_fused_collective.json
```

## Notes

- Requires CUDA and `world_size > 1`.
- Launch with `torchrun`.
- If a backend cannot initialize workspace, it is skipped.
- Use `--disable-oneshot` to benchmark twoshot only.
