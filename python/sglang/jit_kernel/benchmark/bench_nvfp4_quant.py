from __future__ import annotations

import torch
import triton

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.nvfp4 import scaled_fp4_quant

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
BLOCK_SIZE = 16


def _torch_ref_quant(input: torch.Tensor, input_global_scale: torch.Tensor):
    m, n = input.shape
    x = input.view(m, n // BLOCK_SIZE, BLOCK_SIZE)
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = input_global_scale * (vec_max / FLOAT4_E2M1_MAX)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)

    scaled_x = x.to(torch.float32) * output_scale
    clipped = torch.clamp(scaled_x, -6.0, 6.0).reshape(m, n)

    rounded = clipped.clone()
    rounded[(rounded >= 0.0) & (rounded <= 0.25)] = 0.0
    rounded[(rounded > 0.25) & (rounded < 0.75)] = 0.5
    rounded[(rounded >= 0.75) & (rounded <= 1.25)] = 1.0
    rounded[(rounded > 1.25) & (rounded < 1.75)] = 1.5
    rounded[(rounded >= 1.75) & (rounded <= 2.5)] = 2.0
    rounded[(rounded > 2.5) & (rounded < 3.5)] = 3.0
    rounded[(rounded >= 3.5) & (rounded <= 5.0)] = 4.0
    rounded[rounded > 5.0] = 6.0

    # This baseline intentionally keeps work on GPU but does not pack to uint8.
    return rounded, scale


shape_range = get_benchmark_range(
    full_range=[(128, 2048), (512, 4096), (1024, 4096), (2048, 8192)],
    ci_range=[(128, 2048)],
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["shape"],
        x_vals=shape_range,
        x_log=False,
        line_arg="provider",
        line_vals=["jit", "torch_ref"],
        line_names=["JIT NVFP4 Quant", "Torch Ref"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="us",
        plot_name="nvfp4-quant-performance",
        args={},
    )
)
def benchmark(shape, provider):
    m, n = shape
    x = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    if provider == "jit":
        fn = lambda: scaled_fp4_quant(x, global_scale)
    elif provider == "torch_ref":
        fn = lambda: _torch_ref_quant(x, global_scale)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
