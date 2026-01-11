# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py
"""Marlin MXFP4 utilities for SGLang."""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    marlin_make_workspace,
    marlin_permute_bias,
    marlin_permute_scales,
)
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import gptq_marlin_repack

ScalarType, scalar_types = get_scalar_types()

FP4_MARLIN_SUPPORTED_GROUP_SIZES = [16, 32]  # 16 for NVFP4, 32 for MXFP4


def is_fp4_marlin_supported():
    """Check if FP4 Marlin is supported on this device."""
    if not _is_cuda:
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 75


def mxfp4_marlin_process_scales(
    marlin_scales: torch.Tensor,
    input_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Process scales for MXFP4 Marlin kernel.
    
    Args:
        marlin_scales: The permuted scales tensor
        input_dtype: Optional input dtype for activation quantization
        
    Returns:
        Processed scales tensor in float8_e8m0fnu format
    """
    # Fit the layout of fp8 dequantization
    if input_dtype is None or input_dtype.itemsize == 2:
        marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
            marlin_scales.size(0), -1
        )

    marlin_scales = marlin_scales.to(torch.float8_e8m0fnu)
    
    if input_dtype == torch.float8_e4m3fn:
        marlin_scales = marlin_scales.view(torch.uint8)
        assert marlin_scales.max() <= 249
        # exponent_bias (fp4->fp8) = 2 ** 3 - 2 ** 1 = 6
        marlin_scales = marlin_scales + 6
        marlin_scales = marlin_scales.view(torch.float8_e8m0fnu)
        
    return marlin_scales


def marlin_make_workspace_new(
    device: torch.device, max_blocks_per_sm: int = 1
) -> torch.Tensor:
    """Create workspace tensor for Marlin kernel.
    
    In the new Marlin kernel, we use the num of threadblocks as workspace
    size. The num of threadblocks is sms_count * max_blocks_per_sm.
    """
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(
        sms * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False
    )


def prepare_moe_fp4_layer_for_marlin(
    layer: torch.nn.Module,
    input_dtype: Optional[torch.dtype] = None,
) -> None:
    """Prepare MoE FP4 layer for Marlin kernel execution.
    
    This function reorders weights and scales for the Marlin kernel format.
    It processes MXFP4 (group_size=32) quantized weights.
    
    Args:
        layer: The MoE layer containing weights and scales
        input_dtype: Optional dtype for activation quantization (e.g., torch.float8_e4m3fn)
    """
    logger.warning_once(
        "Your GPU does not have native support for FP4 computation but "
        "FP4 quantization is being used. Weight-only FP4 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads."
    )

    is_nvfp4 = hasattr(layer, "w13_weight_scale_2")
    if input_dtype is not None and input_dtype.itemsize == 1:
        if is_nvfp4:
            raise RuntimeError("NVFP4 weight + INT8/FP8 activation is not supported.")
        elif input_dtype != torch.float8_e4m3fn:
            raise RuntimeError("MXFP4 weight + INT8 activation is not supported.")

    group_size = 16 if is_nvfp4 else 32

    e = layer.num_experts
    
    # Derive dimensions from actual weight shapes rather than layer config
    # This handles cases where checkpoint weights may have padding
    # w13_weight shape: [E, 2*N, K//2] (FP4 packed, 2 values per byte)
    # w2_weight shape: [E, K, N//2]
    w13_shape = layer.w13_weight.shape
    w2_shape = layer.w2_weight.shape
    
    # Derive k and n from weight shapes
    # For w13: shape is [E, 2*N, K//2]
    k = w13_shape[2] * 2  # K = last_dim * 2 (FP4 unpacked)
    n = w13_shape[1] // 2  # N = middle_dim / 2 (gate_up combined)
    
    # Store original config dimensions for potential input padding later
    layer.marlin_hidden_size = k
    layer.marlin_intermediate_size = n

    # WORKSPACE
    device = layer.w13_weight.device
    param_dtype = layer.params_dtype
    layer.workspace = marlin_make_workspace_new(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    # WEIGHT
    # Repack weights to marlin format
    for name in ["w13_weight", "w2_weight"]:
        weight = getattr(layer, name)
        tensor_list = []
        if "w13" in name:
            size_n, size_k = n * 2, k
            expected_shape = (e, size_n, size_k // 2)
        else:
            size_n, size_k = k, n
            expected_shape = (e, size_n, size_k // 2)

        # Verify weight shape matches derived dimensions
        if weight.shape != expected_shape:
            logger.warning(
                f"Weight shape mismatch for {name}: expected {expected_shape}, "
                f"got {weight.shape}. Weights may have been created with different padding."
            )

        for i in range(e):
            qweight = weight[i].view(torch.int32).T.contiguous()

            marlin_qweight = gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
            )
            tensor_list.append(marlin_qweight)

        weight = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        weight = torch.nn.Parameter(weight, requires_grad=False)

        setattr(layer, name, weight)

    # WEIGHT SCALES
    # Permute scales
    for name in ["w13", "w2"]:
        scales = getattr(layer, name + "_weight_scale")
        if not is_nvfp4:
            scales = scales.view(torch.float8_e8m0fnu)
        scales = scales.to(param_dtype)

        tensor_list = []
        if "w13" in name:
            size_n, size_k = n * 2, k
        else:
            size_n, size_k = k, n

        for i in range(e):
            scale = scales[i].T

            marlin_scales = marlin_permute_scales(
                s=scale,
                size_k=size_k,
                size_n=size_n,
                group_size=group_size,
            )
            marlin_scales = mxfp4_marlin_process_scales(
                marlin_scales, input_dtype=input_dtype
            )
            tensor_list.append(marlin_scales)

        scales = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        scales = torch.nn.Parameter(scales, requires_grad=False)
        setattr(layer, name + "_weight_scale", scales)

    # BIAS
    # Permute bias
    for name in ["w13_bias", "w2_bias"]:
        if not hasattr(layer, name):
            continue
        bias = getattr(layer, name)
        if bias is None:
            continue
        bias = bias.to(param_dtype)

        tensor_list = []
        for i in range(e):
            expert_bias = bias[i]
            tensor_list.append(marlin_permute_bias(expert_bias))

        bias = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        bias = torch.nn.Parameter(bias, requires_grad=False)
        setattr(layer, name, bias)


def rand_marlin_weight_mxfp4_like(
    weight: torch.Tensor,
    group_size: int,
    input_dtype: Optional[torch.dtype] = None,
):
    """Generate random Marlin MXFP4 weights for testing.
    
    Args:
        weight: Reference weight tensor to match shape
        group_size: Quantization group size (should be 32 for MXFP4)
        input_dtype: Optional input activation dtype
        
    Returns:
        Tuple of (reference_weight, marlin_qweight, marlin_scales)
    """
    assert group_size > 0
    size_n, size_k = weight.shape
    device = weight.device

    scales = torch.randint(
        110,
        120,
        (size_n, size_k // group_size),
        dtype=torch.uint8,
        device=weight.device,
    )
    scales = scales.view(torch.float8_e8m0fnu)

    fp4_weight = torch.randint(
        0, 256, (size_n, size_k // 2), dtype=torch.uint8, device=weight.device
    )
    fp4_weight_part_1 = (fp4_weight & 0b10000000) | ((fp4_weight & 0b01110000) >> 2)
    fp4_weight_part_1 = fp4_weight_part_1.view(torch.float8_e4m3fn)
    fp4_weight_part_1 = fp4_weight_part_1.to(weight.dtype) * (2**6)

    fp4_weight2 = fp4_weight << 4
    fp4_weight_part_2 = (fp4_weight2 & 0b10000000) | ((fp4_weight2 & 0b01110000) >> 2)
    fp4_weight_part_2 = fp4_weight_part_2.view(torch.float8_e4m3fn)
    fp4_weight_part_2 = fp4_weight_part_2.to(weight.dtype) * (2**6)

    weight_ref = torch.cat(
        [fp4_weight_part_2.unsqueeze(2), fp4_weight_part_1.unsqueeze(2)], 2
    ).view(size_n, size_k)
    weight_ref = weight_ref * scales.repeat_interleave(group_size, 1).to(weight.dtype)

    perm = torch.empty(0, dtype=torch.int, device=device)
    fp4_weight = fp4_weight.view(torch.int32).T.contiguous()
    marlin_qweight = gptq_marlin_repack(
        b_q_weight=fp4_weight,
        perm=perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=4,
    )

    marlin_scales = marlin_permute_scales(
        s=scales.T.to(weight.dtype),
        size_k=size_k,
        size_n=size_n,
        group_size=group_size,
    )

    marlin_scales = mxfp4_marlin_process_scales(marlin_scales, input_dtype=input_dtype)

    return weight_ref.T, marlin_qweight, marlin_scales.to(torch.float8_e8m0fnu)
