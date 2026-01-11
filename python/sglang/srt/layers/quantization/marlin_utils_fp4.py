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
    
    # Use the padded dimensions that were set during create_weights
    # These should have been padded to meet Marlin kernel requirements:
    # - hidden_size: multiple of 256
    # - intermediate_size: multiple of 128
    k = layer.hidden_size
    n = layer.intermediate_size_per_partition

    # WORKSPACE
    device = layer.w13_weight.device
    param_dtype = layer.params_dtype
    layer.workspace = marlin_make_workspace_new(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    # Get actual weight shapes from checkpoint
    # w13_weight: [E, 2*N_orig, K_orig//2]
    # w2_weight: [E, K_orig, N_orig//2]
    w13_shape = layer.w13_weight.shape
    k_orig = w13_shape[2] * 2  # Original hidden_size from checkpoint
    n_orig = w13_shape[1] // 2  # Original intermediate_size from checkpoint
    
    # Store original dimensions if not already set (for output unpadding)
    if not hasattr(layer, 'original_hidden_size') or layer.original_hidden_size is None:
        layer.original_hidden_size = k_orig
    if not hasattr(layer, 'original_intermediate_size_per_partition') or layer.original_intermediate_size_per_partition is None:
        layer.original_intermediate_size_per_partition = n_orig
    
    # Check if we need to pad weights to meet Marlin requirements
    need_k_pad = k > k_orig
    need_n_pad = n > n_orig
    
    if need_k_pad or need_n_pad:
        logger.info(
            f"Padding MoE weights for Marlin alignment: "
            f"hidden_size {k_orig} -> {k}, intermediate_size {n_orig} -> {n}"
        )

    # WEIGHT
    # Repack weights to marlin format
    for name in ["w13_weight", "w2_weight"]:
        weight = getattr(layer, name)
        tensor_list = []
        
        if "w13" in name:
            # w13: [E, 2*N, K//2] -> repack with size_n=2*N, size_k=K
            size_n, size_k = n * 2, k
            orig_n, orig_k = n_orig * 2, k_orig
            
            # Pad weight if needed: [E, 2*N_orig, K_orig//2] -> [E, 2*N, K//2]
            if need_k_pad or need_n_pad:
                # Pad K dimension (last dim, packed so pad by half)
                k_pad = (k - k_orig) // 2
                # Pad N dimension (middle dim, already *2)
                n_pad = (n - n_orig) * 2
                weight = torch.nn.functional.pad(
                    weight, (0, k_pad, 0, n_pad), mode="constant", value=0
                )
        else:
            # w2: [E, K, N//2] -> repack with size_n=K, size_k=N
            size_n, size_k = k, n
            orig_n, orig_k = k_orig, n_orig
            
            # Pad weight if needed: [E, K_orig, N_orig//2] -> [E, K, N//2]
            if need_k_pad or need_n_pad:
                # Pad K dimension (middle dim)
                k_pad = k - k_orig
                # Pad N dimension (last dim, packed so pad by half)
                n_pad = (n - n_orig) // 2
                weight = torch.nn.functional.pad(
                    weight, (0, n_pad, 0, k_pad), mode="constant", value=0
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
    # Permute scales - need to pad scales to match padded weight dimensions
    # Scale shape: [E, (size_n), (size_k // group_size)]
    for name in ["w13", "w2"]:
        scales = getattr(layer, name + "_weight_scale")
        if not is_nvfp4:
            scales = scales.view(torch.float8_e8m0fnu)
        scales = scales.to(param_dtype)

        tensor_list = []
        if "w13" in name:
            size_n, size_k = n * 2, k
            orig_n, orig_k = n_orig * 2, k_orig
        else:
            size_n, size_k = k, n
            orig_n, orig_k = k_orig, n_orig

        # Pad scales if needed
        # Scale shape is typically [E, size_n, size_k // group_size]
        if need_k_pad or need_n_pad:
            # Calculate padding for scales
            k_scale_pad = (size_k - orig_k) // group_size
            n_scale_pad = size_n - orig_n
            if k_scale_pad > 0 or n_scale_pad > 0:
                scales = torch.nn.functional.pad(
                    scales, (0, k_scale_pad, 0, n_scale_pad), mode="constant", value=0
                )

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
    # Permute bias - pad if needed
    for name in ["w13_bias", "w2_bias"]:
        if not hasattr(layer, name):
            continue
        bias = getattr(layer, name)
        if bias is None:
            continue
        bias = bias.to(param_dtype)

        # Pad bias if needed
        if "w13" in name:
            bias_size = n * 2
            orig_bias_size = n_orig * 2
        else:
            bias_size = k
            orig_bias_size = k_orig
        
        if bias.shape[-1] < bias_size:
            bias = torch.nn.functional.pad(
                bias, (0, bias_size - orig_bias_size), mode="constant", value=0
            )

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
