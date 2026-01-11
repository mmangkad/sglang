# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py
"""
Utilities for FP4 (MXFP4/NVFP4) weight processing with Marlin kernels.
This enables SM120 (Blackwell RTX 50-series) support for GPT-OSS MXFP4 models.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    marlin_make_workspace,
    marlin_permute_bias,
    marlin_permute_scales,
)
from sglang.srt.utils import get_device_capability, is_cuda, log_info_on_rank0

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

FP4_MARLIN_SUPPORTED_GROUP_SIZES = [16, 32]

_warned_fp4_marlin = False

_is_cuda = is_cuda()

if _is_cuda:
    try:
        from sgl_kernel import gptq_marlin_repack
    except ImportError:
        gptq_marlin_repack = None


def is_fp4_marlin_supported() -> bool:
    """Check if FP4 Marlin is supported on the current device."""
    if not _is_cuda:
        return False
    major, minor = get_device_capability()
    return major * 10 + minor >= 75


def mxfp4_marlin_process_scales(
    marlin_scales: torch.Tensor,
    input_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Process MXFP4 scales for Marlin kernel compatibility.
    
    Args:
        marlin_scales: The permuted scales tensor
        input_dtype: Optional input dtype for activation quantization
        
    Returns:
        Processed scales in float8_e8m0fnu format
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


def prepare_moe_fp4_layer_for_marlin(
    layer: torch.nn.Module,
    input_dtype: Optional[torch.dtype] = None,
) -> None:
    """
    Prepare MXFP4 MoE layer weights for Marlin kernel execution.
    
    This function repacks the FP4 weights and scales into the format
    expected by the Marlin kernel, enabling SM120 (Blackwell) support
    for GPT-OSS MXFP4 models.
    
    Args:
        layer: The MoE layer containing w13_weight, w2_weight, and their scales
        input_dtype: Optional input dtype for activation quantization
    """
    global _warned_fp4_marlin
    
    if gptq_marlin_repack is None:
        raise ImportError(
            "gptq_marlin_repack is not available. Please ensure sgl_kernel "
            "is properly installed with Marlin support."
        )
    
    if not _warned_fp4_marlin:
        log_info_on_rank0(
            logger,
            "Your GPU does not have native support for FP4 computation but "
            "FP4 quantization is being used. Weight-only FP4 compression will "
            "be used leveraging the Marlin kernel. This may degrade "
            "performance for compute-heavy workloads."
        )
        _warned_fp4_marlin = True

    is_nvfp4 = hasattr(layer, "w13_weight_scale_2")
    if input_dtype is not None and input_dtype.itemsize == 1:
        if is_nvfp4:
            raise RuntimeError("NVFP4 weight + INT8/FP8 activation is not supported.")
        elif input_dtype != torch.float8_e4m3fn:
            raise RuntimeError("MXFP4 weight + INT8 activation is not supported.")

    group_size = 16 if is_nvfp4 else 32

    e = layer.num_local_experts
    k = layer.hidden_size
    n = layer.intermediate_size_per_partition

    # WORKSPACE
    device = layer.w13_weight.device
    param_dtype = getattr(layer, "params_dtype", torch.bfloat16)
    layer.marlin_workspace = marlin_make_workspace(device, max_blocks_per_sm=4)
    perm = torch.empty(0, dtype=torch.int, device=device)
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1

    # WEIGHT
    # Repack weights to marlin format
    for name in ["w13_weight", "w2_weight"]:
        weight = getattr(layer, name)
        tensor_list = []
        if "w13" in name:
            size_n, size_k = n * 2, k
        else:
            size_n, size_k = k, n

        assert weight.shape == (e, size_n, size_k // 2), (
            f"Expected shape ({e}, {size_n}, {size_k // 2}), "
            f"got {weight.shape} for {name}"
        )

        for i in range(e):
            qweight = weight[i].view(torch.int32).T.contiguous()

            marlin_qweight = gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
                is_a_8bit=is_a_8bit,
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
    for name in ["w13_weight_bias", "w2_weight_bias"]:
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

    # Mark layer as processed for Marlin
    layer.marlin_mxfp4_processed = True


def get_mxfp4_scalar_type():
    """Get the scalar type for MXFP4 weights."""
    from sgl_kernel.scalar_type import scalar_types
    return scalar_types.float4_e2m1f
