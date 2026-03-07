# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    marlin_make_workspace,
    marlin_permute_bias,
    marlin_permute_scales,
)
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack

logger = logging.getLogger(__name__)
_warned_fp4_marlin = False


def _warn_fp4_marlin_once() -> None:
    global _warned_fp4_marlin
    if _warned_fp4_marlin:
        return
    logger.warning(
        "MXFP4 weights are being executed through the Marlin MoE kernel. "
        "This uses weight-only FP4 packing and may trade some performance for compatibility."
    )
    _warned_fp4_marlin = True


def mxfp4_marlin_process_scales(
    marlin_scales: torch.Tensor, input_dtype: torch.dtype | None = None
) -> torch.Tensor:
    if input_dtype is None or input_dtype.itemsize == 2:
        marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
            marlin_scales.size(0), -1
        )

    marlin_scales = marlin_scales.to(torch.float8_e8m0fnu)
    if input_dtype == torch.float8_e4m3fn:
        marlin_scales = marlin_scales.view(torch.uint8)
        assert marlin_scales.max() <= 249
        marlin_scales = (marlin_scales + 6).view(torch.float8_e8m0fnu)
    return marlin_scales


def prepare_moe_fp4_layer_for_marlin(
    layer: torch.nn.Module, input_dtype: torch.dtype | None = None
) -> None:
    if not _is_cuda:
        raise RuntimeError("MXFP4 Marlin MoE is only supported on CUDA.")

    _warn_fp4_marlin_once()

    if input_dtype is not None and input_dtype.itemsize == 1:
        raise RuntimeError("MXFP4 weight + INT8/FP8 activation is not supported.")

    param_dtype = getattr(layer, "params_dtype", layer.w13_weight_bias.dtype)
    layer.workspace = marlin_make_workspace(layer.w13_weight.device, 4)
    perm = torch.empty(0, dtype=torch.int, device=layer.w13_weight.device)
    size_map = {
        "w13": (layer.w13_weight.shape[1], layer.w13_weight.shape[2] * 2),
        "w2": (layer.w2_weight.shape[1], layer.w2_weight.shape[2] * 2),
    }

    for name in ["w13_weight", "w2_weight"]:
        weight = getattr(layer, name)
        tensor_list = []

        if name == "w13_weight":
            size_n, size_k = size_map["w13"]
        else:
            size_n, size_k = size_map["w2"]

        for i in range(weight.shape[0]):
            qweight = weight[i].view(torch.int32).T.contiguous()
            marlin_qweight = gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
            )
            tensor_list.append(marlin_qweight)

        setattr(
            layer,
            name,
            torch.nn.Parameter(
                torch.stack(tensor_list, dim=0).contiguous(), requires_grad=False
            ),
        )

    for name in ["w13", "w2"]:
        scales = getattr(layer, f"{name}_weight_scale").view(torch.float8_e8m0fnu)
        scales = scales.to(param_dtype)
        tensor_list = []

        if name == "w13":
            size_n, size_k = size_map["w13"]
        else:
            size_n, size_k = size_map["w2"]

        for i in range(scales.shape[0]):
            marlin_scales = marlin_permute_scales(
                s=scales[i].T.contiguous(),
                size_k=size_k,
                size_n=size_n,
                group_size=32,
            )
            marlin_scales = mxfp4_marlin_process_scales(
                marlin_scales, input_dtype=input_dtype
            )
            tensor_list.append(marlin_scales)

        setattr(
            layer,
            f"{name}_weight_scale",
            torch.nn.Parameter(
                torch.stack(tensor_list, dim=0).contiguous(), requires_grad=False
            ),
        )

    for name in ["w13_weight_bias", "w2_weight_bias"]:
        if not hasattr(layer, name):
            continue

        bias = getattr(layer, name).to(param_dtype)
        tensor_list = []
        for i in range(bias.shape[0]):
            tensor_list.append(marlin_permute_bias(bias[i]))

        setattr(
            layer,
            name,
            torch.nn.Parameter(
                torch.stack(tensor_list, dim=0).contiguous(), requires_grad=False
            ),
        )
