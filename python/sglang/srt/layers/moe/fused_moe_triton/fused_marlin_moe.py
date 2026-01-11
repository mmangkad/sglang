# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_marlin_moe.py
"""Fused MoE utilities for Marlin MXFP4."""

from typing import Callable, Optional

import torch

from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import silu_and_mul
    from sgl_kernel.scalar_type import scalar_types


def get_scalar_type(num_bits: int, has_zp: bool, is_mxfp4: bool = False):
    if is_mxfp4:
        return scalar_types.float4_e2m1f
    if has_zp:
        assert num_bits == 4
        return scalar_types.uint4
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128


def default_activation_func(
    activation: str, output: torch.Tensor, input: torch.Tensor
) -> None:
    """Apply activation function for MoE layer."""
    if activation == "silu":
        silu_and_mul(input, output)
    elif activation == "swigluoai":
        # SwigluOAI activation used by GPT-OSS models
        # Try to use the optimized kernel if available, otherwise fallback to PyTorch
        try:
            torch.ops.sgl_kernel.swigluoai_and_mul(output, input)
        except (AttributeError, RuntimeError):
            # Fallback to PyTorch implementation
            _swigluoai_and_mul_pytorch(output, input, alpha=1.702, limit=7.0)
    else:
        raise ValueError(
            f"Unsupported activation: {activation}. "
            "Only silu and swigluoai activations are supported."
        )


def _swigluoai_and_mul_pytorch(
    output: torch.Tensor, input: torch.Tensor, alpha: float = 1.702, limit: float = 7.0
) -> None:
    """
    SwigluOAI activation for split format input [gate, up].
    Formula: gate * sigmoid(gate * alpha) * (up + 1)
    with clamping: gate clamped to max=limit, up clamped to [-limit, limit]
    """
    d = input.shape[-1] // 2
    gate = input[..., :d]
    up = input[..., d:]

    # Clamp values
    clamped_gate = gate.clamp(max=limit)
    clamped_up = up.clamp(min=-limit, max=limit)

    # Compute swigluoai: gate * sigmoid(gate * alpha) * (up + 1)
    glu = clamped_gate * torch.sigmoid(clamped_gate * alpha)
    output.copy_(glu * (clamped_up + 1))


def marlin_moe_intermediate_size(w1: torch.Tensor, w2: torch.Tensor) -> int:
    """Get intermediate size from Marlin weight tensors."""
    return w2.shape[1] * 16


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: str = "silu",
    activation_func: Callable[
        [str, torch.Tensor, torch.Tensor], None
    ] = default_activation_func,
    moe_sum: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    expert_map: Optional[torch.Tensor] = None,
    input_global_scale1: Optional[torch.Tensor] = None,
    input_global_scale2: Optional[torch.Tensor] = None,
    global_scale1: Optional[torch.Tensor] = None,
    global_scale2: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    intermediate_cache13: Optional[torch.Tensor] = None,
    intermediate_cache2: Optional[torch.Tensor] = None,
    is_k_full: bool = True,
    output: Optional[torch.Tensor] = None,
    input_dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - bias1 (torch.Tensor|None): Optional bias for w1.
    - bias2 (torch.Tensor|None): Optional bias for w2.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - gating_output (torch.Tensor|None): The output of the gating
        operation (before softmax).
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - quant_type_id (int): The scalar type ID for quantization.
    - apply_router_weight_on_input (bool): Whether to apply router weights on input.
    - global_num_experts (int): Global number of experts.
    - activation (str): Activation function name.
    - activation_func: Custom activation function.
    - moe_sum: Custom MoE sum function.
    - expert_map (torch.Tensor|None): Expert mapping tensor.
    - g_idx1 (torch.Tensor|None): The first set of act_order indices.
    - g_idx2 (torch.Tensor|None): The second set of act_order indices.
    - sort_indices1 (torch.Tensor|None): The first act_order input permutation.
    - sort_indices2 (torch.Tensor|None): The second act_order input permutation.
    - w1_zeros (torch.Tensor|None): Optional zero points to be used for w1.
    - w2_zeros (torch.Tensor|None): Optional zero points to be used for w2.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    if inplace:
        assert output is None, "Conflicting request"

    # Determine num_bits from quant_type_id by comparing with known type IDs
    bit4_type_ids = [
        scalar_types.uint4.id,
        scalar_types.uint4b8.id,
        scalar_types.float4_e2m1f.id,
    ]
    num_bits = 4 if quant_type_id in bit4_type_ids else 8

    M, K = hidden_states.size()
    E = w1.size(0)
    topk = topk_ids.size(1)
    N = marlin_moe_intermediate_size(w1, w2)

    # Check constraints.
    if gating_output is not None:
        assert gating_output.size(0) == M, "Number of tokens mismatch"
    assert w1.size(1) * 16 == K, "Hidden size mismatch w1"
    assert w2.size(2) // (num_bits // 2) == K, "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert num_bits in [4, 8]
    assert topk_weights.dtype == torch.float32

    # M block size selection logic
    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / E / block_size_m < 0.9:
            break

    if input_dtype is not None and input_dtype.itemsize == 1:
        block_size_m = max(block_size_m, 16)

    if global_num_experts == -1:
        global_num_experts = E
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_size_m,
        global_num_experts,
    )

    if workspace is None:
        max_workspace_size = (max(2 * N, K) // 64) * (
            sorted_token_ids.size(0) // block_size_m
        )
        device = hidden_states.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_workspace_size = min(max_workspace_size, sms * 4)
        workspace = torch.zeros(
            max_workspace_size, dtype=torch.int, device=device, requires_grad=False
        )

    if intermediate_cache13 is None:
        intermediate_cache13 = torch.empty(
            (M * topk * max(2 * N, K),),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    if intermediate_cache2 is None:
        intermediate_cache2 = torch.empty(
            (M * topk, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    intermediate_cache1 = intermediate_cache13[: M * topk * 2 * N].view(-1, 2 * N)
    intermediate_cache3 = intermediate_cache13[: M * topk * K].view(-1, K)

    # First GEMM (gate_up_proj)
    intermediate_cache1 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
        hidden_states,
        intermediate_cache1,
        w1,
        bias1,
        w1_scale,
        global_scale1,
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        block_size_m,
        topk,
        apply_router_weight_on_input,
        expert_map is not None,
        quant_type_id,
        M,
        2 * N,
        K,
        is_k_full,
        False,  # use_atomic_add
        True,   # use_fp32_reduce
        False,  # is_zp_float
    )

    # Apply activation function
    activation_func(
        activation, intermediate_cache2, intermediate_cache1.view(-1, 2 * N)
    )

    if output is None:
        output = intermediate_cache3

    if expert_map is not None:
        output.zero_()

    # Second GEMM (down_proj)
    output = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
        intermediate_cache2,
        output,
        w2,
        bias2,
        w2_scale,
        global_scale2,
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        block_size_m,
        1,  # top_k
        not apply_router_weight_on_input,
        expert_map is not None,
        quant_type_id,
        M * topk,
        K,
        N,
        is_k_full,
        False,  # use_atomic_add
        True,   # use_fp32_reduce
        False,  # is_zp_float
    )

    # Reshape for reduction
    moe_output = output.view(-1, topk, K)

    # Create output buffer
    if inplace:
        final_output = hidden_states
    else:
        final_output = torch.empty_like(hidden_states)

    # Final reduction
    if moe_sum is None:
        return torch.sum(moe_output, dim=1, out=final_output)
    else:
        return moe_sum(moe_output, final_output)
