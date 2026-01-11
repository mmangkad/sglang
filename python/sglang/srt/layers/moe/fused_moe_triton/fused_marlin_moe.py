from typing import Callable, Optional

import torch

from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import moe_sum_reduce, silu_and_mul
    from sgl_kernel.scalar_type import scalar_types
    
    # Try to import swigluoai_and_mul if available
    try:
        from sgl_kernel import swigluoai_and_mul as _swigluoai_and_mul
        _has_swigluoai = True
    except ImportError:
        _has_swigluoai = False
        _swigluoai_and_mul = None


def get_scalar_type(num_bits: int, has_zp: bool):
    from sgl_kernel.scalar_type import scalar_types

    if has_zp:
        assert num_bits == 4
        return scalar_types.uint4
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128


def get_fp4_scalar_type():
    """Get the scalar type for FP4 (MXFP4/NVFP4) quantization."""
    from sgl_kernel.scalar_type import scalar_types

    return scalar_types.float4_e2m1f


def default_activation_func(
    activation: str, output: torch.Tensor, input: torch.Tensor
) -> None:
    """Apply activation function (SiLU or SwiGLU-OAI)."""
    if activation == "silu":
        silu_and_mul(input, output)
    elif activation == "swigluoai":
        if _has_swigluoai and _swigluoai_and_mul is not None:
            # alpha = 1.702, limit = 7.0
            _swigluoai_and_mul(input, output)
        else:
            # Fallback: implement swigluoai using SiLU if swigluoai_and_mul not available
            # swigluoai formula: output = silu(gate) * clamp(alpha * sigmoid(gate) + beta, -limit, limit) * up
            # For simplicity, fall back to silu if swigluoai is not available
            import warnings
            warnings.warn(
                "swigluoai_and_mul is not available in sgl_kernel, falling back to silu_and_mul. "
                "This may affect model accuracy."
            )
            silu_and_mul(input, output)
    else:
        raise ValueError(
            f"Unsupported activation: {activation}. "
            "Only silu and swigluoai activations are supported."
        )


@register_custom_op(out_shape="hidden_states")
def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
    inplace: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - g_idx1 (Optional[torch.Tensor]): The first set of act_order indices.
    - g_idx2 (Optional[torch.Tensor]): The second set of act_order indices.
    - sort_indices1 (Optional[torch.Tensor]): The first act_order input
        permutation.
    - sort_indices2 (Optional[torch.Tensor]): The second act_order input
        permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
    - num_bits (int): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (
        num_bits // 2
    ), "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert (
        hidden_states.dtype == w1_scale.dtype
    ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w1_scale.dtype ({w1_scale.dtype})"
    assert (
        hidden_states.dtype == w2_scale.dtype
    ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w2_scale.dtype ({w2_scale.dtype})"
    assert num_bits in [4, 8]

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16
    topk = topk_ids.shape[1]

    # M block size selection logic
    # TODO: tune this further for specific models
    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / E / block_size_m < 0.9:
            break

    if global_num_experts == -1:
        global_num_experts = E
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, global_num_experts
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

    scalar_type1 = get_scalar_type(num_bits, w1_zeros is not None)
    scalar_type2 = get_scalar_type(num_bits, w2_zeros is not None)

    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache13 = torch.empty(
        (M * topk_ids.shape[1] * max(2 * N, K),),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = intermediate_cache13[: M * topk_ids.shape[1] * 2 * N]
    intermediate_cache1 = intermediate_cache1.view(-1, 2 * N)
    intermediate_cache3 = intermediate_cache13[: M * topk_ids.shape[1] * K]
    intermediate_cache3 = intermediate_cache3.view(-1, K)

    use_atomic_add = (
        hidden_states.dtype == torch.half
        or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
    )

    intermediate_cache1 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
        hidden_states,
        intermediate_cache1,
        w1,
        None,  # b_bias_or_none
        w1_scale,
        None,  # global_scale_or_none
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=expert_map is not None,
        b_q_type_id=scalar_type1.id,
        size_m=M,
        size_n=2 * N,
        size_k=K,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    silu_and_mul(intermediate_cache1.view(-1, 2 * N), intermediate_cache2)

    if expert_map is not None:
        intermediate_cache3.zero_()

    intermediate_cache3 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
        intermediate_cache2,
        intermediate_cache3,
        w2,
        None,  # b_bias_or_none
        w2_scale,
        None,  # global_scale_or_none
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=expert_map is not None,
        b_q_type_id=scalar_type2.id,
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    ).view(-1, topk, K)

    output = hidden_states if inplace else torch.empty_like(hidden_states)

    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    moe_sum_reduce(
        intermediate_cache3,
        output,
        routed_scaling_factor,
    )
    return output


def fused_marlin_moe_mxfp4(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    activation: str = "silu",
    activation_func: Optional[Callable[[str, torch.Tensor, torch.Tensor], None]] = None,
    apply_router_weight_on_input: bool = False,
    expert_map: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    intermediate_cache13: Optional[torch.Tensor] = None,
    intermediate_cache2: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    input_dtype: Optional[torch.dtype] = None,
    is_k_full: bool = True,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Fused MoE kernel for MXFP4 quantized weights using the Marlin kernel.

    This function computes a Mixture of Experts (MoE) layer using MXFP4
    quantized weights (4-bit float with 32-element group size).

    After gptq_marlin_repack, weight shapes change from original FP4 packed format
    to Marlin tiled format with tile_size=16 and pack_factor=8:
    - w1: [E, K//16, 4*N] where K=hidden_size, N=intermediate_size
    - w2: [E, N//16, 2*K]

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer [M, K].
    - w1 (torch.Tensor): Marlin repacked gate+up weights [E, K//16, 4*N].
    - w2 (torch.Tensor): Marlin repacked down weights [E, N//16, 2*K].
    - w1_bias (Optional[torch.Tensor]): Bias for w1 [E, 2*N].
    - w2_bias (Optional[torch.Tensor]): Bias for w2 [E, K].
    - w1_scale (torch.Tensor): Scale for w1 quantization.
    - w2_scale (torch.Tensor): Scale for w2 quantization.
    - router_logits (torch.Tensor): Router logits (unused for MXFP4, kept for compatibility).
    - topk_weights (torch.Tensor): Top-k weights [M, topk].
    - topk_ids (torch.Tensor): Indices of top-k elements [M, topk].
    - global_num_experts (int): Total number of experts globally.
    - activation (str): Activation function name ("silu" or "swigluoai").
    - activation_func: Optional custom activation function.
    - apply_router_weight_on_input (bool): Whether to apply router weights on input.
    - expert_map (Optional[torch.Tensor]): Expert mapping for expert parallelism.
    - workspace (Optional[torch.Tensor]): Workspace tensor for Marlin kernel.
    - intermediate_cache13 (Optional[torch.Tensor]): Pre-allocated intermediate cache.
    - intermediate_cache2 (Optional[torch.Tensor]): Pre-allocated intermediate cache.
    - output (Optional[torch.Tensor]): Pre-allocated output tensor.
    - input_dtype (Optional[torch.dtype]): Input dtype for activation quantization.
    - is_k_full (bool): Whether K dimension is full (no act_order).
    - inplace (bool): Whether to write output in-place to hidden_states.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    if activation_func is None:
        activation_func = default_activation_func

    assert hidden_states.ndim == 2
    M, K = hidden_states.size()
    E = w1.size(0)
    
    # After Marlin repacking, the weight shapes are:
    # w1: [E, K // 16, 4*N] - where tile_size=16, pack_factor=8
    # w2: [E, N // 16, 2*K] - after gptq_marlin_repack
    # So: K = w1.size(1) * 16, and N = w2.size(1) * 16
    MARLIN_TILE_SIZE = 16
    N = w2.size(1) * MARLIN_TILE_SIZE  # Intermediate size
    topk = topk_ids.size(1)

    # Validate shapes (similar to vLLM)
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert topk_weights.dtype == torch.float32
    assert w1.size(1) * MARLIN_TILE_SIZE == K, (
        f"Hidden size mismatch w1: w1.size(1)={w1.size(1)}, "
        f"expected K // tile_size = {K // MARLIN_TILE_SIZE}"
    )

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
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            marlin_make_workspace_new,
        )

        workspace = marlin_make_workspace_new(hidden_states.device, 4)

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

    # Reshape caches
    intermediate_cache1 = intermediate_cache13[: M * topk * 2 * N].view(-1, 2 * N)
    intermediate_cache3 = intermediate_cache13[: M * topk * K].view(-1, K)
    intermediate_cache2 = intermediate_cache2.view(M * topk, N)

    # Get FP4 scalar type
    quant_type = get_fp4_scalar_type()

    use_atomic_add = (
        hidden_states.dtype == torch.half
        or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
    )

    # Handle activation quantization if needed
    a_scales1 = None
    gate_up_input = hidden_states
    if input_dtype == torch.float8_e4m3fn:
        # TODO: Implement input quantization for FP8 activations
        # For now, use BF16/FP16 input directly
        pass

    # First GEMM: gate_up projection
    intermediate_cache1 = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
        gate_up_input,
        intermediate_cache1,
        w1,
        w1_bias,  # bias
        w1_scale,
        None,  # global_scale_or_none
        None,  # zeros
        None,  # g_idx
        None,  # sort_indices
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=apply_router_weight_on_input,
        is_ep=expert_map is not None,
        b_q_type_id=quant_type.id,
        size_m=M,
        size_n=2 * N,
        size_k=K,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    # Apply activation function
    activation_func(activation, intermediate_cache2, intermediate_cache1.view(-1, 2 * N))

    if output is None:
        output = intermediate_cache3

    if expert_map is not None:
        output.zero_()

    # Second GEMM: down projection
    output = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
        intermediate_cache2,
        output,
        w2,
        w2_bias,  # bias
        w2_scale,
        None,  # global_scale_or_none
        None,  # zeros
        None,  # g_idx
        None,  # sort_indices
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=not apply_router_weight_on_input,
        is_ep=expert_map is not None,
        b_q_type_id=quant_type.id,
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    # Reduce across top-k experts
    output = output.view(-1, topk, K)

    if inplace:
        final_output = hidden_states
    else:
        final_output = torch.empty_like(hidden_states)

    moe_sum_reduce(output, final_output, 1.0)
    return final_output
