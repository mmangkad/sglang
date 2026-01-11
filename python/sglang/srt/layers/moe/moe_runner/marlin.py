from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

MARLIN_MOE_WORKSPACE: Optional[torch.Tensor] = None


@dataclass
class MarlinRunnerInput(RunnerInput):
    """Input bundle passed to the Marlin runner core."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinRunnerOutput(RunnerOutput):
    """Output bundle returned from the Marlin runner core."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinMoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by the Marlin backend."""

    w13_qweight: torch.Tensor
    w2_qweight: torch.Tensor
    w13_scales: torch.Tensor
    w2_scales: torch.Tensor
    w13_g_idx_sort_indices: Optional[torch.Tensor]
    w2_g_idx_sort_indices: Optional[torch.Tensor]
    weight_bits: int

    # GPTQ specific (Optional)
    w13_g_idx: Optional[torch.Tensor] = None
    w2_g_idx: Optional[torch.Tensor] = None
    is_k_full: bool = True

    # AWQ specific (Optional)
    w13_qzeros: Optional[torch.Tensor] = None
    w2_qzeros: Optional[torch.Tensor] = None

    # Optional
    expert_map: Optional[torch.Tensor] = None


@dataclass
class MarlinMxfp4MoeQuantInfo(MoeQuantInfo):
    """Quantization payload for MXFP4 quantized weights with Marlin backend.
    
    After Marlin repacking (gptq_marlin_repack), weight shapes are:
    - w13_qweight: [E, K // 16, 4*N] where K=hidden_size, N=intermediate_size
    - w2_qweight: [E, N // 16, 2*K]
    
    The tile_size is 16 and pack_factor is 8 for 4-bit weights.
    To get dimensions: K = w13.size(1) * 16, N = w2.size(1) * 16
    """

    w13_qweight: torch.Tensor  # Marlin repacked: [E, K//16, 4*N]
    w2_qweight: torch.Tensor   # Marlin repacked: [E, N//16, 2*K]
    w13_scales: torch.Tensor   # float8_e8m0fnu scales
    w2_scales: torch.Tensor    # float8_e8m0fnu scales
    
    # Bias tensors (optional)
    w13_bias: Optional[torch.Tensor] = None
    w2_bias: Optional[torch.Tensor] = None
    
    # Expert map for expert parallelism (optional)
    expert_map: Optional[torch.Tensor] = None
    
    # Global number of experts
    global_num_experts: int = -1
    
    # Activation function name
    activation: str = "silu"


@register_fused_func("none", "marlin")
def fused_experts_none_to_marlin(
    dispatch_output: StandardDispatchOutput,
    quant_info: MarlinMoeQuantInfo | MarlinMxfp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Fused MoE kernel for Marlin backend supporting both regular quantization and MXFP4."""
    global MARLIN_MOE_WORKSPACE
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import (
        BypassedTopKOutput,
        StandardTopKOutput,
        TopKOutputChecker,
        fused_topk,
    )

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    # Handle different topk_output formats
    if TopKOutputChecker.format_is_bypassed(topk_output):
        # BypassedTopKOutput - need to compute topk_weights and topk_ids from router_logits
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=topk_output.router_logits,
            topk=topk_output.topk_config.top_k,
            renormalize=topk_output.topk_config.renormalize,
        )
        router_logits = topk_output.router_logits
    elif TopKOutputChecker.format_is_standard(topk_output):
        # StandardTopKOutput - already has topk_weights and topk_ids
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids
        router_logits = topk_output.router_logits
    else:
        raise ValueError(f"Unsupported topk_output format: {type(topk_output)}")

    # Check if this is MXFP4 quantization
    if isinstance(quant_info, MarlinMxfp4MoeQuantInfo):
        # MXFP4 path
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe_mxfp4,
        )
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            marlin_make_workspace_new,
        )

        if (
            MARLIN_MOE_WORKSPACE is None
            or MARLIN_MOE_WORKSPACE.device != hidden_states.device
        ):
            MARLIN_MOE_WORKSPACE = marlin_make_workspace_new(
                hidden_states.device, max_blocks_per_sm=4
            )

        output = fused_marlin_moe_mxfp4(
            hidden_states=hidden_states,
            w1=quant_info.w13_qweight,
            w2=quant_info.w2_qweight,
            w1_bias=quant_info.w13_bias,
            w2_bias=quant_info.w2_bias,
            w1_scale=quant_info.w13_scales,
            w2_scale=quant_info.w2_scales,
            router_logits=router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=quant_info.global_num_experts,
            activation=quant_info.activation,
            apply_router_weight_on_input=False,
            expert_map=quant_info.expert_map,
            workspace=MARLIN_MOE_WORKSPACE,
            inplace=runner_config.inplace,
        ).to(hidden_states.dtype)

        return StandardCombineInput(
            hidden_states=output,
        )
    else:
        # Regular Marlin quantization path (GPTQ/AWQ)
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

        assert runner_config.activation == "silu", "Only SiLU activation is supported."

        if (
            MARLIN_MOE_WORKSPACE is None
            or MARLIN_MOE_WORKSPACE.device != hidden_states.device
        ):
            MARLIN_MOE_WORKSPACE = marlin_make_workspace(
                hidden_states.device, max_blocks_per_sm=4
            )

        output = fused_marlin_moe(
            hidden_states=hidden_states,
            w1=quant_info.w13_qweight,
            w2=quant_info.w2_qweight,
            w1_scale=quant_info.w13_scales,
            w2_scale=quant_info.w2_scales,
            gating_output=router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=quant_info.expert_map,
            g_idx1=quant_info.w13_g_idx,
            g_idx2=quant_info.w2_g_idx,
            sort_indices1=quant_info.w13_g_idx_sort_indices,
            sort_indices2=quant_info.w2_g_idx_sort_indices,
            w1_zeros=quant_info.w13_qzeros,
            w2_zeros=quant_info.w2_qzeros,
            workspace=MARLIN_MOE_WORKSPACE,
            num_bits=quant_info.weight_bits,
            is_k_full=quant_info.is_k_full,
            inplace=runner_config.inplace,
            routed_scaling_factor=runner_config.routed_scaling_factor,
        ).to(hidden_states.dtype)

        return StandardCombineInput(
            hidden_states=output,
        )
