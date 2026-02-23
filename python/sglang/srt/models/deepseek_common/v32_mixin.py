# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from transformers import PretrainedConfig

from sglang.srt.configs.model_config import (
    get_nsa_index_head_dim,
    get_nsa_index_n_heads,
    get_nsa_index_topk,
    is_deepseek_nsa,
)
from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer
from sglang.srt.layers.attention.nsa.utils import (
    can_cp_split,
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    is_nsa_enable_prefill_cp,
    nsa_use_prefill_cp,
    prepare_input_dp_with_cp_dsa,
)
from sglang.srt.layers.communicator import LayerCommunicator, get_attn_tp_context
from sglang.srt.layers.communicator_nsa_cp import NSACPLayerCommunicator
from sglang.srt.layers.dp_attention import get_attention_cp_rank, get_attention_cp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args


class DeepseekV32Mixin:
    def init_v32_attention(
        self,
        *,
        config: PretrainedConfig,
        hidden_size: int,
        qk_rope_head_dim: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling: Optional[Dict[str, Any]],
        quant_config,
        layer_id: int,
        alt_stream: Optional[torch.cuda.Stream],
        prefix: str,
    ) -> None:
        self.use_nsa = is_deepseek_nsa(config)
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            assert self.use_nsa, "CP currently only supports deepseek v3.2 model"
        self.cp_size = (
            get_attention_cp_size()
            if self.nsa_enable_prefill_cp and self.use_nsa
            else None
        )

        if self.use_nsa:
            self._init_nsa_indexer(
                config=config,
                hidden_size=hidden_size,
                qk_rope_head_dim=qk_rope_head_dim,
                q_lora_rank=q_lora_rank,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                quant_config=quant_config,
                layer_id=layer_id,
                alt_stream=alt_stream,
                prefix=prefix,
            )

    def _init_nsa_indexer(
        self,
        *,
        config: PretrainedConfig,
        hidden_size: int,
        qk_rope_head_dim: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling: Optional[Dict[str, Any]],
        quant_config,
        layer_id: int,
        alt_stream: Optional[torch.cuda.Stream],
        prefix: str,
    ) -> None:
        is_neox_style = not getattr(config, "indexer_rope_interleave", False)
        self.indexer = Indexer(
            hidden_size=hidden_size,
            index_n_heads=get_nsa_index_n_heads(config),
            index_head_dim=get_nsa_index_head_dim(config),
            rope_head_dim=qk_rope_head_dim,
            index_topk=get_nsa_index_topk(config),
            q_lora_rank=q_lora_rank,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            scale_fmt="ue8m0",
            block_size=128,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            prefix=prefix,
            quant_config=quant_config,
            layer_id=layer_id,
            alt_stream=alt_stream,
        )

    def _fuse_rope_for_trtllm_mla(self, forward_batch: ForwardBatch) -> bool:
        if self.current_attention_backend == "nsa":
            return (
                get_global_server_args().nsa_decode_backend == "trtllm"
                or get_global_server_args().nsa_prefill_backend == "trtllm"
            ) and forward_batch.attn_backend.kv_cache_dtype == torch.float8_e4m3fn

        return (
            self.current_attention_backend == "trtllm_mla"
            and (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
            )
            and forward_batch.attn_backend.data_type == torch.float8_e4m3fn
        )

    def rebuild_cp_kv_cache(
        self,
        latent_cache: torch.Tensor,
        forward_batch: ForwardBatch,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
    ):
        latent_cache[..., : self.kv_lora_rank] = k_nope.squeeze(1)
        latent_cache[..., self.kv_lora_rank :] = k_pe.squeeze(1)
        latent_cache_output = cp_all_gather_rerange_output(
            latent_cache.contiguous(),
            self.cp_size,
            forward_batch,
            torch.cuda.current_stream(),
        )
        k_nope = latent_cache_output[..., : self.kv_lora_rank].unsqueeze(1)
        k_pe = latent_cache_output[..., self.kv_lora_rank :].unsqueeze(1)
        return k_nope, k_pe

    def init_v32_layer_cp(self) -> None:
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

    def create_layer_communicator(
        self,
        *,
        layer_scatter_modes,
        input_layernorm,
        post_attention_layernorm,
        is_last_layer: bool,
        qkv_latent_func,
    ):
        layer_communicator_cls = (
            NSACPLayerCommunicator if self.nsa_enable_prefill_cp else LayerCommunicator
        )
        return layer_communicator_cls(
            layer_scatter_modes=layer_scatter_modes,
            input_layernorm=input_layernorm,
            post_attention_layernorm=post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=is_last_layer,
            qkv_latent_func=qkv_latent_func,
        )

    def init_v32_model_cp(self) -> None:
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        self.cp_size = get_attention_cp_size() if self.nsa_enable_prefill_cp else None

    def maybe_split_model_inputs_for_cp(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        is_first_pp_rank: bool,
    ):
        if nsa_use_prefill_cp(forward_batch):
            if is_first_pp_rank:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)
        return hidden_states, positions

    def maybe_gather_model_outputs_for_cp(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if not nsa_use_prefill_cp(forward_batch):
            return hidden_states

        return cp_all_gather_rerange_output(
            hidden_states,
            self.cp_size,
            forward_batch,
            torch.cuda.current_stream(),
        )

    def init_v32_for_causal_lm(self, config: PretrainedConfig) -> None:
        self.use_nsa = is_deepseek_nsa(config)
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_rank = get_attention_cp_rank()
            self.cp_size = get_attention_cp_size()
        else:
            self.cp_rank = None
            self.cp_size = None

    def init_v32_attn_tp_context(self, config: PretrainedConfig) -> None:
        q_lora_rank = config.q_lora_rank if hasattr(config, "q_lora_rank") else None
        get_attn_tp_context().init_context(q_lora_rank, self.use_nsa)

    def maybe_prepare_nsa_cp_metadata(
        self,
        input_length: int,
        forward_batch: ForwardBatch,
    ) -> None:
        if not self.nsa_enable_prefill_cp:
            return

        if can_cp_split(input_length, self.cp_size, self.use_nsa, forward_batch):
            forward_batch.nsa_cp_metadata = prepare_input_dp_with_cp_dsa(
                input_length,
                self.cp_rank,
                self.cp_size,
                forward_batch.seq_lens_cpu.tolist(),
            )
