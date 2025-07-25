# Copyright 2024-2025 LMCache Authors.
#
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

# Third Party
from vllm.attention import Attention
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata
import torch

# First Party
from lmcache.v1.compute.attention.abstract import AttentionInterface
from lmcache.v1.compute.attention.metadata import LMCFlashAttnMetadata


class LMCFlashAttnBackend(AttentionInterface):
    """
    FlashAttention backend for LMCache.
    This backend uses the FlashAttention implementation
    for efficient attention computation.
    """

    def __init__(
        self,
        vllm_attn: Attention,
    ):
        self.vllm_attn = vllm_attn
        self.vllm_attn_impl: FlashAttentionImpl = vllm_attn.impl

        # TODO(Jiayi): remove this hardcode
        self.aot_schedule = False

    def forward_contiguous(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: LMCFlashAttnMetadata,
        **kwargs,
    ) -> torch.Tensor:
        # num_actual_tokens = query.shape[0]

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        cu_seqlens_k = attn_metadata.cu_seqlens_k
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len

        descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

        scheduler_metadata = self._schedule(
            batch_size=1,  # NOTE(Jiayi): Assuming batch size is 1,
            # since we are processing request by request.
            cu_query_lens=cu_seqlens_q,
            max_query_len=max_seqlen_q,
            seqlens=seqused_k,
            max_seq_len=max_seqlen_k,
            causal=True,  # Assuming causal attention
        )

        flash_attn_varlen_func(
            q=query,  # contiguous
            k=key,  # contiguous
            v=value,  # contiguous
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            # seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.vllm_attn_impl.scale,
            causal=True,
            alibi_slopes=self.vllm_attn_impl.alibi_slopes,
            window_size=self.vllm_attn_impl.sliding_window,
            block_table=None,
            softcap=self.vllm_attn_impl.logits_soft_cap,
            scheduler_metadata=scheduler_metadata,
            fa_version=self.vllm_attn_impl.vllm_flash_attn_version,
            q_descale=self.vllm_attn._q_scale.expand(descale_shape),
            k_descale=self.vllm_attn._k_scale.expand(descale_shape),
            v_descale=self.vllm_attn._v_scale.expand(descale_shape),
        )

        return output

    def _schedule(
        self, batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
    ):
        if self.aot_schedule:
            return get_scheduler_metadata(
                batch_size=batch_size,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_seq_len,
                cache_seqlens=seqlens,
                num_heads_q=self.vllm_attn_impl.num_heads_q,
                num_heads_kv=self.vllm_attn_impl.num_heads_kv,
                headdim=self.vllm_attn_impl.headdim,
                page_size=self.vllm_attn_impl.block_size,
                cu_seqlens_q=cu_query_lens,
                causal=causal,
                window_size=self.vllm_attn_impl.aot_sliding_window,
            )
        return None
