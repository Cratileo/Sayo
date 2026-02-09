from transformers import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import TransformersKwargs
from torch import nn
import torch
from transformers.utils.deprecation import deprecate_kwarg
from typing import Optional, Tuple, Callable, Union

@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def NeoAttnforward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    attention_mask_eager: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
    

    attention_interface: Callable = eager_attention_forward

    with torch.no_grad():
        _, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask_eager,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
    # attn_output = attn_o.reshape(*input_shape, -1).contiguous()
    # attn_output = self.o_proj(attn_output)

    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, _ = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    # value_states = repeat_kv(value, module.num_key_value_groups)
    # attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)).mul_(scaling)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        # attn_weights = attn_weights + causal_mask
        attn_weights.add_(causal_mask)

    if attn_weights.size(-2) == 1:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=query.dtype)
    # attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    # attn_output = torch.matmul(attn_weights, value_states)
    # attn_output = attn_output.transpose(1, 2).contiguous()

    return None, attn_weights


def get_top_entropy_indices_list(entropies, top_ratio=0.3):
    seq_len = entropies.size(1)
    k = max(1, int(seq_len * top_ratio))
    _, top_indices = torch.topk(entropies, k, dim=-1)
    indices_list = top_indices.tolist()
    
    return indices_list