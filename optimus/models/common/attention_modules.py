import torch
import torch.nn as nn

from .parallel_converter_utils import populate_ColumnLinear_from_Linear, populate_RowLinear_from_Linear

from .comms_ptfunctions import forward_identity_backward_allreduce
from .comms_ptfunctions import forward_allreduce_backward_identity
from .norm_modules import RMSNorm

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ParallelAttention(nn.Module):
    def __init__(self, config, pmap = None):
        super().__init__()
        self.config = config

        self.pmap = pmap
        self.num_attention_heads_per_rank = config.num_attention_heads // self.pmap.tensor_parallelism
        self.num_key_value_heads_per_rank = config.num_key_value_heads // self.pmap.tensor_parallelism

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_attention_heads_per_rank * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads_per_rank * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads_per_rank * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads_per_rank * self.head_dim, config.hidden_size, bias=False
        )

        if config.use_qk_norm:
            assert pmap.tensor_parallelism == 1, "QK Norm is only supported for non-tensor-parallel case"
            self.q_norm = RMSNorm(config.num_attention_heads * self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(config.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings):
        if self.pmap.tensor_parallelism > 1:
            hidden_states = forward_identity_backward_allreduce.apply(hidden_states, self.pmap.tp_group)

        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        if not self.config.skip_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(bsz, q_len, self.num_attention_heads_per_rank, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads_per_rank, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads_per_rank, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                is_causal=True
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_attention_heads_per_rank * self.head_dim)
        
        attn_output = self.o_proj(attn_output)

        if self.pmap.tensor_parallelism > 1:
            attn_output = forward_allreduce_backward_identity.apply(attn_output, self.pmap.tp_group)

        return attn_output

    def set_parameters_from_full_module(self, module):
        populate_ColumnLinear_from_Linear(module.q_proj, self.q_proj, self.pmap.tp_ind, tensor_parallelism=self.pmap.tensor_parallelism)
        populate_ColumnLinear_from_Linear(module.k_proj, self.k_proj, self.pmap.tp_ind, tensor_parallelism=self.pmap.tensor_parallelism)
        populate_ColumnLinear_from_Linear(module.v_proj, self.v_proj, self.pmap.tp_ind, tensor_parallelism=self.pmap.tensor_parallelism)
        populate_RowLinear_from_Linear(   module.o_proj, self.o_proj, self.pmap.tp_ind, tensor_parallelism=self.pmap.tensor_parallelism)
