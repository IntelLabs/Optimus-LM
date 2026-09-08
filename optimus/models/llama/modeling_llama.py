from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_activation_checkpoint

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from .configuration_llama import LlamaParallelConfig

# Parallelization related functions, modules, and utils
from ..common.comms_ptfunctions import forward_identity_backward_allreduce
from ..common.comms_ptfunctions import forward_allreduce_backward_identity

from ..common.parallel_modules import ParallelEmbedding, ParallelLMHead
from ..common.parallel_converter_utils import populate_ColumnLinear_from_Linear, populate_RowLinear_from_Linear


class LlamaParallelOutput:
    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits

class LlamaParallelRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaParallelMLP(nn.Module):
    def __init__(self, config, pmap=None):
        super().__init__()
        self.config = config

        self.pmap = pmap
        self.intermediate_size_per_rank = config.intermediate_size // self.pmap.tensor_parallelism

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size_per_rank, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size_per_rank, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size_per_rank, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        if self.pmap.tensor_parallelism > 1:
            x = forward_identity_backward_allreduce.apply(x, self.pmap.tp_group)

        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        if self.pmap.tensor_parallelism > 1:
            down_proj = forward_allreduce_backward_identity.apply(down_proj, self.pmap.tp_group)
        return down_proj

    def set_parameters_from_full_module(self, module):    
        populate_ColumnLinear_from_Linear(module.gate_proj, self.gate_proj, self.pmap.tp_ind, tensor_parallelism=self.pmap.tensor_parallelism)
        populate_ColumnLinear_from_Linear(module.up_proj,   self.up_proj,   self.pmap.tp_ind, tensor_parallelism=self.pmap.tensor_parallelism)
        populate_RowLinear_from_Linear(   module.down_proj, self.down_proj, self.pmap.tp_ind, tensor_parallelism=self.pmap.tensor_parallelism)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class LlamaParallelAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, pmap=None):
        super().__init__()
        self.config = config

        self.pmap = pmap
        self.num_attention_heads_per_rank = config.num_attention_heads // self.pmap.tensor_parallelism
        self.num_key_value_heads_per_rank = config.num_key_value_heads // self.pmap.tensor_parallelism

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

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

    def forward(self, hidden_states, position_embeddings):
        if self.pmap.tensor_parallelism > 1:
            hidden_states = forward_identity_backward_allreduce.apply(hidden_states, self.pmap.tp_group)

        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

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

class LlamaParallelDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=0, pmap=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.pmap = pmap
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaParallelAttention(config, pmap=pmap)
        self.mlp = LlamaParallelMLP(config, pmap=pmap)
        self.input_layernorm = LlamaParallelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaParallelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_activation_checkpointing_for_norm = False
        if config.use_activation_checkpointing:
            self.use_activation_checkpointing_for_norm = config.activation_checkpointing_level & (1 << 2) # Bit 2 for Norm

    def forward(self, hidden_states, position_embeddings):
        # Attention
        residual = hidden_states
        if self.use_activation_checkpointing_for_norm:
            hidden_states = torch_activation_checkpoint(self.input_layernorm, hidden_states, use_reentrant=False)
        else:
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.use_activation_checkpointing_for_norm:
            hidden_states = torch_activation_checkpoint(self.post_attention_layernorm, hidden_states, use_reentrant=False)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    
    def set_parameters_from_full_module(self, full_module):
        self.self_attn.set_parameters_from_full_module(full_module.self_attn)
        self.mlp.set_parameters_from_full_module(full_module.mlp)
        self.input_layernorm.load_state_dict(full_module.input_layernorm.state_dict())
        self.post_attention_layernorm.load_state_dict(full_module.post_attention_layernorm.state_dict())

class LlamaParallelRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaParallelConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class LlamaParallelModel(nn.Module):
    def __init__(self, config, pmap=None):
        super().__init__()
        self.pmap = pmap

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if self.pmap.is_first_stage_rank:
            if self.pmap.tensor_parallelism > 1:
                self.embed_tokens = ParallelEmbedding(config.vocab_size, config.hidden_size, self.padding_idx,
                                            tensor_parallelism=self.pmap.tensor_parallelism, tp_ind=self.pmap.tp_ind, group=self.pmap.tp_group)
            else:
                self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        else:
            self.embed_tokens = nn.Identity()

        self.num_hidden_layers_per_rank = config.num_hidden_layers // self.pmap.pipeline_parallelism
        self.layer_start_idx = self.pmap.pp_ind * self.num_hidden_layers_per_rank
        self.layer_end_idx = (self.pmap.pp_ind + 1) * self.num_hidden_layers_per_rank
        self.layers = nn.ModuleList(
            [LlamaParallelDecoderLayer(config, layer_idx=layer_idx, pmap=self.pmap) for layer_idx in range(self.layer_start_idx, self.layer_end_idx)]
        )
        
        if self.pmap.is_last_stage_rank:
            self.norm = LlamaParallelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = nn.Identity()
        
        self.rotary_emb = LlamaParallelRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.num_hidden_layers = config.num_hidden_layers

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)

        position_ids = torch.arange(0, hidden_states.size(1), device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)
        hidden_states = self.norm(hidden_states)

        return hidden_states
    
    def set_parameters_from_full_module(self, module):
        if self.pmap.is_first_stage_rank:
            if self.pmap.tensor_parallelism > 1:
                self.embed_tokens.set_parameters_from_full_module(module.embed_tokens)
            else:
                self.embed_tokens.load_state_dict(module.embed_tokens.state_dict())
        
        for local_layer_idx in range(self.num_hidden_layers_per_rank):
            layer_idx = self.layer_start_idx + local_layer_idx
            self.layers[local_layer_idx].set_parameters_from_full_module(module.layers[layer_idx])

        if self.pmap.is_last_stage_rank:
            self.norm.load_state_dict(module.norm.state_dict())

class LlamaParallelForCausalLM(nn.Module):
    def __init__(self, config, pmap=None):
        super().__init__()
        self.config = config
        self.pmap = pmap

        self.model = LlamaParallelModel(config, pmap=self.pmap)
        self.vocab_size = config.vocab_size
        if self.pmap.is_last_stage_rank:
            if (self.pmap.tensor_parallelism > 1):
                self.lm_head = ParallelLMHead(config.hidden_size, config.vocab_size, bias=False,
                                    tensor_parallelism=self.pmap.tensor_parallelism, tp_ind=self.pmap.tp_ind, group=self.pmap.tp_group)
            else:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = nn.Identity()

    def loss_function(self, logits, labels, vocab_size):
        # Choosing first (S-1) for logits and last (S-1) for labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Cross entropy loss
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
        return loss

    def forward(self, input_ids, labels=None):
        hidden_states = self.model(input_ids)

        if (self.pmap.pipeline_parallelism > 1) and (not self.pmap.is_last_stage_rank):
            output_dict = {}
            output_dict["output"] = hidden_states
            output_dict["router_logits"] = None
            return output_dict

        logits = self.lm_head(hidden_states)

        if (self.pmap.pipeline_parallelism > 1) and (self.pmap.is_last_stage_rank):
            output_dict = {}
            output_dict["output"] = logits
            output_dict["router_logits"] = None
            return output_dict

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size)

        # print(f"Rank : {self.pmap.rank} ,  Aux Loss: {aux_loss:6.2f}", flush=True)
        return LlamaParallelOutput(loss=loss, logits=logits)
    
    def set_parameters_from_full_module(self, module):
        self.model.set_parameters_from_full_module(module.model)
        if self.pmap.is_last_stage_rank:
            if (self.pmap.tensor_parallelism > 1):
                self.lm_head.set_parameters_from_full_module(module.lm_head)
            else:
                self.lm_head.load_state_dict(module.lm_head.state_dict())

    def get_flops(self, context_size):
        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        # Attention
        attn_q_proj_flops = 6 * context_size * self.config.hidden_size * (self.config.num_attention_heads * head_dim)
        attn_k_proj_flops = 6 * context_size * self.config.hidden_size * (self.config.num_key_value_heads * head_dim)
        attn_v_proj_flops = 6 * context_size * self.config.hidden_size * (self.config.num_key_value_heads * head_dim)
        attn_o_proj_flops = 6 * context_size * (self.config.num_attention_heads * head_dim) * self.config.hidden_size

        attn_qkt_flops = 6 * self.config.num_attention_heads * (context_size * context_size * head_dim)
        attn_ktv_flops = 6 * self.config.num_attention_heads * (context_size * context_size * head_dim)

        # MLP
        mlp_gate_proj_flops = (6 * context_size * self.config.hidden_size * self.config.intermediate_size)
        mlp_up_proj_flops   = (6 * context_size * self.config.hidden_size * self.config.intermediate_size)
        mlp_down_proj_flops = (6 * context_size * self.config.intermediate_size * self.config.hidden_size)

        # LM Head
        lm_head_flops = (6 * context_size * self.config.hidden_size * self.config.vocab_size)

        # Groupings
        attn_flops = attn_q_proj_flops + attn_k_proj_flops + attn_v_proj_flops + attn_o_proj_flops + attn_qkt_flops + attn_ktv_flops
        mlp_flops = mlp_gate_proj_flops + mlp_up_proj_flops + mlp_down_proj_flops
        
        decoder_block_flops = 0
        decoder_block_flops += attn_flops
        decoder_block_flops += mlp_flops
        decoder_stack_flops = (self.config.num_hidden_layers // self.pmap.pipeline_parallelism) * decoder_block_flops

        # Calculate total FLOPS
        total_flops = decoder_stack_flops
        if self.pmap.is_last_stage_rank:
            total_flops += lm_head_flops

        if self.config.tensor_parallelism > 1:
            total_flops = total_flops / self.config.tensor_parallelism

        # if torch.distributed.is_initialized() and (torch.distributed.get_rank() == 0):
        if False:
            flops_desc_str  = f"FLOPS breakdown \n"
            flops_desc_str += f"Model          : {total_flops} \n"
            flops_desc_str += f"-Decoder stack : {decoder_stack_flops} \n"
            flops_desc_str += f"-LM head       : {lm_head_flops} \n"
            flops_desc_str += f"\n"
            flops_desc_str += f"Decoder        : {decoder_block_flops} \n"
            flops_desc_str += f"-Attention     : {attn_flops} \n"
            flops_desc_str += f"-MLP           : {mlp_flops} \n"

            print(flops_desc_str, flush=True)

        return total_flops
    
    def get_tp_replicated_param_list(self):
        params_sharded = []
        
        params_sharded.append("model.embed_tokens.weight")

        for i in range(len(self.model.layers)):
            params_sharded.append(f"model.layers.{i}.self_attn.q_proj.weight")
            params_sharded.append(f"model.layers.{i}.self_attn.k_proj.weight")
            params_sharded.append(f"model.layers.{i}.self_attn.v_proj.weight")
            params_sharded.append(f"model.layers.{i}.self_attn.o_proj.weight")

        for i in range(len(self.model.layers)):
            params_sharded.append(f"model.layers.{i}.mlp.gate_proj.weight")
            params_sharded.append(f"model.layers.{i}.mlp.up_proj.weight")
            params_sharded.append(f"model.layers.{i}.mlp.down_proj.weight")

        params_sharded.append("lm_head.weight")

        # Grouping gradients
        params_replicated = []
        for name,p in self.named_parameters():
            if p.grad is not None:
                if not name in params_sharded:
                    params_replicated.append(name)

        return params_replicated


"""
###############################################
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

causal_mask = torch.tril(torch.ones((q_len, q_len), dtype=torch.bool, device=attn_weights.device))
causal_mask = causal_mask.view(1, 1, q_len, q_len) 
mask_value = torch.finfo(attn_weights.dtype).min
mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
attn_output = torch.matmul(attn_weights, value_states)
attn_output = attn_output.transpose(1, 2).contiguous()
##############################################
"""