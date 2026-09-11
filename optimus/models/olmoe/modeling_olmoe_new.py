from typing import List, Optional, Tuple, Union

import torch

import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint as torch_activation_checkpoint

from ..common.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

# Parallelization related functions, modules, and utils
from ..common.comms_ptfunctions import forward_allgather_backward_split
from ..common.comms_ptfunctions import forward_identity_backward_allreduce
from ..common.comms_ptfunctions import forward_reducescatter_backward_allgather
from ..common.comms_ptfunctions import forward_identity_backward_synchronize
from ..common.comms_ptfunctions import forward_synchronize_backward_identity
from ..common.comms_ptfunctions import forward_allgather_backward_reducescatter

from ..common.parallel_modules import ParallelEmbedding, ParallelLMHead

from ..common.parallel_converter_utils import populate_ColumnLinear_from_Linear, populate_RowLinear_from_Linear

from ..common.ral_functions import layer_level_load_balancing_loss_func

from ...profilers import record_pcl_function

# Copied from transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func
def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class OlmoeParallelOutput:
    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits

class OlmoeParallelRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Olmoe
class OlmoeParallelRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
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

class OlmoeParallelAttention(nn.Module):
    def __init__(self, config, pmap=None):
        super().__init__()
        # Parallelism related
        self.config = config
        self.pmap = pmap

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = OlmoeParallelRMSNorm(self.num_attention_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = OlmoeParallelRMSNorm(self.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)

        self.use_activation_checkpointing_for_norm = False
        self.use_activation_checkpointing_for_attn = False
        if config.use_activation_checkpointing:
            self.use_activation_checkpointing_for_attn = config.activation_checkpointing_level & (1 << 1) # Bit 1 for Attention
            if not self.use_activation_checkpointing_for_attn:
                self.use_activation_checkpointing_for_norm = config.activation_checkpointing_level & (1 << 2) # Bit 2 for Norm
        
    def forward(self, hidden_states, position_embeddings):
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if not self.config.skip_qk_norm:
            if self.use_activation_checkpointing_for_norm:
                query_states = torch_activation_checkpoint(self.q_norm, query_states, use_reentrant=False)
                key_states = torch_activation_checkpoint(self.k_norm, key_states, use_reentrant=False)
            else:
                query_states = self.q_norm(query_states)
                key_states = self.k_norm(key_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

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
        attn_output = attn_output.view(bsz, q_len, self.num_attention_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output

    def set_parameters_from_full_module(self, module):
        self.load_state_dict(module.state_dict())

class OlmoeParallelMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class OlmoeParallelSparseMoeBlock(nn.Module):
    def __init__(self, config, pmap=None):
        super().__init__()
        # Parallelism related
        self.pmap = pmap
        self.num_experts_per_rank = config.num_experts // config.expert_parallelism
        self.expert_start_idx = self.pmap.ep_ind * self.num_experts_per_rank
        self.expert_end_idx = (self.pmap.ep_ind + 1) * self.num_experts_per_rank

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.output_router_logits = config.output_router_logits

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([OlmoeParallelMLP(config) for _ in range(self.num_experts_per_rank)])

        self.use_activation_checkpointing_for_moe = False
        if config.use_activation_checkpointing:
            self.use_activation_checkpointing_for_moe = config.activation_checkpointing_level & (1 << 0) # Bit 0 for MLP
        
        self.mask = None
        self.force_uniform_routing = config.force_uniform_routing

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        router_logits = self.gate(hidden_states.view(-1, hidden_dim)) # router_logits: (batch * sequence_length, n_experts)

        #####################################################################
        if self.force_uniform_routing:
            if self.mask is None:
                from optimus.utils import get_load_balanced_expert_mask
                self.mask = get_load_balanced_expert_mask(sequence_length, self.num_experts, self.num_experts_per_tok, self.pmap.ep_ind)
                self.mask = (self.mask == False).to(hidden_states.device)  # Invert the mask to use with `masked_fill`
            router_logits = router_logits.masked_fill(self.mask, float('-inf'))
        ######################################################################

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        def expert_computation_block(hidden_states, selected_experts, routing_weights):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)

            if self.pmap.expert_parallelism > 1:
                hidden_states = forward_allgather_backward_reducescatter.apply(hidden_states, self.pmap.ep_ind, self.pmap.expert_parallelism, self.pmap.ep_group)
                routing_weights = forward_allgather_backward_reducescatter.apply(routing_weights, self.pmap.ep_ind, self.pmap.expert_parallelism, self.pmap.ep_group)
                selected_experts = forward_allgather_backward_split.apply(selected_experts, self.pmap.ep_ind, self.pmap.expert_parallelism, self.pmap.ep_group)

            final_hidden_states = torch.zeros(
                (self.pmap.expert_parallelism * batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )

            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be selected
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

            if self.pmap.expert_parallelism > 1:
                hidden_states = forward_identity_backward_synchronize.apply(hidden_states, self.pmap.ep_group)

            # Loop over all available experts in the model and perform the computation on each expert
            for local_expert_idx in range(self.num_experts_per_rank):
                expert_idx = self.expert_start_idx + local_expert_idx
                expert_layer = self.experts[local_expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])

                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                # final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype)
                final_hidden_states.index_add_(0, top_x, current_hidden_states)
            
            if self.pmap.expert_parallelism > 1:
                forward_synchronize_backward_identity.apply(self.pmap.ep_group)
                final_hidden_states = forward_reducescatter_backward_allgather.apply(final_hidden_states, self.pmap.ep_ind, self.pmap.expert_parallelism, self.pmap.ep_group, False)
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            
            return final_hidden_states

        args = [hidden_states, selected_experts, routing_weights]
        if self.use_activation_checkpointing_for_moe:
            final_hidden_states = torch_activation_checkpoint(expert_computation_block, *args, use_reentrant=False)
        else:
            final_hidden_states = expert_computation_block(*args)

        if self.output_router_logits:
            return final_hidden_states, router_logits
        else:
            return final_hidden_states, None

    def set_parameters_from_full_module(self, module):
        self.gate.load_state_dict(module.gate.state_dict())
        for local_expert_idx in range(self.num_experts_per_rank):
            self.experts[local_expert_idx].load_state_dict(module.experts[self.expert_start_idx + local_expert_idx].state_dict())

class OlmoeParallelDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=0, pmap=None, memory_tracker=None):
        super().__init__()
        self.self_attn = OlmoeParallelAttention(config, pmap=pmap)
        # from ..common.attention_modules import ParallelAttention
        # self.self_attn = ParallelAttention(config, pmap=pmap)
        if config.use_fast_moe:
            from ..common.moe import FastOlmoeParallelSparseMoeBlock
            self.mlp = FastOlmoeParallelSparseMoeBlock(config, pmap=pmap)
        else:
            self.mlp = OlmoeParallelSparseMoeBlock(config, pmap=pmap)
        self.input_layernorm = OlmoeParallelRMSNorm(config.hidden_size)
        self.post_attention_layernorm = OlmoeParallelRMSNorm(config.hidden_size)

        self.layer_idx = layer_idx
        self.memory_tracker = memory_tracker
        self.config = config

        self.use_activation_checkpointing_for_attn = False
        self.use_activation_checkpointing_for_norm = False
        if config.use_activation_checkpointing:
            self.use_activation_checkpointing_for_attn = config.activation_checkpointing_level & (1 << 1) # Bit 1 for Attention
            self.use_activation_checkpointing_for_norm = config.activation_checkpointing_level & (1 << 2) # Bit 2 for Norm

    def forward(self, hidden_states, position_embeddings):
        if self.memory_tracker is not None:
            self.memory_tracker.log(f"L{self.layer_idx} Before Decoder")
        # Attention
        residual = hidden_states
        if self.use_activation_checkpointing_for_norm:
            hidden_states = torch_activation_checkpoint(self.input_layernorm, hidden_states, use_reentrant=False)
        else:
            hidden_states = self.input_layernorm(hidden_states)
        if self.memory_tracker is not None:
            self.memory_tracker.log(f"L{self.layer_idx} Before Attention")
        if self.use_activation_checkpointing_for_attn:
            hidden_states = torch_activation_checkpoint(self.self_attn, hidden_states, position_embeddings, use_reentrant=False)
        else:
            hidden_states = self.self_attn(hidden_states, position_embeddings)
        if self.memory_tracker is not None:
            self.memory_tracker.log(f"L{self.layer_idx} After Attention")
        hidden_states = hidden_states + residual

        # MLP
        residual = hidden_states
        if self.use_activation_checkpointing_for_norm:
            hidden_states = torch_activation_checkpoint(self.post_attention_layernorm, hidden_states, use_reentrant=False)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
        if self.memory_tracker is not None:
            self.memory_tracker.log(f"L{self.layer_idx} Before MoE")
        hidden_states, router_logits = self.mlp(hidden_states)
        if self.memory_tracker is not None:
            self.memory_tracker.log(f"L{self.layer_idx} After MoE")
        hidden_states = hidden_states + residual

        if self.memory_tracker is not None:
            self.memory_tracker.log(f"L{self.layer_idx} After decoder")
        return hidden_states, router_logits

    def set_parameters_from_full_module(self, module):
        self.self_attn.set_parameters_from_full_module(module.self_attn)
        self.mlp.set_parameters_from_full_module(module.mlp)
        self.input_layernorm.load_state_dict(module.input_layernorm.state_dict())
        self.post_attention_layernorm.load_state_dict(module.post_attention_layernorm.state_dict())

class OlmoeParallelModel(nn.Module):
    def __init__(self, config, pmap=None, memory_tracker=None):
        super().__init__()
        # Parallelism related
        self.config = config
        self.pmap = pmap
        self.memory_tracker = memory_tracker

        self.num_hidden_layers = config.num_hidden_layers
        self.padding_idx = config.padding_idx
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        if self.pmap.is_first_stage_rank:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        else:
            self.embed_tokens = nn.Identity()

        self.num_hidden_layers_per_rank = config.num_hidden_layers // self.pmap.pipeline_parallelism
        self.num_hidden_layers_per_vpp_rank = self.num_hidden_layers_per_rank // config.virtual_pipeline_parallelism
        self.layer_ids = []
        for v in range(config.virtual_pipeline_parallelism):
            base_layer = (v * config.pipeline_parallelism + self.pmap.pp_ind) * self.num_hidden_layers_per_vpp_rank
            for l in range(self.num_hidden_layers_per_vpp_rank):
                self.layer_ids.append(base_layer + l)

        self.layers = nn.ModuleList(
            [OlmoeParallelDecoderLayer(config, layer_idx=self.layer_ids[i], pmap=pmap, memory_tracker=self.memory_tracker) for i in range(self.num_hidden_layers_per_rank)]
        )
        
        if self.pmap.is_last_stage_rank:
            self.norm = OlmoeParallelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = nn.Identity()

        self.rotary_emb = OlmoeParallelRotaryEmbedding(config)

        self.use_activation_checkpointing_for_norm = False
        if config.use_activation_checkpointing:
            self.use_activation_checkpointing_for_norm = config.activation_checkpointing_level & (1 << 2) # Bit 2 for Norm

    def forward(self, input_ids, vp_ind=0):
        hidden_states = self.embed_tokens(input_ids) if (vp_ind == 0) else input_ids

        position_ids = torch.arange(0, hidden_states.size(1), device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        router_logits_list = []
        for i in range(vp_ind*self.num_hidden_layers_per_vpp_rank, (vp_ind+1)*self.num_hidden_layers_per_vpp_rank):
            layer = self.layers[i]
            hidden_states, router_logits = layer(hidden_states, position_embeddings)
            router_logits_list.append(router_logits)
        router_logits = tuple(router_logits_list)

        VPP = self.config.virtual_pipeline_parallelism
        if (vp_ind == VPP-1):
            if self.use_activation_checkpointing_for_norm:
                hidden_states = torch_activation_checkpoint(self.norm, hidden_states, use_reentrant=False)
            else:
                hidden_states = self.norm(hidden_states)

        return hidden_states, router_logits


    def set_parameters_from_full_module(self, module):
        if self.pmap.is_first_stage_rank:
            self.embed_tokens.load_state_dict(module.embed_tokens.state_dict())
        
        for local_layer_idx in range(self.num_hidden_layers_per_rank):
            self.layers[local_layer_idx].set_parameters_from_full_module(module.layers[self.layer_ids[local_layer_idx]])

        if self.pmap.is_last_stage_rank:
            self.norm.load_state_dict(module.norm.state_dict())

class OlmoeParallelForCausalLM(nn.Module):
    def __init__(self, config, pmap=None, memory_tracker=None):
        super().__init__()
        # Parallelism related
        self.config = config
        self.pmap = pmap
        self.memory_tracker = memory_tracker

        self.vocab_size = config.vocab_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_aux_loss_coef = config.router_aux_loss_coef

        self.model = OlmoeParallelModel(config, pmap=self.pmap, memory_tracker=self.memory_tracker)
        if self.pmap.is_last_stage_rank:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = nn.Identity()

        self.output_router_logits = config.output_router_logits
        self.use_local_router_aux_loss = config.use_local_router_aux_loss

    def loss_function(self, logits, labels, vocab_size):
        # Choosing first (S-1) for logits and last (S-1) for labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Cross entropy loss
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
        return loss

    def forward(self, input_ids, vp_ind=0, labels=None):
        hidden_states, router_logits = self.model(input_ids, vp_ind=vp_ind)

        VPP = self.config.virtual_pipeline_parallelism
        if (self.pmap.pipeline_parallelism > 1) and \
            ((not self.pmap.is_last_stage_rank) or (self.pmap.is_last_stage_rank and (vp_ind != (VPP - 1)))):
            output_dict = {}
            output_dict["output"] = hidden_states
            output_dict["router_logits"] = router_logits
            return output_dict

        logits = self.lm_head(hidden_states)

        if (self.pmap.pipeline_parallelism > 1) and (self.pmap.is_last_stage_rank and (vp_ind == (VPP-1))):
            output_dict = {}
            output_dict["output"] = logits
            output_dict["router_logits"] = router_logits
            return output_dict

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size)

        # Auxiliary loss for load balancing in MoE
        aux_loss = None
        if self.output_router_logits:
            attention_mask = None
            if not self.use_local_router_aux_loss:
                aux_loss = load_balancing_loss_func(
                    router_logits,
                    self.num_experts,
                    self.num_experts_per_tok,
                    attention_mask,
                )
                if labels is not None:
                    loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device
            else:
                aux_loss = layer_level_load_balancing_loss_func(
                    router_logits,
                    self.num_experts, 
                    self.num_experts_per_tok,
                    attention_mask,
                )

                if labels is not None:
                    loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        # print(f"Rank : {self.pmap.rank} ,  Aux Loss: {aux_loss:6.2f}", flush=True)
        return OlmoeParallelOutput(loss=loss, logits=logits)
    
    def set_parameters_from_full_module(self, module):
        self.model.set_parameters_from_full_module(module.model)
        if self.pmap.is_last_stage_rank:
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
        mlp_gate_proj_flops = self.config.num_experts_per_tok * (6 * context_size * self.config.hidden_size * self.config.intermediate_size)
        mlp_up_proj_flops   = self.config.num_experts_per_tok * (6 * context_size * self.config.hidden_size * self.config.intermediate_size)
        mlp_down_proj_flops = self.config.num_experts_per_tok * (6 * context_size * self.config.intermediate_size * self.config.hidden_size)

        # LM Head
        lm_head_flops = (6 * context_size * self.config.hidden_size * self.config.vocab_size)

        # Groupings
        attn_flops = attn_q_proj_flops + attn_k_proj_flops + attn_v_proj_flops + attn_o_proj_flops + attn_qkt_flops + attn_ktv_flops
        mlp_flops = mlp_gate_proj_flops + mlp_up_proj_flops + mlp_down_proj_flops
        
        if self.config.use_activation_checkpointing:
            attn_flops = int((4/3)*attn_flops) if (self.config.activation_checkpointing_level & (1 << 0)) else attn_flops # Bit 0 for MoE
            mlp_flops = int((4/3)*mlp_flops) if (self.config.activation_checkpointing_level & (1 << 1)) else mlp_flops # Bit 1 for Attention

        decoder_block_flops = 0
        decoder_block_flops += attn_flops
        decoder_block_flops += mlp_flops
        decoder_stack_flops = (self.config.num_hidden_layers // self.pmap.pipeline_parallelism) * decoder_block_flops

        # Calculate total FLOPS
        total_flops = decoder_stack_flops
        if self.pmap.is_last_stage_rank:
            total_flops += lm_head_flops

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

    def get_ep_replicated_param_list(self):
        # Only EP
        params_replicated = []

        # Embedding
        params_replicated.append("model.embed_tokens.weight")

        # Attention
        for i in range(len(self.model.layers)):
            params_replicated.append(f"model.layers.{i}.self_attn.q_proj.weight")
            params_replicated.append(f"model.layers.{i}.self_attn.k_proj.weight")
            params_replicated.append(f"model.layers.{i}.self_attn.v_proj.weight")
            params_replicated.append(f"model.layers.{i}.self_attn.o_proj.weight")
            params_replicated.append(f"model.layers.{i}.self_attn.q_norm.weight")
            params_replicated.append(f"model.layers.{i}.self_attn.k_norm.weight")

        # MLP 
        for i in range(len(self.model.layers)):
            for e in range(self.model.layers[i].mlp.num_experts):
                params_replicated.append(f"model.layers.{i}.mlp.gate.weight")
        
        # Norms
        for i in range(len(self.model.layers)):
            params_replicated.append(f"model.layers.{i}.input_layernorm.weight")
            params_replicated.append(f"model.layers.{i}.post_attention_layernorm.weight")

        # Final norm and LMHead
        params_replicated.append("model.norm.weight")
        params_replicated.append("lm_head.weight")

        return params_replicated