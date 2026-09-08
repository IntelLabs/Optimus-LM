import math

import torch
from torch import nn

import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_activation_checkpoint

# https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py
# https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
# https://www.youtube.com/watch?v=8v2l6SJECW4

from .rope_utils import _compute_yarn_parameters, rotate_half, apply_rotary_pos_emb_interleave, yarn_get_mscale

from ..common.comms_ptfunctions import forward_allgather_backward_split
from ..common.comms_ptfunctions import forward_identity_backward_allreduce
from ..common.comms_ptfunctions import forward_reducescatter_backward_allgather
from ..common.comms_ptfunctions import forward_identity_backward_synchronize
from ..common.comms_ptfunctions import forward_synchronize_backward_identity
from ..common.comms_ptfunctions import forward_allgather_backward_reducescatter

from ..common.parallel_converter_utils import populate_ColumnLinear_from_Linear, populate_RowLinear_from_Linear

class DeepseekV3ParallelOutput:
    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits

class DeepseekV3ParallelRMSNorm(nn.Module):
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

    def set_parameters_from_full_module(self, module):
        self.load_state_dict(module.state_dict())


class DeepseekV3ParallelRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        assert self.rope_type == "yarn", "DeepseekV3RotaryEmbedding only supports 'yarn' rope type for now to avoid complexity of handling multiple cases in the code"

        self.config = config
        self.rope_init_fn = _compute_yarn_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
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

class DeepseekV3ParallelMLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None, pmap=None):
        super().__init__()
        self.config = config
        self.pmap = pmap

        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        if self.pmap is None:
            self.intermediate_size_per_rank = self.intermediate_size
        else:
            self.intermediate_size_per_rank = config.intermediate_size // self.pmap.expert_parallelism
            
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size_per_rank, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size_per_rank, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size_per_rank, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        self.use_allreduce_for_reducescatter = False

        self.use_activation_checkpointing_for_moe = False
        if config.use_activation_checkpointing:
            self.use_activation_checkpointing_for_moe = config.activation_checkpointing_level & (1 << 0) # Bit 0 for MLP

    def forward(self, hidden_states):
        def expert_computation_block(hidden_states):
            if (self.pmap is not None) and (self.pmap.expert_parallelism > 1):
                hidden_states = forward_allgather_backward_reducescatter.apply(hidden_states, self.pmap.ep_ind, self.pmap.expert_parallelism, self.pmap.ep_group, self.use_allreduce_for_reducescatter)

            hidden_states = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

            if (self.pmap is not None) and (self.pmap.expert_parallelism > 1):
                hidden_states = forward_reducescatter_backward_allgather.apply(hidden_states, self.pmap.ep_ind, self.pmap.expert_parallelism, self.pmap.ep_group, self.use_allreduce_for_reducescatter, False)

            return hidden_states

        if self.use_activation_checkpointing_for_moe:
            hidden_states = torch_activation_checkpoint(expert_computation_block, hidden_states, use_reentrant=False)
        else:
            hidden_states = expert_computation_block(hidden_states)

        return hidden_states

    def set_parameters_from_full_module(self, module):
        if (self.pmap is None):
            self.load_state_dict(module.state_dict())
        else:
            populate_ColumnLinear_from_Linear(module.gate_proj, self.gate_proj, self.pmap.ep_ind, tensor_parallelism=self.pmap.expert_parallelism)
            populate_ColumnLinear_from_Linear(module.up_proj,   self.up_proj,   self.pmap.ep_ind, tensor_parallelism=self.pmap.expert_parallelism)
            populate_RowLinear_from_Linear(   module.down_proj, self.down_proj, self.pmap.ep_ind, tensor_parallelism=self.pmap.expert_parallelism)

class DeepseekV3ParallelTopkRouter(nn.Module):
    def __init__(self, config, pmap=None):
        super().__init__()
        self.config = config
        self.pmap = pmap

        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))

        self.mask = None
        self.force_uniform_routing = config.force_uniform_routing

    @torch.no_grad()
    def get_topk_indices(self, scores):
        #####################################################################
        if self.force_uniform_routing:
            if self.mask is None:
                from optimus.utils import get_load_balanced_expert_mask
                num_tokens = scores.numel() // scores.shape[-1]
                self.mask = get_load_balanced_expert_mask(num_tokens, self.n_routed_experts, self.top_k, self.pmap.ep_ind)
                self.mask = (self.mask == False).to(scores.device)  # Invert the mask to use with `masked_fill`
            scores = scores.masked_fill(self.mask, float('-inf'))
            topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)[1]
            return topk_indices
        ######################################################################

        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

class DeepseekV3ParallelMoE(nn.Module):
    def __init__(self, config, pmap=None):
        super().__init__()
        self.config = config
        self.pmap = pmap

        self.n_routed_experts_per_rank = config.n_routed_experts // config.expert_parallelism
        self.expert_start_idx = self.pmap.ep_ind * self.n_routed_experts_per_rank
        self.expert_end_idx = (self.pmap.ep_ind + 1) * self.n_routed_experts_per_rank

        self.experts = nn.ModuleList(
            [
                DeepseekV3ParallelMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(self.n_routed_experts_per_rank)
            ]
        )
        self.gate = DeepseekV3ParallelTopkRouter(config, pmap=self.pmap)
        self.shared_experts = DeepseekV3ParallelMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )

        self.use_allreduce_for_reducescatter = False

        self.use_activation_checkpointing_for_moe = False
        if config.use_activation_checkpointing:
            self.use_activation_checkpointing_for_moe = config.activation_checkpointing_level & (1 << 0) # Bit 0 for MLP

    def moe(self, hidden_states, topk_indices, topk_weights):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        # expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=self.config.n_routed_experts)
        expert_mask = expert_mask.permute(2, 0, 1)

        for local_expert_idx in range(self.n_routed_experts_per_rank):
            expert_idx = self.expert_start_idx + local_expert_idx

            expert = self.experts[local_expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        # Gate outside to deal with activation checkpointing issue
        topk_indices, topk_weights = self.gate(hidden_states)

        def expert_computation_block(hidden_states, topk_indices, topk_weights):
            residuals = hidden_states
            orig_shape = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            if self.pmap.expert_parallelism > 1:
                hidden_states = forward_allgather_backward_reducescatter.apply(hidden_states, self.pmap.ep_ind, self.pmap.expert_parallelism, self.pmap.ep_group, self.use_allreduce_for_reducescatter)
                topk_weights = forward_allgather_backward_reducescatter.apply(topk_weights, self.pmap.ep_ind, self.pmap.expert_parallelism, self.pmap.ep_group, self.use_allreduce_for_reducescatter)
                topk_indices = forward_allgather_backward_split.apply(topk_indices, self.pmap.ep_ind, self.pmap.expert_parallelism, self.pmap.ep_group)

            final_hidden_states = self.moe(hidden_states, topk_indices, topk_weights)

            if self.pmap.expert_parallelism > 1:
                final_hidden_states = forward_reducescatter_backward_allgather.apply(final_hidden_states, self.pmap.ep_ind, self.pmap.expert_parallelism, self.pmap.ep_group, self.use_allreduce_for_reducescatter, False)

            final_hidden_states = final_hidden_states.view(*orig_shape)
            final_hidden_states = final_hidden_states + self.shared_experts(residuals)

            return final_hidden_states
        
        args = [hidden_states, topk_indices, topk_weights]
        if self.use_activation_checkpointing_for_moe:
            final_hidden_states = torch_activation_checkpoint(expert_computation_block, *args, use_reentrant=False)
        else:
            final_hidden_states = expert_computation_block(*args)

        return final_hidden_states

    def set_parameters_from_full_module(self, module):
        # self.load_state_dict(module.state_dict())
        self.gate.load_state_dict(module.gate.state_dict())
        for local_expert_idx in range(self.n_routed_experts_per_rank):
            self.experts[local_expert_idx].load_state_dict(module.experts[self.expert_start_idx + local_expert_idx].state_dict())
        self.shared_experts.load_state_dict(module.shared_experts.state_dict())

class DeepseekV3ParallelAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Limiting support to actual DSV3 configs to avoid complexity of handling multiple cases in the code
        assert config.num_attention_heads == config.num_key_value_heads, "DeepseekV3ParallelAttention only supports num_attention_heads == num_key_value_heads"
        assert config.attention_bias == False, "DeepseekV3ParallelAttention only supports attention_bias == False"
        assert config.attention_dropout == 0.0, "DeepseekV3ParallelAttention only supports attention_dropout == 0.0"
        assert config.q_lora_rank is not None, "DeepseekV3ParallelAttention only supports q_lora_rank is not None"
        assert config.rope_interleave == True, "DeepseekV3ParallelAttention only supports rope_interleave == True"

        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim

        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim

        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim

        self.rope_theta = config.rope_theta

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = DeepseekV3ParallelRMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = DeepseekV3ParallelRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim,config.hidden_size, bias=False)

        self.scaling = self.qk_head_dim ** (-0.5)
        mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
        scaling_factor = self.config.rope_scaling["factor"]
        if mscale_all_dim:
            mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
            self.scaling = self.scaling * mscale * mscale
        
        self.causal_mask = torch.tril(torch.ones((config.max_position_embeddings, config.max_position_embeddings), dtype=torch.bool))
        self.causal_mask = self.causal_mask.view(1, 1, config.max_position_embeddings, config.max_position_embeddings)

        self.skip_causal_mask = config.skip_causal_mask if hasattr(config, "skip_causal_mask") else False

    def forward(self, hidden_states, position_embeddings):
         # Move mask to device
        if hidden_states.device != self.causal_mask.device:
            self.causal_mask = self.causal_mask.to(hidden_states.device)

        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, self.num_heads, self.qk_head_dim)
        key_shape = (batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)

        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        # Multi Head Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        
        # Apply causal mask
        if not self.skip_causal_mask:
            causal_mask = self.causal_mask[:,:,:seq_length,:seq_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None
    
    def set_parameters_from_full_module(self, module):
        self.load_state_dict(module.state_dict())

class DeepseekV3ParallelDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=0, pmap=None, memory_tracker=None):
        super().__init__()
        self.self_attn = DeepseekV3ParallelAttention(config)
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3ParallelMoE(config, pmap=pmap)
        else:
            self.mlp = DeepseekV3ParallelMLP(config, pmap=pmap)
        self.input_layernorm = DeepseekV3ParallelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3ParallelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layer_idx = layer_idx
        self.memory_tracker = memory_tracker

        self.use_activation_checkpointing_for_moe = False
        self.use_activation_checkpointing_for_attn = False
        self.use_activation_checkpointing_for_norm = False
        
        if config.use_activation_checkpointing:
            self.use_activation_checkpointing_for_moe = config.activation_checkpointing_level & (1 << 0) # Bit 0 for MoE
            self.use_activation_checkpointing_for_attn = config.activation_checkpointing_level & (1 << 1) # Bit 1 for Attention
            self.use_activation_checkpointing_for_norm = config.activation_checkpointing_level & (1 << 2) # Bit 2 for Norm

    def forward(self, hidden_states, position_embeddings):
        if self.memory_tracker is not None:
            self.memory_tracker.log(f"L{self.layer_idx} Before Decoder")
        residual = hidden_states
        if self.use_activation_checkpointing_for_norm:
            hidden_states = torch_activation_checkpoint(self.input_layernorm, hidden_states, use_reentrant=False)
        else:
            hidden_states = self.input_layernorm(hidden_states)
        if self.memory_tracker is not None:
            self.memory_tracker.log(f"L{self.layer_idx} Before Attention")
        if self.use_activation_checkpointing_for_attn:
            hidden_states, _ = torch_activation_checkpoint(self.self_attn, hidden_states, position_embeddings, use_reentrant=False)
        else:
            hidden_states, _ = self.self_attn(hidden_states, position_embeddings)
        if self.memory_tracker is not None:
            self.memory_tracker.log(f"L{self.layer_idx} After Attention")
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        if self.use_activation_checkpointing_for_norm:
            hidden_states = torch_activation_checkpoint(self.post_attention_layernorm, hidden_states, use_reentrant=False)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
        if self.memory_tracker is not None:
            self.memory_tracker.log(f"L{self.layer_idx} Before MoE")
        hidden_states = self.mlp(hidden_states)
        if self.memory_tracker is not None:
            self.memory_tracker.log(f"L{self.layer_idx} After MoE")
        hidden_states = residual + hidden_states

        if self.memory_tracker is not None:
            self.memory_tracker.log(f"L{self.layer_idx} After Decoder")

        return hidden_states
    
    def set_parameters_from_full_module(self, module):
        self.self_attn.set_parameters_from_full_module(module.self_attn)
        self.mlp.set_parameters_from_full_module(module.mlp)
        self.input_layernorm.load_state_dict(module.input_layernorm.state_dict())
        self.post_attention_layernorm.load_state_dict(module.post_attention_layernorm.state_dict())

class DeepseekV3ParallelModel(nn.Module):
    def __init__(self, config, pmap=None, memory_tracker=None):
        super().__init__()
        self.pmap = pmap
        self.memory_tracker = memory_tracker

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if self.pmap.is_first_stage_rank:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        else:
            self.embed_tokens = nn.Identity()
        
        self.num_hidden_layers_per_rank = config.num_hidden_layers // self.pmap.pipeline_parallelism
        self.layer_start_idx = self.pmap.pp_ind * self.num_hidden_layers_per_rank
        self.layer_end_idx = (self.pmap.pp_ind + 1) * self.num_hidden_layers_per_rank
        self.layers = nn.ModuleList(
            [DeepseekV3ParallelDecoderLayer(config, layer_idx=layer_idx, pmap=self.pmap, memory_tracker=self.memory_tracker) for layer_idx in range(self.layer_start_idx, self.layer_end_idx)]
        )

        if self.pmap.is_last_stage_rank:
            self.norm = DeepseekV3ParallelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = nn.Identity()

        self.rotary_emb = DeepseekV3ParallelRotaryEmbedding(config=config)

        self.use_activation_checkpointing_for_norm = False
        if config.use_activation_checkpointing:
            self.use_activation_checkpointing_for_norm = config.activation_checkpointing_level & (1 << 2) # Bit 2 for Norm
        
    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)

        position_ids = torch.arange(0, hidden_states.size(1), device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)

        if self.use_activation_checkpointing_for_norm:
            hidden_states = torch_activation_checkpoint(self.norm, hidden_states, use_reentrant=False)
        else:
            hidden_states = self.norm(hidden_states)
        
        return hidden_states
    
    def set_parameters_from_full_module(self, module):
        if self.pmap.is_first_stage_rank:
            self.embed_tokens.load_state_dict(module.embed_tokens.state_dict())
        
        for local_layer_idx in range(self.num_hidden_layers_per_rank):
            layer_idx = self.layer_start_idx + local_layer_idx
            self.layers[local_layer_idx].set_parameters_from_full_module(module.layers[layer_idx])

        if self.pmap.is_last_stage_rank:
            self.norm.load_state_dict(module.norm.state_dict())

class DeepseekV3ParallelForCausalLM(nn.Module):
    def __init__(self, config, pmap=None, memory_tracker=None):
        super().__init__()
        # Parallelism related
        self.config = config
        self.pmap = pmap
        self.memory_tracker = memory_tracker

        self.vocab_size = config.vocab_size

        self.model = DeepseekV3ParallelModel(config, pmap=self.pmap, memory_tracker=self.memory_tracker)
        if self.pmap.is_last_stage_rank:
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
        
        return DeepseekV3ParallelOutput(loss=loss, logits=logits)
    
    def set_parameters_from_full_module(self, module):
        self.model.set_parameters_from_full_module(module.model)
        if self.pmap.is_last_stage_rank:
            self.lm_head.load_state_dict(module.lm_head.state_dict())

    def get_flops(self, context_size):
        # Attention
        attn_q_a_proj_flops = 6 * context_size * self.config.hidden_size * self.config.q_lora_rank
        attn_q_b_proj_flops = 6 * context_size * self.config.q_lora_rank * (self.config.num_attention_heads * self.config.qk_head_dim)
        attn_kv_a_proj_flops = 6 * context_size * self.config.hidden_size * (self.config.kv_lora_rank + self.config.qk_rope_head_dim)
        attn_kv_b_proj_flops = 6 * context_size * self.config.kv_lora_rank * (self.config.num_attention_heads * (self.config.qk_nope_head_dim + self.config.v_head_dim))
        attn_o_proj_flops = 6 * context_size * (self.config.num_attention_heads * self.config.v_head_dim) * self.config.hidden_size

        attn_qkt_flops = 6 * self.config.num_attention_heads * (context_size * context_size * self.config.qk_head_dim)
        attn_ktv_flops = 6 * self.config.num_attention_heads * (context_size * context_size * (self.config.qk_nope_head_dim + self.config.v_head_dim))

        # Dense MLP
        dense_mlp_gate_proj_flops = (6 * context_size * self.config.hidden_size * self.config.intermediate_size)
        dense_mlp_up_proj_flops   = (6 * context_size * self.config.hidden_size * self.config.intermediate_size)
        dense_mlp_down_proj_flops = (6 * context_size * self.config.intermediate_size * self.config.hidden_size)

        # Expert MLP
        moe_mlp_gate_proj_flops = (self.config.num_experts_per_tok + self.config.n_shared_experts) * (6 * context_size * self.config.hidden_size * self.config.moe_intermediate_size)
        moe_mlp_up_proj_flops   = (self.config.num_experts_per_tok + self.config.n_shared_experts) * (6 * context_size * self.config.hidden_size * self.config.moe_intermediate_size)
        moe_mlp_down_proj_flops = (self.config.num_experts_per_tok + self.config.n_shared_experts) * (6 * context_size * self.config.moe_intermediate_size * self.config.hidden_size)

        # LM Head
        lm_head_flops = (6 * context_size * self.config.hidden_size * self.config.vocab_size)

        # Groupings
        attn_flops = attn_q_a_proj_flops + attn_q_b_proj_flops + attn_kv_a_proj_flops + attn_kv_b_proj_flops + attn_o_proj_flops + attn_qkt_flops + attn_ktv_flops
        dense_mlp_flops = dense_mlp_gate_proj_flops + dense_mlp_up_proj_flops + dense_mlp_down_proj_flops
        moe_mlp_flops = moe_mlp_gate_proj_flops + moe_mlp_up_proj_flops + moe_mlp_down_proj_flops
        
        if self.config.use_activation_checkpointing:
            attn_flops = int((4/3)*attn_flops) if (self.config.activation_checkpointing_level & (1 << 0)) else attn_flops # Bit 0 for MoE
            dense_mlp_flops = int((4/3)*dense_mlp_flops) if (self.config.activation_checkpointing_level & (1 << 1)) else dense_mlp_flops # Bit 1 for Attention
            moe_mlp_flops = int((4/3)*moe_mlp_flops) if (self.config.activation_checkpointing_level & (1 << 2)) else moe_mlp_flops # Bit 2 for MoE

        # Decoder flops
        dense_decoder_block_flops = (attn_flops + dense_mlp_flops)
        moe_decoder_block_flops = (attn_flops + moe_mlp_flops)
        decoder_stack_flops = (self.config.first_k_dense_replace * dense_decoder_block_flops + (self.config.num_hidden_layers - self.config.first_k_dense_replace) * moe_decoder_block_flops)
        
        # Calculate total FLOPS
        total_flops = decoder_stack_flops + lm_head_flops

        # Adjusting for pipeline parallelism
        total_flops = total_flops / self.pmap.pipeline_parallelism

        return total_flops

    def get_ep_replicated_param_list(self):
        # Only EP
        params_replicated = []

        # Embedding
        if self.pmap.is_first_stage_rank:
            params_replicated.append("model.embed_tokens.weight")

        # Attention
        for i in range(len(self.model.layers)):
            params_replicated.append(f"model.layers.{i}.self_attn.q_a_proj.weight")
            params_replicated.append(f"model.layers.{i}.self_attn.q_a_layernorm.weight")
            params_replicated.append(f"model.layers.{i}.self_attn.q_b_proj.weight")
            params_replicated.append(f"model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight")
            params_replicated.append(f"model.layers.{i}.self_attn.kv_a_layernorm.weight")
            params_replicated.append(f"model.layers.{i}.self_attn.kv_b_proj.weight")
            params_replicated.append(f"model.layers.{i}.self_attn.o_proj.weight")

        # MoE (Gate and shared experts)
        for i in range(len(self.model.layers)):
            if (self.model.layer_start_idx + i) >= self.config.first_k_dense_replace:
                params_replicated.append(f"model.layers.{i}.mlp.gate.weight")
                params_replicated.append(f"model.layers.{i}.mlp.shared_experts.gate_proj.weight")
                params_replicated.append(f"model.layers.{i}.mlp.shared_experts.up_proj.weight")
                params_replicated.append(f"model.layers.{i}.mlp.shared_experts.down_proj.weight")

        # Norms
        for i in range(len(self.model.layers)):
            params_replicated.append(f"model.layers.{i}.input_layernorm.weight")
            params_replicated.append(f"model.layers.{i}.post_attention_layernorm.weight")

        # Final norm and LMHead
        if self.pmap.is_last_stage_rank:
            params_replicated.append("model.norm.weight")
            params_replicated.append("lm_head.weight")

        return params_replicated