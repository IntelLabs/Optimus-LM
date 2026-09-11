import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

# Parallelization related functions, modules, and utils
from ..common.comms_ptfunctions import forward_identity_backward_allreduce
from ..common.comms_ptfunctions import forward_allreduce_backward_identity

from ..common.parallel_modules import ParallelEmbedding, ParallelLMHead
from ..common.parallel_converter_utils import populate_ColumnLinear_from_Linear, populate_RowLinear_from_Linear

class OlmoParallelOutput:
    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits

class OlmoLayerNorm(nn.Module):
    """LayerNorm but with no learnable weight or bias."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.normalized_shape = (hidden_size,)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        return F.layer_norm(hidden_states.to(dtype=torch.float32), self.normalized_shape, None, None, eps=1e-5).to(
            orig_dtype
        )

class OlmoParallelRMSNorm(nn.Module):
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

class OlmoParallelMLP(nn.Module):
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

class OlmoParallelAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, pmap = None):
        super().__init__()
        self.config = config

        self.pmap = pmap
        self.num_attention_heads_per_rank = config.num_attention_heads // self.pmap.tensor_parallelism
        self.num_key_value_heads_per_rank = config.num_key_value_heads // self.pmap.tensor_parallelism

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_attention_heads_per_rank * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads_per_rank * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads_per_rank * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads_per_rank * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

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
    

class OlmoParallelDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, pmap=None):
        super().__init__()
        self.pmap = pmap

        self.hidden_size = config.hidden_size
        self.self_attn = OlmoParallelAttention(config, pmap=pmap)
        # from ..common.attention_modules import ParallelAttention
        # self.self_attn = ParallelAttention(config, pmap=pmap)

        self.mlp = OlmoParallelMLP(config, pmap=pmap)
        
        if not config.use_rms_norm:
            self.input_layernorm = OlmoLayerNorm(config.hidden_size)
            self.post_attention_layernorm = OlmoLayerNorm(config.hidden_size)
        else:
            self.input_layernorm = OlmoParallelRMSNorm(config.hidden_size)
            self.post_attention_layernorm = OlmoParallelRMSNorm(config.hidden_size)

    def forward(self, hidden_states, position_embeddings):
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def set_parameters_from_full_module(self, full_module):
        self.self_attn.set_parameters_from_full_module(full_module.self_attn)
        self.mlp.set_parameters_from_full_module(full_module.mlp)
        if full_module.input_layernorm.__class__ == OlmoParallelRMSNorm:
            self.input_layernorm.load_state_dict(full_module.input_layernorm.state_dict())
        if full_module.post_attention_layernorm.__class__ == OlmoParallelRMSNorm:
            self.post_attention_layernorm.load_state_dict(full_module.post_attention_layernorm.state_dict())

class OlmoRotaryEmbedding(nn.Module):
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

class OlmoParallelModel(nn.Module):
    def __init__(self, config, pmap=None):
        super().__init__()
        self.config = config
        self.pmap = pmap
        self.num_hidden_layers = config.num_hidden_layers

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
        self.num_hidden_layers_per_vpp_rank = self.num_hidden_layers_per_rank // config.virtual_pipeline_parallelism
        self.layer_ids = []
        for v in range(config.virtual_pipeline_parallelism):
            base_layer = (v * config.pipeline_parallelism + self.pmap.pp_ind) * self.num_hidden_layers_per_vpp_rank
            for l in range(self.num_hidden_layers_per_vpp_rank):
                self.layer_ids.append(base_layer + l)

        self.layers = nn.ModuleList(
            [OlmoParallelDecoderLayer(config, self.layer_ids[i], pmap=self.pmap) for i in range(self.num_hidden_layers_per_rank)]
        )

        if self.pmap.is_last_stage_rank:
            if not config.use_rms_norm:
                self.norm = OlmoLayerNorm(config.hidden_size)
            else:
                self.norm = OlmoParallelRMSNorm(config.hidden_size)
        else:
            self.norm = nn.Identity()
        
        self.rotary_emb = OlmoRotaryEmbedding(config=config)

    def forward(self, input_ids, vp_ind=0):
        hidden_states = self.embed_tokens(input_ids) if (vp_ind == 0) else input_ids

        position_ids = torch.arange(0, hidden_states.size(1), device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i in range(vp_ind*self.num_hidden_layers_per_vpp_rank, (vp_ind+1)*self.num_hidden_layers_per_vpp_rank):
            layer = self.layers[i]
            hidden_states = layer(hidden_states, position_embeddings)

        VPP = self.config.virtual_pipeline_parallelism
        hidden_states = self.norm(hidden_states) if (vp_ind == (VPP-1)) else hidden_states

        return hidden_states

    def set_parameters_from_full_module(self, module):
        if self.pmap.is_first_stage_rank:
            if self.pmap.tensor_parallelism > 1:
                self.embed_tokens.set_parameters_from_full_module(module.embed_tokens)
            else:
                self.embed_tokens.load_state_dict(module.embed_tokens.state_dict())

        for i in range(self.num_hidden_layers_per_rank):
            self.layers[i].set_parameters_from_full_module(module.layers[self.layer_ids[i]])

        if self.pmap.is_last_stage_rank:
            if module.norm.__class__ == OlmoParallelRMSNorm:
                self.norm.load_state_dict(module.norm.state_dict())

class OlmoParallelForCausalLM(nn.Module):
    def __init__(self, config, pmap=None):
        super().__init__()
        self.config = config
        self.pmap = pmap

        self.model = OlmoParallelModel(config, pmap=self.pmap)
        self.vocab_size = config.vocab_size
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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

    def forward(self, input_ids, vp_ind=0, labels=None):
        hidden_states = self.model(input_ids, vp_ind=vp_ind)

        VPP = self.config.virtual_pipeline_parallelism
        if (self.pmap.pipeline_parallelism > 1) and \
            ((not self.pmap.is_last_stage_rank) or (self.pmap.is_last_stage_rank and (vp_ind != (VPP - 1)))):
            return hidden_states

        logits = self.lm_head(hidden_states)

        if (self.pmap.pipeline_parallelism > 1) and (self.pmap.is_last_stage_rank and (vp_ind == (VPP-1))):
            return logits

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size)

        # print(f"Rank : {self.pmap.rank} ,  Aux Loss: {aux_loss:6.2f}", flush=True)
        return OlmoParallelOutput(loss=loss, logits=logits)

    def set_parameters_from_full_module(self, module):
        self.model.set_parameters_from_full_module(module.model)
        if self.pmap.is_last_stage_rank:
            if (self.pmap.tensor_parallelism > 1):
                self.lm_head.set_parameters_from_full_module(module.lm_head)
            else:
                self.lm_head.load_state_dict(module.lm_head.state_dict())

    def get_flops(self, context_size):
        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        # Attention (6 = 2 for gemm op, 3 gemm ops)
        attn_q_proj_flops = 6 * context_size * self.config.hidden_size * (self.config.num_attention_heads * head_dim)
        attn_k_proj_flops = 6 * context_size * self.config.hidden_size * (self.config.num_key_value_heads * head_dim)
        attn_v_proj_flops = 6 * context_size * self.config.hidden_size * (self.config.num_key_value_heads * head_dim)
        attn_o_proj_flops = 6 * context_size * (self.config.num_attention_heads * head_dim) * self.config.hidden_size

        attn_qkt_flops = 6 * self.config.num_attention_heads * (context_size * context_size * head_dim)
        attn_ktv_flops = 6 * self.config.num_attention_heads * (context_size * context_size * head_dim)

        # MLP
        mlp_gate_proj_flops = 6 * context_size * self.config.hidden_size * self.config.intermediate_size
        mlp_up_proj_flops = 6 * context_size * self.config.hidden_size * self.config.intermediate_size
        mlp_down_proj_flops = 6 * context_size * self.config.intermediate_size * self.config.hidden_size

        # LM Head
        lm_head_flops = 6 * context_size * self.config.hidden_size * self.config.vocab_size

        # Adjusting for tensor parallelism
        attn_flops = (attn_q_proj_flops + attn_k_proj_flops + attn_v_proj_flops + attn_o_proj_flops + attn_qkt_flops + attn_ktv_flops) // self.pmap.tensor_parallelism
        mlp_flops = (mlp_gate_proj_flops + mlp_up_proj_flops + mlp_down_proj_flops) // self.pmap.tensor_parallelism
        lm_head_flops = lm_head_flops // self.pmap.tensor_parallelism
        decoder_block_flops = (attn_flops + mlp_flops)
        decoder_stack_flops = (self.config.num_hidden_layers * decoder_block_flops) // self.pmap.pipeline_parallelism

        # Calculate total FLOPS
        total_flops = decoder_stack_flops + lm_head_flops

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
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

    def calculate_grad_norm(self):
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
        grads_sharded = []
        grads_replicated = []
        for name,p in self.named_parameters():
            if p.grad is not None:
                if name in params_sharded:
                    grads_sharded.append(p.grad)
                else:
                    grads_replicated.append(p.grad)
                    params_replicated.append(name)
        
        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     for _ in params_replicated:
        #         print(_,flush=True)

        # Calculate norm
        grad_norm_sharded = torch.nn.utils.get_total_norm(grads_sharded).to(torch.float32)
        grad_norm_replicated = torch.nn.utils.get_total_norm(grads_replicated).to(torch.float32)

        # Undo sqrt
        grad_norm_sharded_sq = torch.pow(grad_norm_sharded, 2)
        grad_norm_replicated_sq = torch.pow(grad_norm_replicated, 2)

        # Default grad_norm
        if torch.distributed.is_initialized() and (self.config.tensor_parallelism > 1):
            grad_norm_sq = grad_norm_sharded_sq + grad_norm_replicated_sq / self.config.tensor_parallelism
            torch.xpu.synchronize()
            torch.distributed.barrier(group=self.pmap.tp_group)
            torch.distributed.all_reduce(grad_norm_sq, group=self.pmap.tp_group)
            grad_norm = grad_norm_sq.sqrt()
        else:
            grad_norm = torch.sqrt(grad_norm_sharded_sq + grad_norm_replicated_sq)

        return grad_norm
        