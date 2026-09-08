import torch
import torch.nn as nn

from .common.custom_linear_modules import ColumnLinear, RowLinear
from .common.parallel_modules import ParallelEmbedding, ParallelLMHead
from ..mapper import ParallelMapper

from .common.comms_ptfunctions import forward_allreduce_backward_identity
from .common.norm_modules import PclLlamaRMSNorm

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

class ParallelLlmConfig():
    def __init__(
        self,
        model_choice = None,
        vocab_size = 128256,
        hidden_size = 768,
        intermediate_size = 3072,
        num_hidden_layers = 12,
        num_attention_heads = 12,
        num_key_value_heads = None,
        # Parallel
        pmap = None,
        device = None
    ):
        if model_choice == None:
            self.hf_config = None
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads if num_key_value_heads != None else num_attention_heads
        else:
            self.hf_config = AutoConfig.from_pretrained(model_choice)
            model_choices = ["meta-llama/Llama-3.2-1B"]
            assert model_choice in model_choices, f"Model name must be one of {model_choices}"
            if model_choice == "meta-llama/Llama-3.2-1B":
                self.vocab_size = self.hf_config.vocab_size
                self.hidden_size = self.hf_config.hidden_size
                self.intermediate_size = self.hf_config.intermediate_size
                self.num_hidden_layers = self.hf_config.num_hidden_layers
                self.num_attention_heads = self.hf_config.num_attention_heads
                self.num_key_value_heads = self.hf_config.num_key_value_heads

        self.pmap = pmap if pmap != None else ParallelMapper()
        self.device = device

class ParallelMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
                
        intermediate_size_per_rank = self.intermediate_size // config.pmap.tensor_parallelism
        self.gate_proj = nn.Linear(self.hidden_size, intermediate_size_per_rank, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, intermediate_size_per_rank, bias=False)
        self.down_proj = nn.Linear(intermediate_size_per_rank, self.hidden_size, bias=False)

        self.act_fn = nn.SiLU()

        self.pmap = config.pmap

    def forward(self, hidden_states):
        out_gp = self.gate_proj(hidden_states)
        out_up = self.up_proj(hidden_states)
        hidden_states = self.act_fn(out_gp) * out_up
        hidden_states = self.down_proj(hidden_states)

        if self.pmap.tensor_parallelism > 1:
            hidden_states = forward_allreduce_backward_identity.apply(hidden_states, self.pmap.tp_group)
        return hidden_states

class ParallelAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.num_attention_heads_per_rank = self.num_attention_heads // config.pmap.tensor_parallelism
        self.num_key_value_heads_per_rank = self.num_key_value_heads // config.pmap.tensor_parallelism

        assert self.hidden_size % self.num_attention_heads == 0, "Hidden size must be divisible by number of attention heads"
        self.head_size = self.hidden_size // self.num_attention_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads_per_rank * self.head_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads_per_rank * self.head_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads_per_rank * self.head_size, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads_per_rank * self.head_size, self.hidden_size, bias=False)

        self.pmap = config.pmap

    def forward(self, hidden_states, position_embeddings):
        # Notation for convenices
        B, S, _ = hidden_states.shape

        # Linear projections
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Change tensor view
        query = query.view(B, S, self.num_attention_heads_per_rank, self.head_size).transpose(1,2)
        key = key.view(B, S, self.num_key_value_heads_per_rank, self.head_size).transpose(1,2)
        value = value.view(B, S, self.num_key_value_heads_per_rank, self.head_size)

        # Apply rotary embeddings
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Undoing transpose
        query = query.transpose(1,2)
        key = key.transpose(1,2)

        # Expand key and value tensors for GAQ
        if self.num_attention_heads_per_rank != self.num_key_value_heads_per_rank:
            G = self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
            key = key.unsqueeze(-2).repeat(1, 1, 1, G, 1).reshape(B, S, self.num_attention_heads_per_rank, self.head_size)
            value = value.unsqueeze(-2).repeat(1, 1, 1, G, 1).reshape(B, S, self.num_attention_heads_per_rank, self.head_size)

        # Transpose
        query = query.transpose(1,2) # (B,S,N,H) -> (B,N,S,H)
        key = key.transpose(1,2)
        value = value.transpose(1,2)

        # Compute scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_size ** 0.5 # (B,N,S,S)

        # Mask
        mask = torch.tril(torch.ones(S, S)).view(1, 1, S, S).to(hidden_states.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute probabilites
        probs = nn.functional.softmax(scores, dim=-1)

        # Contextual embeddings
        attn_output = torch.matmul(probs, value) # (B,N,S,H)
        attn_output = attn_output.transpose(1,2).reshape(B, S, self.num_attention_heads_per_rank * self.head_size)

        # Output projection linear layer
        output = self.o_proj(attn_output)

        if self.pmap.tensor_parallelism > 1:
            output = forward_allreduce_backward_identity.apply(output, self.pmap.tp_group)

        return output

class ParallelLlmDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = ParallelAttention(config)
        self.mlp = ParallelMLP(config)
        # self.input_norm = nn.LayerNorm(config.hidden_size)
        # self.post_attention_norm = nn.LayerNorm(config.hidden_size)
        self.input_norm = PclLlamaRMSNorm(config.hidden_size)
        self.post_attention_norm = PclLlamaRMSNorm(config.hidden_size)
    
    def forward(self, hidden_states, position_embeddings):
        # Attention block
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings)
        hidden_states = hidden_states + residual

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states

class ParallelLlmModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pmap = config.pmap

        if self.pmap.is_first_stage_rank:
            if self.pmap.tensor_parallelism ==  1:
                self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            else:
                self.embed_tokens = ParallelEmbedding(config.vocab_size, config.hidden_size,
                                        tensor_parallelism=self.pmap.tensor_parallelism, tp_ind=self.pmap.tp_ind, group=self.pmap.tp_group)

        self.num_hidden_layers_per_rank = config.num_hidden_layers // self.pmap.pipeline_parallelism
        self.layers = torch.nn.ModuleList([ParallelLlmDecoderLayer(config) for _ in range(self.num_hidden_layers_per_rank)])
        self.layer_start_id = self.pmap.pp_ind * self.num_hidden_layers_per_rank
        self.layer_end_id = self.layer_start_id + self.num_hidden_layers_per_rank

        if self.pmap.is_last_stage_rank:
            # self.norm = nn.LayerNorm(config.hidden_size)
            self.norm = PclLlamaRMSNorm(config.hidden_size)
            if self.pmap.tensor_parallelism ==  1:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            else:
                self.lm_head = ParallelLMHead(config.hidden_size, config.vocab_size, bias=False,
                                    tensor_parallelism=self.pmap.tensor_parallelism, tp_ind=self.pmap.tp_ind, group=self.pmap.tp_group)
    
        # Rotary embedding
        self.rotary_emb = LlamaRotaryEmbedding(config=config.hf_config, device=config.device)

    def forward(self, input):
        if self.pmap.is_first_stage_rank:
            hidden_states = self.embed_tokens(input)
        else:
            hidden_states = input

        position_ids = torch.arange(0, hidden_states.size(1), device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings)

        if self.pmap.is_last_stage_rank:
            hidden_states = self.norm(hidden_states)
            output = self.lm_head(hidden_states)
        else:
            output = hidden_states
        
        return output
    
    def set_parameters_from_full_module(self, module):
        # embed_tokens
        if self.pmap.is_first_stage_rank:
            if self.pmap.tensor_parallelism ==  1:
                self.embed_tokens.weight.data.copy_(module.embed_tokens.weight)
            else:
                self.embed_tokens.set_parameters_from_full_module(module.embed_tokens)

        # layers
        from .common.parallel_converter_utils import populate_ColumnLinear_from_Linear, populate_RowLinear_from_Linear
        for i in range(self.num_hidden_layers_per_rank):
            gi = self.layer_start_id + i
            # LayerNorm
            self.layers[i].input_norm.weight.data.copy_(module.layers[gi].input_norm.weight)
            if self.layers[i].input_norm.bias != None:
                self.layers[i].input_norm.bias.data.copy_(module.layers[gi].input_norm.bias)

            # Attention
            populate_ColumnLinear_from_Linear(module.layers[gi].self_attn.q_proj, self.layers[i].self_attn.q_proj, self.pmap.tp_ind, tensor_parallelism=self.pmap.tensor_parallelism)
            populate_ColumnLinear_from_Linear(module.layers[gi].self_attn.k_proj, self.layers[i].self_attn.k_proj, self.pmap.tp_ind, tensor_parallelism=self.pmap.tensor_parallelism)
            populate_ColumnLinear_from_Linear(module.layers[gi].self_attn.v_proj, self.layers[i].self_attn.v_proj, self.pmap.tp_ind, tensor_parallelism=self.pmap.tensor_parallelism)
            populate_RowLinear_from_Linear(   module.layers[gi].self_attn.o_proj, self.layers[i].self_attn.o_proj, self.pmap.tp_ind, tensor_parallelism=self.pmap.tensor_parallelism)

            # LayerNorm
            self.layers[i].post_attention_norm.weight.data.copy_(module.layers[gi].post_attention_norm.weight)
            if self.layers[i].post_attention_norm.bias != None:
                self.layers[i].post_attention_norm.bias.data.copy_(module.layers[gi].post_attention_norm.bias)

            # MLP
            populate_ColumnLinear_from_Linear(module.layers[gi].mlp.gate_proj, self.layers[i].mlp.gate_proj, self.pmap.tp_ind, tensor_parallelism=self.pmap.tensor_parallelism)
            populate_ColumnLinear_from_Linear(module.layers[gi].mlp.up_proj,   self.layers[i].mlp.up_proj,   self.pmap.tp_ind, tensor_parallelism=self.pmap.tensor_parallelism)
            populate_RowLinear_from_Linear(   module.layers[gi].mlp.down_proj, self.layers[i].mlp.down_proj, self.pmap.tp_ind, tensor_parallelism=self.pmap.tensor_parallelism)

        if self.pmap.is_last_stage_rank:
            # Final layer norm
            self.norm.weight.data.copy_(module.norm.weight)
            if self.norm.bias != None:
                self.norm.bias.data.copy_(module.norm.bias)

            # lm_head
            if self.pmap.tensor_parallelism ==  1:
                self.lm_head.weight.data.copy_(module.lm_head.weight)
                if self.lm_head.bias != None:
                    self.lm_head.bias.data.copy_(module.lm_head.bias)
            else:
                self.lm_head.set_parameters_from_full_module(module.lm_head)