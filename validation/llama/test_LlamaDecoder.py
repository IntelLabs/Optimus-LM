import os

# Framework imports
import torch
import intel_extension_for_pytorch

# Transformer imports
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

# Optimus imports
from optimus.models.llama.configuration_llama import LlamaParallelConfig
from optimus.models.llama.modeling_llama import LlamaParallelDecoderLayer, LlamaParallelRotaryEmbedding
from optimus.utils import print_error_info
from optimus.mapper import ParallelDpPpEpTpMapper

# Reproducibility
torch.manual_seed(42)

# Input configuration
batch_size = 2
seq_length = 4096
dtype = torch.float32
device = "xpu"
run_only_forward = False

# Configurations
config = LlamaParallelConfig()
config_hf = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")
config_hf._attn_implementation = "sdpa"

# Input and ouptut gradient tensors
hidden_states = torch.randn((batch_size, seq_length, config.hidden_size), dtype=dtype, device=device)
output_grad = torch.randn((batch_size, seq_length, config.hidden_size), dtype=dtype, device=device)

hidden_states_hf = torch.zeros((batch_size, seq_length, config_hf.hidden_size), dtype=dtype, device=device)
output_grad_hf = torch.zeros((batch_size, seq_length, config_hf.hidden_size), dtype=dtype, device=device)

with torch.no_grad():
    hidden_states_hf.copy_(hidden_states)
    output_grad_hf.copy_(output_grad)

# Enable requires_grad for backward pass
if not run_only_forward:
    hidden_states.requires_grad_(True)
    hidden_states_hf.requires_grad_(True)

# Use position_embeddings for both models
with torch.no_grad():
    rotary_emb = LlamaParallelRotaryEmbedding(config=config)
    position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
    position_embeddings = rotary_emb(hidden_states, position_ids)

    rotary_emb_hf = LlamaRotaryEmbedding(config=config_hf)
    position_ids_hf = torch.arange(seq_length, device=device).unsqueeze(0)
    position_embeddings_hf = rotary_emb_hf(hidden_states_hf, position_ids_hf)

# HF model
model_hf = LlamaDecoderLayer(config_hf, layer_idx=0)
# OM model
model = LlamaParallelDecoderLayer(config, pmap=ParallelDpPpEpTpMapper())
# Setting OM model from HF model
with torch.no_grad():
    model.load_state_dict(model_hf.state_dict())

# Set dtype and device
model = model.to(dtype).to(device)
model_hf = model_hf.to(dtype).to(device)

# Fwd/Bwd
output = model(hidden_states, position_embeddings=position_embeddings)
output_hf = model_hf(hidden_states_hf, position_embeddings=position_embeddings_hf)
if not run_only_forward:
    output.backward(output_grad)
    output_hf.backward(output_grad)

# Output comparision
print_error_info(output, output_hf, "Output")
if not run_only_forward:
    print_error_info(hidden_states.grad, hidden_states_hf.grad, "Input gradient")