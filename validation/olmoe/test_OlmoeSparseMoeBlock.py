import os

# Framework imports
import torch
import intel_extension_for_pytorch

# Transformer imports
from transformers import AutoConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

# Optimus imports
from optimus.models.olmoe.configuration_olmoe import OlmoeParallelConfig
from optimus.models.olmoe.modeling_olmoe import OlmoeParallelSparseMoeBlock
from optimus.mapper import ParallelDpPpEpTpMapper
from optimus.utils import print_error_info

# Reproducibility
torch.manual_seed(42)

# Input configuration
batch_size = 2
seq_length = 4096
dtype = torch.float32
device = "xpu"
run_only_forward = False

# Configurations
config = OlmoeParallelConfig()
config_hf = AutoConfig.from_pretrained("allenai/OLMoE-1B-7B-0924")

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
    
# HF model
model_hf = OlmoeSparseMoeBlock(config_hf)
# OM model
pmap = ParallelDpPpEpTpMapper()
model = OlmoeParallelSparseMoeBlock(config, pmap=pmap)
# Setting OM model from HF model
with torch.no_grad():
    model.load_state_dict(model_hf.state_dict())

# Set dtype and device
model = model.to(dtype).to(device)
model_hf = model_hf.to(dtype).to(device)

# Fwd/Bwd pass
output = model(hidden_states)[0]
output_hf = model_hf(hidden_states_hf)[0]
if not run_only_forward:
    output.backward(output_grad)
    output_hf.backward(output_grad_hf)

# Output comparision
print_error_info(output, output_hf, f"Output")
if not run_only_forward:
    print_error_info(hidden_states.grad, hidden_states_hf.grad, f"Input gradient")