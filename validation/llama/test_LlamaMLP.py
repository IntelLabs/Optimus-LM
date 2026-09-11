import os

import torch
import intel_extension_for_pytorch

# Transformer imports
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaMLP

# Optimus imports
from optimus.models.llama.configuration_llama import LlamaParallelConfig
from optimus.models.llama.modeling_llama import LlamaParallelMLP

from optimus.mapper import ParallelDpPpEpTpMapper
from optimus.utils import print_error_info

# Reproducibility
torch.manual_seed(42)


# Configuration
batch_size = 2
seq_length = 4096
dtype = torch.float32
device = "xpu"
run_only_forward = False

# Serial configuration
config = LlamaParallelConfig()
config_hf = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")

# Input and ouptut gradient tensors
hidden_states = torch.randn((batch_size, seq_length, config.hidden_size), dtype=dtype, device=device)
output_grad = torch.randn((batch_size, seq_length, config.hidden_size), dtype=dtype, device=device)

hidden_states_hf = torch.zeros((batch_size, seq_length, config_hf.hidden_size), dtype=dtype, device=device)
with torch.no_grad():
    hidden_states_hf.copy_(hidden_states)

# Enable requires_grad for backward pass
if not run_only_forward:
    hidden_states.requires_grad_(True)
    hidden_states_hf.requires_grad_(True)

# Modules
pmap = ParallelDpPpEpTpMapper()

model_hf = LlamaParallelMLP(config_hf)
model = LlamaParallelMLP(config, pmap=pmap)
with torch.no_grad():
    model.load_state_dict(model_hf.state_dict())

# Set dtype and device
model = model.to(dtype).to(device)
model_hf = model_hf.to(dtype).to(device)

# Forward pass
output = model(hidden_states)
output_hf = model_hf(hidden_states_hf)

# Backward pass
if not run_only_forward:
    output.backward(output_grad)
    output_hf.backward(output_grad)

# Output comparision
from optimus.utils import print_error_info
print_error_info(output, output_hf, "Output")
if not run_only_forward:
    print_error_info(hidden_states.grad, hidden_states_hf.grad, "Input gradient")