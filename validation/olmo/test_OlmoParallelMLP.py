import os

import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch

# Transformer imports
from transformers import AutoConfig
from transformers.models.olmo.modeling_olmo import OlmoMLP

# Optimus imports
from optimus.models.olmo.configuration_olmo import OlmoParallelConfig
from optimus.models.olmo.modeling_olmo import OlmoParallelMLP

from optimus.mapper import ParallelDpTpMapper
from optimus.utils import print_error_info

# Reproducibility
torch.manual_seed(42)

# Distributed setup
################################################################
rank, world_size, local_rank, local_world_size = 0,1,0,1
if int(os.getenv("PMI_SIZE", "1")) > 1:
    from optimus.dutils import setup_xpu_distributed
    rank, world_size, local_rank, local_world_size = setup_xpu_distributed(dist_backend="ccl")
################################################################

# Configuration
batch_size = 2
seq_length = 4096
dtype = torch.float32
device = "xpu"
run_only_forward = False

# Parallel configuration
tensor_parallelism = world_size
tensor_parallelism_degree = 15
config = OlmoParallelConfig(
    tensor_parallelism=tensor_parallelism,
    enable_tp_for_embedding=(tensor_parallelism_degree & 1) == 1,
    enable_tp_for_attn_block=(tensor_parallelism_degree & 2) == 2,
    enable_tp_for_mlp_block=(tensor_parallelism_degree & 4) == 4,
    enable_tp_for_lmhead=(tensor_parallelism_degree & 8) == 8,
    )
config_hf = AutoConfig.from_pretrained("allenai/OLMo-7B-hf")

print(config)
print(config_hf)

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
pmap = ParallelDpTpMapper(tensor_parallelism=tensor_parallelism, rank=rank, create_groups=True if tensor_parallelism > 1 else False)

model = OlmoParallelMLP(config, pmap=pmap)
model_hf = OlmoMLP(config_hf)
with torch.no_grad():
    model.set_parameters_from_full_module(model_hf)

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
if rank == 0:
    print_error_info(output, output_hf, "Output")
    if not run_only_forward:
        print_error_info(hidden_states.grad, hidden_states_hf.grad, "Input gradient")