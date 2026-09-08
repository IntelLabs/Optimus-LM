import os

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

# Distributed setup
################################################################
dist_backend = "xccl"
rank, world_size, local_rank, local_world_size = 0,1,0,1
if int(os.getenv("PMI_SIZE", "1")) > 1:
    from optimus.dutils import setup_xpu_distributed
    rank, world_size, local_rank, local_world_size = setup_xpu_distributed(dist_backend=dist_backend)
################################################################

# Options
expert_parallelism = world_size

# Input configuration
batch_size = 1
seq_length = 4096
dtype = torch.float32
device = "xpu"
run_only_forward = False
global_batch_size = batch_size * expert_parallelism

# Configurations
config = OlmoeParallelConfig(expert_parallelism=expert_parallelism)
config_oms = OlmoeParallelConfig()

# Input and ouptut gradient tensors
hidden_states_oms = torch.randn((global_batch_size, seq_length, config_oms.hidden_size), dtype=dtype, device=device)
output_grad_oms = torch.randn((global_batch_size, seq_length, config_oms.hidden_size), dtype=dtype, device=device)

hidden_states = torch.zeros((batch_size, seq_length, config.hidden_size), dtype=dtype, device=device)
output_grad = torch.zeros((batch_size, seq_length, config.hidden_size), dtype=dtype, device=device)

with torch.no_grad():
    hidden_states.copy_(hidden_states_oms[rank*batch_size:(rank+1)*batch_size])
    output_grad.copy_(output_grad_oms[rank*batch_size:(rank+1)*batch_size])

# Enable requires_grad for backward pass
if not run_only_forward:
    hidden_states.requires_grad_(True)
    hidden_states_oms.requires_grad_(True)
    
# Modules
###################################################################
# Hugging face model for initializing serial and parallel OM model
config_hf = AutoConfig.from_pretrained("allenai/OLMoE-1B-7B-0924")
model_hf = OlmoeSparseMoeBlock(config_hf)
#####################################################################
model = OlmoeParallelSparseMoeBlock(config, pmap=ParallelDpPpEpTpMapper(expert_parallelism=expert_parallelism, rank=rank, create_groups=True))
model_oms = OlmoeParallelSparseMoeBlock(config_oms, pmap=ParallelDpPpEpTpMapper(expert_parallelism=1, rank=rank, create_groups=False))

# Copy from HF model
with torch.no_grad():
    model.set_parameters_from_full_module(model_hf)
    model_oms.set_parameters_from_full_module(model_hf)

# Set dtype and device
model = model.to(dtype).to(device)
model_oms = model_oms.to(dtype).to(device)
model_hf = model_hf.to(dtype).to(device)

# Forward pass
output = model(hidden_states)[0]
output_oms = model_oms(hidden_states_oms)[0]
# output_oms = model_hf(hidden_states_oms)[0]

# Backward pass
if not run_only_forward:
    output.backward(output_grad)
    output_oms.backward(output_grad_oms)

# Output comparision
from optimus.utils import print_error_info

if model.pmap.expert_parallelism > 1:
    output_all = torch.zeros((global_batch_size, seq_length, config_hf.hidden_size), dtype=dtype, device=device)
    torch.distributed.all_gather_into_tensor(output_all, output, group=model.pmap.ep_group)

    if not run_only_forward:
        input_grad_all = torch.zeros((global_batch_size, seq_length, config_hf.hidden_size), dtype=dtype, device=device)
        torch.distributed.all_gather_into_tensor(input_grad_all, hidden_states.grad, group=model.pmap.ep_group)

    if rank == 0:
        print_error_info(output_all, output_oms, f"Rank : {rank} Output")
        if not run_only_forward:
            print_error_info(input_grad_all, hidden_states_oms.grad, f"Rank : {rank} Input gradient")
else:
    print_error_info(output, output_oms, f"Rank : {rank} Output")
    if not run_only_forward:
        print_error_info(hidden_states.grad, hidden_states_oms.grad, f"Rank : {rank} Input gradient")

if torch.distributed.is_initialized():
    torch.xpu.synchronize()
    torch.distributed.barrier()

# export PYTHONPATH=$PYTHONPATH:$PWD
# bash launch_dist.sh 2 1 python validation/olmoe/test_OlmoeExpertParallelSparseMoeBlock.py