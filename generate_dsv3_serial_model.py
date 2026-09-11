import os
import time

import torch
import torch.nn as nn

from optimus.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3ParallelConfig
from optimus.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ParallelForCausalLM, DeepseekV3ParallelRMSNorm, DeepseekV3ParallelTopkRouter

from optimus.mapper import ParallelDpPpEpTpMapper
from optimus.config_utils import get_modified_config
from optimus.utils import no_init_weights

###############################################################
# Distributed initialization
rank, world_size, local_rank, local_world_size = 0,1,0,1
if int(os.getenv("PMI_SIZE", "1")) > 1:
    from optimus.dutils import setup_xpu_distributed
    rank, world_size, local_rank, local_world_size = setup_xpu_distributed(dist_backend="xccl")

if torch.distributed.is_initialized():
    torch.distributed.barrier()
if rank == 0:
    print("Distributed initialization completed", flush=True)
###############################################################

# Seed seed based on rank
torch.manual_seed(rank)

# DSV3
# model_choice = "DeepSeek-V3--L4-H1024-I1536-MI512-N264" #(Test)
# model_choice = "deepseek-ai/DeepSeek-V3--L32-DL1-H2048-I10944-MI1408-E64-C6-SE2-A16" #(Debug)
# model_choice = "deepseek-ai/DeepSeek-V3--N264-L96" #(DSV3 scaled to Trillion parameter model)

import argparse
parser = argparse.ArgumentParser(description="Generate serial model for DSV3")
parser.add_argument("--model_choice", type=str, default="deepseek-ai/DeepSeek-V3--N264-L96", help="Model choice for DSV3")
args = parser.parse_args()

# Arguments
serial_models_dump_dir = "./serial_models"
model_choice = args.model_choice

# Create directory to save serial model
serial_model_dir_name = model_choice.replace('/','__')
serial_model_dir = os.path.join(serial_models_dump_dir, serial_model_dir_name)
os.makedirs(serial_model_dir, exist_ok=True)

# Serial (unsharded) mapper : a single full copy of the model, same as in generate_dsv3_parallel_model.py
pmap = ParallelDpPpEpTpMapper(
            data_parallelism=1,
            pipeline_parallelism=1,
            expert_parallelism=1,
            tensor_parallelism=1,
            rank=0,
            create_groups=False)

# Create dummy model
if rank == 0:
    print("Creating empty init model...", flush=True)
st = time.time()

config = DeepseekV3ParallelConfig(
        pipeline_parallelism=1,
        expert_parallelism=1,
        tensor_parallelism=1,
        virtual_pipeline_parallelism=1,
    )
config = get_modified_config(model_choice, config)

with no_init_weights():
    model = DeepseekV3ParallelForCausalLM(config, pmap=pmap)

et = time.time()
if rank == 0:
    print(f"Completed dummy model creation in {(et-st)/60:.2f} minutes", flush=True)

param_names = [n for n,p in model.named_parameters()]
if rank == 0:
    print(f"Number of parameter names : {len(param_names)}", flush=True)
    print(f"Number of parameters : {sum(p.numel() for p in model.parameters())}", flush=True)

# Initialization
if rank == 0:
    print("Starting serial model creation...", flush=True)
st = time.time()

saved_param_names = []
def save_parameter(tensor, name):
    tensor_path = os.path.join(serial_model_dir, f"{name}.pt")
    if not os.path.exists(tensor_path):
        torch.save(tensor, tensor_path)
    saved_param_names.append(name)
    # print(f"Processed {len(saved_param_names)} / {len(param_names)}", flush=True)
    print(f"Rank {rank} : processed {name}", flush=True)
    del tensor

module_id = 0

std = config.initializer_range
for name, module in model.named_modules():
    if (module_id % world_size) == rank:
        if isinstance(module, nn.Linear):
            weight = torch.empty_like(module.weight, device="cpu")
            weight.normal_(mean=0.0, std=std)
            save_parameter(weight, f"{name}.weight")

            if module.bias is not None:
                bias = torch.empty_like(module.bias, device="cpu")
                bias.zero_()
                save_parameter(bias, f"{name}.bias")
        elif isinstance(module, DeepseekV3ParallelRMSNorm):
            weight = torch.empty_like(module.weight, device="cpu")
            weight.fill_(1.0)
            save_parameter(weight, f"{name}.weight")
        elif isinstance(module, nn.Embedding):
            weight = torch.empty_like(module.weight, device="cpu")
            weight.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                weight[module.padding_idx].zero_()
            save_parameter(weight, f"{name}.weight")
        elif isinstance(module, DeepseekV3ParallelTopkRouter):
            weight = torch.empty_like(module.weight, device="cpu")
            weight.normal_(mean=0.0, std=std)
            save_parameter(weight, f"{name}.weight")
        else:
            # Check if module is leaf
            if (len(list(module.children())) == 0) and (module._parameters != {}):
                print(f"WARNING : Module {name} of type {type(module)} is a leaf module but not handled explicitly", flush=True)
                exit(-1)
        
    # Increment module
    module_id += 1

et = time.time()

if torch.distributed.is_initialized():
    torch.distributed.barrier()

if rank == 0:
    print(f"Rank {rank} : Time taken for serial model creation : {(et-st)/60:.2f} minutes", flush=True)