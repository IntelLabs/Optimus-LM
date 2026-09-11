import os

import torch
import torch.nn as nn

from optimus.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3ParallelConfig
from optimus.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ParallelForCausalLM

from optimus.dutils import setup_xpu_distributed
from optimus.mapper import ParallelDpPpEpTpMapper
from optimus.config_utils import get_modified_config
from optimus.utils import no_init_weights


###############################################################
# Distributed initialization
import os
rank, world_size, local_rank, local_world_size = 0,1,0,1
if int(os.getenv("PMI_SIZE", "1")) > 1:
    from optimus.dutils import setup_xpu_distributed
    rank, world_size, local_rank, local_world_size = setup_xpu_distributed(dist_backend="xccl")

if torch.distributed.is_initialized():
    torch.distributed.barrier()
if rank == 0:
    print("Distributed initialization completed", flush=True)
###############################################################

# DSV3
# model_choice = "DeepSeek-V3--L4-H1024-I1536-MI512-N264" #(Test)
# model_choice = "deepseek-ai/DeepSeek-V3--L32-DL1-H2048-I10944-MI1408-E64-C6-SE2-A16" #(Debug)
# model_choice = "deepseek-ai/DeepSeek-V3--N264-L96" #(DSV3 scaled to Trillion parameter model)

import argparse
parser = argparse.ArgumentParser(description="Generate parallel model for DSV3")
parser.add_argument("--model_choice", type=str, default="deepseek-ai/DeepSeek-V3--N264-L96", help="Model choice for DSV3")
parser.add_argument("--expert_parallelism", type=int, default=1, help="Expert parallelism")
parser.add_argument("--pipeline_parallelism", type=int, default=1, help="Pipeline parallelism")
parser.add_argument("--virtual_pipeline_parallelism", type=int, default=1, help="Virtual pipeline parallelism")
# parser.add_argument("--use_merged_mlp_in_fast_moe", action="store_true", help="Use merged MLP in Fast MoE")

args = parser.parse_args()

# Arguments
serial_models_dump_dir = "./serial_models"
parallel_models_dump_dir = "./parallel_models"

# Note : serial_model_choice is given as separate to avoid generating serial model for model variants with just 
# change in the number of layers. (Ex : If you generate serial model for 1T model deepseek-ai/DeepSeek-V3--N264-L96, you can use 
# the same serial model for all model variants with changes only in the number of layers. Ex : deepseek-ai/DeepSeek-V3--N264-L64)
# If you want to reuse the bigger model, manually change serial_model_choice to point to the bigger model
serial_model_choice = args.model_choice 
model_choice = args.model_choice

# Parallelization configuration
pipeline_parallelism = args.pipeline_parallelism
expert_parallelism = args.expert_parallelism
tensor_parallelism = 1
virtual_pipeline_parallelism = args.virtual_pipeline_parallelism

assert world_size == (pipeline_parallelism * expert_parallelism * tensor_parallelism), f"World size {world_size} does not match pp*ep {pipeline_parallelism * expert_parallelism * tensor_parallelism}"

pmap = ParallelDpPpEpTpMapper(
            data_parallelism=1, 
            pipeline_parallelism=pipeline_parallelism,
            expert_parallelism=expert_parallelism,
            rank=rank, 
            create_groups=False)

# Model config
hf_model_choice = model_choice.split("--")[0]
config = DeepseekV3ParallelConfig(
        pipeline_parallelism=pipeline_parallelism,
        expert_parallelism=expert_parallelism,
        tensor_parallelism=tensor_parallelism,
        virtual_pipeline_parallelism=virtual_pipeline_parallelism,
    )

config = get_modified_config(model_choice, config)

with no_init_weights():
    model = DeepseekV3ParallelForCausalLM(config, pmap=pmap).to(torch.bfloat16)

########################################################################################
serial_model_dir_name = serial_model_choice.replace('/','__')
serial_model_dir = os.path.join(serial_models_dump_dir, serial_model_dir_name)
for name, param in model.named_parameters():
    # print(f"Rank {pmap.rank} : Processing {name}", flush=True)
    orig_name = name

    split_linear_in_ofm = False
    split_linear_in_ifm = False

    # Adjusting layer_id
    if "layers" in name:
        local_layer_id = int(name.split(".")[2])
        num_hidden_layers_per_rank = config.num_hidden_layers // config.pipeline_parallelism
        num_hidden_layers_per_vpp_rank = num_hidden_layers_per_rank // config.virtual_pipeline_parallelism
        vp_ind = local_layer_id // num_hidden_layers_per_vpp_rank
        vp_local_layer_id = local_layer_id % num_hidden_layers_per_vpp_rank
        layer_id = (vp_ind * config.pipeline_parallelism + pmap.pp_ind) * num_hidden_layers_per_vpp_rank + vp_local_layer_id

        # print(f"{pmap.pp_ind} {pmap.ep_ind} {local_layer_id} {vp_ind} {vp_local_layer_id} => {layer_id}", flush=True)

        # layer_id = pmap.pp_ind * num_hidden_layers_per_rank + local_layer_id
        name = name.replace(f"layers.{local_layer_id}.", f"layers.{layer_id}.")

        if (expert_parallelism > 1) and (layer_id < config.first_k_dense_replace):
            if "gate_proj" in name or "up_proj" in name:
                split_linear_in_ofm = True
            if "down_proj" in name:
                split_linear_in_ifm = True

    # Adjust expert_id
    if (".experts." in name):
        local_expert_id = int(name.split(".")[5])
        num_experts_per_rank = config.n_routed_experts // pmap.expert_parallelism
        expert_id = pmap.ep_ind * num_experts_per_rank + local_expert_id
        name = name.replace(f".experts.{local_expert_id}.", f".experts.{expert_id}.")

    assert not (split_linear_in_ofm and split_linear_in_ifm), "Cannot split the same tensor both row-wise and column-wise"

    # Handle first dense MLP layers differently
    tensor_path = os.path.join(serial_model_dir, f"{name}.pt")
    tensor = torch.load(tensor_path, map_location="cpu")

    if split_linear_in_ofm:
        split_size = tensor.shape[0] // config.expert_parallelism
        start_ofm = pmap.ep_ind * split_size
        end_ofm = (pmap.ep_ind + 1) * split_size
        tensor = tensor[start_ofm:end_ofm, :]

    if split_linear_in_ifm:
        split_size = tensor.shape[1] // config.expert_parallelism
        start_ifm = pmap.ep_ind * split_size
        end_ifm = (pmap.ep_ind + 1) * split_size
        tensor = tensor[:, start_ifm:end_ifm]

    param.data.copy_(tensor)

print(f"Rank {pmap.rank} : Model parameters initialized from serial model", flush=True)
########################################################################################

# Save sharded model
parallel_model_dir_name = f"{model_choice.replace('/','__')}_pp{pipeline_parallelism}-ep{expert_parallelism}-tp{tensor_parallelism}-vpp{virtual_pipeline_parallelism}"
# if use_merged_mlp_in_fast_moe:
#     parallel_model_dir_name += "_mmlp"
parallel_model_dir = os.path.join(parallel_models_dump_dir, parallel_model_dir_name)
state_dict_path = os.path.join(parallel_model_dir, f"model_{pmap.mp_ind}.pth")

os.makedirs(parallel_model_dir, exist_ok=True)
torch.save(model.state_dict(), state_dict_path)
print(f"Rank {pmap.rank} : Saved the model at {state_dict_path}", flush=True)

if torch.distributed.is_initialized():
    torch.distributed.barrier()

# python generate_dsv3_parallel_model.py
# bash launch_dist.sh 12 1 python generate_dsv3_parallel_model.py
# bash launch_dist.sh 12 4 python generate_dsv3_parallel_model.py
# bash launch_dist.sh 12 8 python generate_dsv3_parallel_model.py
# bash launch_dist.sh 12 16 python generate_dsv3_parallel_model.py
# bash launch_dist.sh 12 24 python generate_dsv3_parallel_model.py