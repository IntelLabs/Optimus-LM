import os

import torch
import torch.nn as nn

from transformers import AutoConfig

from optimus.models.olmo.configuration_olmo import OlmoParallelConfig
from optimus.models.olmo.modeling_olmo import OlmoParallelForCausalLM

from optimus.models.olmoe.configuration_olmoe import OlmoeParallelConfig
from optimus.models.olmoe.modeling_olmoe import OlmoeParallelForCausalLM

from optimus.dutils import setup_xpu_distributed
from optimus.mapper import ParallelDpPpEpTpMapper
from optimus.config_utils import get_modified_config
from optimus.utils import no_init_weights

# OLMoE (Paper variants)
# model_choice = "allenai/OLMo-1B-hf"
# model_choice = "allenai/OLMoE-1B-7B-0924" #(OLMoE)
# model_choice = "allenai/OLMoE-1B-7B-0924--SE1.5-L32" #(OLMoE-L)
# model_choice = "allenai/OLMoE-1B-7B-0924--SA1.5-SH1.5-SE2.25-SI1.5-L48" #(OLMoE-XL)
# model_choice = "allenai/OLMoE-1B-7B-0924--SA1.5-SH1.5-SE3.75-SI1.5-L64" #(OLMoE-XXL)


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

import argparse
parser = argparse.ArgumentParser(description="Generate parallel model for OLMoE")
parser.add_argument("--model_choice", type=str, default="allenai/OLMoE-1B-7B-0924", help="Model choice for OLMoE")
parser.add_argument("--expert_parallelism", type=int, default=1, help="Expert parallelism")
parser.add_argument("--pipeline_parallelism", type=int, default=1, help="Pipeline parallelism")
parser.add_argument("--virtual_pipeline_parallelism", type=int, default=1, help="Virtual pipeline parallelism")
parser.add_argument("--use_merged_mlp_in_fast_moe", action="store_true", help="Use merged MLP in Fast MoE")

args = parser.parse_args()

# Arguments
serial_models_dump_dir = "./serial_models"
parallel_models_dump_dir = "./parallel_models"

serial_model_choice = args.model_choice
model_choice = serial_model_choice

# Parallelization configuration
pipeline_parallelism = args.pipeline_parallelism
expert_parallelism = args.expert_parallelism
tensor_parallelism = 1
virtual_pipeline_parallelism = args.virtual_pipeline_parallelism
use_merged_mlp_in_fast_moe = args.use_merged_mlp_in_fast_moe

assert world_size == (pipeline_parallelism * expert_parallelism * tensor_parallelism), f"World size {world_size} does not match pp*ep*tp {pipeline_parallelism * expert_parallelism * tensor_parallelism}"

pmap = ParallelDpPpEpTpMapper(
            data_parallelism=1, 
            pipeline_parallelism=pipeline_parallelism,
            expert_parallelism=expert_parallelism,
            tensor_parallelism=tensor_parallelism, 
            rank=rank, 
            create_groups=False)

# Model config
hf_model_choice = model_choice.split("--")[0]
if hf_model_choice == "allenai/OLMo-1B-hf":
    config_hf_ref = AutoConfig.from_pretrained(hf_model_choice)
    config = OlmoParallelConfig(
        hidden_size=config_hf_ref.hidden_size,
        intermediate_size= config_hf_ref.intermediate_size,
        num_hidden_layers = config_hf_ref.num_hidden_layers,
        num_attention_heads = config_hf_ref.num_attention_heads,                
        # Parallel
        tensor_parallelism=tensor_parallelism,
        pipeline_parallelism=pipeline_parallelism,
        virtual_pipeline_parallelism=virtual_pipeline_parallelism)

    config = get_modified_config(model_choice, config)
    config.check_compatibility()

    with no_init_weights():
        model = OlmoParallelForCausalLM(config, pmap=pmap).to(torch.bfloat16)

elif hf_model_choice == "allenai/OLMoE-1B-7B-0924":
    config = OlmoeParallelConfig(
            pipeline_parallelism=pipeline_parallelism,
            expert_parallelism=expert_parallelism,
            tensor_parallelism=tensor_parallelism,
            virtual_pipeline_parallelism=virtual_pipeline_parallelism,
            use_fast_moe=True,
            use_merged_mlp_in_fast_moe=use_merged_mlp_in_fast_moe
        )

    config = get_modified_config(model_choice, config)
    config.check_compatibility()

    with no_init_weights():
        model = OlmoeParallelForCausalLM(config, pmap=pmap).to(torch.bfloat16)
else:
    exit(-1)

########################################################################################
serial_model_dir_name = serial_model_choice.replace('/','__')
serial_model_dir = os.path.join(serial_models_dump_dir, serial_model_dir_name)
for name, param in model.named_parameters():
    # print(f"Rank {pmap.rank} : Processing {name}", flush=True)
    orig_name = name

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

    # Adjust expert_id
    if ("experts." in name) and (not use_merged_mlp_in_fast_moe):
        local_expert_id = int(name.split(".")[5])
        num_experts_per_rank = config.num_experts // pmap.expert_parallelism
        expert_id = pmap.ep_ind * num_experts_per_rank + local_expert_id
        name = name.replace(f"experts.{local_expert_id}.", f"experts.{expert_id}.")

    # Handle experts differently
    if ("experts." in name) and (use_merged_mlp_in_fast_moe):
        #model.layers.0.mlp.experts.gate_proj_merged_weight
        linear_key = name.split(".")[-1].replace("_merged_weight","")
        prefix = name.replace(f".{linear_key}_merged_weight","")
        num_experts_per_rank = config.num_experts // pmap.expert_parallelism

        for local_expert_id in range(num_experts_per_rank):
            expert_id = pmap.ep_ind * num_experts_per_rank + local_expert_id
            serial_name = f"{prefix}.{expert_id}.{linear_key}.weight"
            tensor_path = os.path.join(serial_model_dir, f"{serial_name}.pt")
            # print(f"Rank {rank} : Copying {serial_name} to {name} ({local_expert_id})")
            tensor = torch.load(tensor_path, map_location="cpu")
            param.data[local_expert_id].copy_(tensor)
    else:
        # if orig_name != name:
        #     print(f"Rank {rank} : Replacing {orig_name} to {name}", flush=True)

        tensor_path = os.path.join(serial_model_dir, f"{name}.pt")
        tensor = torch.load(tensor_path, map_location="cpu")
        param.data.copy_(tensor)

print(f"Rank {pmap.rank} : Model parameters initialized from serial model", flush=True)
########################################################################################

# Save sharded model
parallel_model_dir_name = f"{model_choice.replace('/','__')}_pp{pipeline_parallelism}-ep{expert_parallelism}-tp{tensor_parallelism}-vpp{virtual_pipeline_parallelism}"
if use_merged_mlp_in_fast_moe:
    parallel_model_dir_name += "_mmlp"
parallel_model_dir = os.path.join(parallel_models_dump_dir, parallel_model_dir_name)
state_dict_path = os.path.join(parallel_model_dir, f"model_{pmap.mp_ind}.pth")

os.makedirs(parallel_model_dir, exist_ok=True)
torch.save(model.state_dict(), state_dict_path)
print(f"Rank {pmap.rank} : Saved the model at {state_dict_path}", flush=True)

if torch.distributed.is_initialized():
    torch.distributed.barrier()

# python generate_parallel_model.py
# bash launch_dist.sh 12 1 python generate_parallel_model.py
# bash launch_dist.sh 12 4 python generate_parallel_model.py
# bash launch_dist.sh 12 8 python generate_parallel_model.py