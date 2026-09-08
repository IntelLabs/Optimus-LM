import os
import sys

import argparse

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file

from optimus.config_utils import get_modified_config

"""
def set_parameters_from_parallel_model(model_state_dict, parallel_model_state_dict, config, expert_parallelism, ep_ind, use_init_empty_weights=False):
    for name,tensor in parallel_model_state_dict.items():
        # Adjust expert_id
        if "experts." in name:
            local_expert_id = int(name.split(".")[5])
            num_experts_per_rank = config.num_experts // expert_parallelism
            expert_id = ep_ind * num_experts_per_rank + local_expert_id
            name = name.replace(f"experts.{local_expert_id}.", f"experts.{expert_id}.")
        
        if not use_init_empty_weights:
            model_state_dict[name].copy_(tensor)
        else:
            if "." in name:
                module_name, param_name = name.rsplit(".", 1)
                module = model.get_submodule(module_name)
            else:
                module = model
                param_name = name
            module._parameters[param_name] = nn.Parameter(tensor)
"""

def set_parameters_from_parallel_model(model_state_dict, parallel_model_state_dict, config, expert_parallelism, ep_ind, use_init_empty_weights=False):

    is_merged_mlp = any("merged_weight" in name for name in parallel_model_state_dict.keys())
    
    if not is_merged_mlp:
        for name,tensor in parallel_model_state_dict.items():
            # Adjust expert_id
            if "experts." in name:
                local_expert_id = int(name.split(".")[5])
                num_experts_per_rank = config.num_experts // expert_parallelism
                expert_id = ep_ind * num_experts_per_rank + local_expert_id
                name = name.replace(f"experts.{local_expert_id}.", f"experts.{expert_id}.")
            
            if not use_init_empty_weights:
                model_state_dict[name].copy_(tensor)
            else:
                if "." in name:
                    module_name, param_name = name.rsplit(".", 1)
                    module = model.get_submodule(module_name)
                else:
                    module = model
                    param_name = name
                module._parameters[param_name] = nn.Parameter(tensor)
    else:
        num_experts_per_rank = config.num_experts // expert_parallelism
        expert_start_id = ep_ind * num_experts_per_rank
        expert_end_id = (ep_ind + 1) * num_experts_per_rank
        for name,tensor in parallel_model_state_dict.items():
            serial_names_list = []
            serial_tensor_list = []
            if "experts" in name:
                tokens = name.split(".")
                linear_key = tokens[-1].replace("_merged_weight","")
                for expert_id in range(expert_start_id, expert_end_id):    
                    serial_name = ".".join(tokens[:-1] + [str(expert_id), linear_key+".weight"])
                    serial_names_list.append(serial_name)
                    serial_tensor_list.append(tensor[expert_id - expert_start_id])
            else:
                serial_names_list.append(name)
                serial_tensor_list.append(tensor)
                
            for serial_name, serial_tensor in zip(serial_names_list, serial_tensor_list):
                # print(f"Setting parameter {serial_name} from parallel checkpoint")
                if not use_init_empty_weights:
                    model_state_dict[serial_name].copy_(serial_tensor)
                else:
                    if "." in serial_name:
                        module_name, param_name = serial_name.rsplit(".", 1)
                        module = model.get_submodule(module_name)
                    else:
                        module = model
                        param_name = serial_name
                    module._parameters[param_name] = nn.Parameter(serial_tensor)


if __name__ == "__main__":
    hf_model_choice = "allenai/OLMoE-1B-7B-0924"
    expert_parallelism = 1

    parser = argparse.ArgumentParser(description="Convert Optimus checkpoint to Hugging Face format")
    parser.add_argument("--model_choice", type=str, default=None, help="Hugging Face model choice")
    parser.add_argument("--om_checkpoint_dir", type=str, default=None, help="Directory containing Optimus checkpoint")
    parser.add_argument("--hf_checkpoint_dir", type=str, default="checkpoint_hf", help="Directory to save Hugging Face checkpoint")
    parser.add_argument("--use_init_empty_weights", action='store_true', help="Use init_empty_weights for model creation")

    args = parser.parse_args()

    print(args)

    assert args.model_choice.split("--")[0] == hf_model_choice, "Model choice should match the base model."

    ###########################################################################
    config = AutoConfig.from_pretrained(hf_model_choice)
    config = get_modified_config(args.model_choice, config)

    # Model and tokenizer
    print("Creating HF model...", flush=True)
    if args.use_init_empty_weights:
        from accelerate import init_empty_weights
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_choice)
    ############################################################################

    if args.om_checkpoint_dir is not None:
        for ep_ind in range(expert_parallelism):
            print(f"Processing model shard {ep_ind}")
            model_checkpoint_shard = f"model_checkpoint_shard-{ep_ind}.pth" if expert_parallelism > 1 else "model_checkpoint.pth"
            ep_model_cp_path = os.path.join(args.om_checkpoint_dir, model_checkpoint_shard)
            ep_model_cp = torch.load(ep_model_cp_path, map_location='cpu')
            ep_model_cp_state_dict = ep_model_cp["state_dict"]

            set_parameters_from_parallel_model(model.state_dict(), ep_model_cp_state_dict, config, expert_parallelism, ep_ind, args.use_init_empty_weights)
        
        
    # Save HF model
    print(f"Saving HF checkpoint to {args.hf_checkpoint_dir}", flush=True)
    model.save_pretrained(args.hf_checkpoint_dir)
    tokenizer.save_pretrained(args.hf_checkpoint_dir)
    print(f"Saving checkpint done")


    """
    python convert_olmoe_om_checkpoint_to_hf.py --model_choice allenai/OLMoE-1B-7B-0924--SE1.125 --om_checkpoint_dir ../pretrain_parallel_models/OLMoE-1B-7B-0924--SE1.125_om_gral_bf16_n128xg12_gbs1536-bs1-cs4096-gas1_dp128-pp1-ep12-tp1_omopt-pgshard_final/checkpoints/main/

    python convert_olmoe_om_checkpoint_to_hf.py --model_choice allenai/OLMoE-1B-7B-0924 --om_checkpoint_dir ../pretrain_parallel_models/OLMoE-1B-7B-0924_olmoemix0924_om_omcached_gral_fastmoe-mmlp_bf16_n256xg12_gbs3072-bs1-cs2048-gas1_dp3072-pp1-ep1-tp1_omopt-shard_final/checkpoints/step293000-tokens1843B --use_init_empty_weights

    """