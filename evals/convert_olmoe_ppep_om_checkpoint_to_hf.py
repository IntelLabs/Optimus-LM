import os
import sys

import argparse

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file

from optimus.config_utils import get_modified_config


def set_parameters_from_parallel_model(model_state_dict, parallel_model_state_dict, config, pipeline_parallelism, expert_parallelism, pp_ind, ep_ind, use_init_empty_weights=False):
    for name,tensor in parallel_model_state_dict.items():
        # Adjusting layer_id
        if "layers" in name:
            local_layer_id = int(name.split(".")[2])
            num_hidden_layers_per_rank = config.num_hidden_layers // pipeline_parallelism
            layer_id = pp_ind * num_hidden_layers_per_rank + local_layer_id
            name = name.replace(f"layers.{local_layer_id}.", f"layers.{layer_id}.")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Optimus checkpoint to Hugging Face format")
    parser.add_argument("--model_choice", type=str, default=None, help="Hugging Face model choice")
    parser.add_argument("--om_checkpoint_dir", type=str, default=None, help="Directory containing Optimus checkpoint")
    parser.add_argument("--hf_checkpoint_dir", type=str, default="checkpoint_hf", help="Directory to save Hugging Face checkpoint")
    parser.add_argument("--use_init_empty_weights", action='store_true', help="Use init_empty_weights for model creation")

    args = parser.parse_args()

    print(args)

    pipeline_parallelism = 4
    expert_parallelism = 12
    hf_model_choice = args.model_choice.split("--")[0]

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
        for pp_ind in range(pipeline_parallelism):
            for ep_ind in range(expert_parallelism):
                mp_ind = pp_ind * expert_parallelism + ep_ind
                print(f"Processing parallel model {mp_ind} (PP {pp_ind}, EP {ep_ind})...", flush=True)
                parallel_model_cp_path = os.path.join(args.om_checkpoint_dir, f"model_checkpoint_shard-{mp_ind}.pth")
                parallel_model_cp = torch.load(parallel_model_cp_path, map_location='cpu')
                parallel_model_cp_state_dict = parallel_model_cp["state_dict"]
                set_parameters_from_parallel_model(model.state_dict(), parallel_model_cp_state_dict, config, pipeline_parallelism, expert_parallelism, pp_ind, ep_ind, args.use_init_empty_weights)
        
        
    # Save HF model
    print(f"Saving HF checkpoint to {args.hf_checkpoint_dir}", flush=True)
    model.save_pretrained(args.hf_checkpoint_dir)
    tokenizer.save_pretrained(args.hf_checkpoint_dir)
    print(f"Saving checkpint done")


    """
    python convert_olmoe_ppep_om_checkpoint_to_hf.py --model_choice allenai/OLMoE-1B-7B-0924--SA3-SH1.5-SE2.25-SC0.5-SI2-L36 --om_checkpoint_dir ../pretrain_parallel_models/OLMoE-1B-7B-0924--SA3-SH1.5-SE2.25-SC0.5-SI2-L36_om_omcached_gral_ac1_bf16_n256xg12_gbs6144-bs8-cs4096-gas1_dp64-pp4-ep12-tp1-ps1f1b-mbs1-rb1_omopt-pgshard-pgar_final/checkpoints/main --use_init_empty_weights

    """