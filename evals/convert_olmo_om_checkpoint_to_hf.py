import os
import sys

import argparse

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from optimus.config_utils import get_modified_config

def set_parameters_from_parallel_model(model_state_dict, parallel_model_state_dict, config, use_init_empty_weights=False):
    for name,tensor in parallel_model_state_dict.items():        
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
    hf_model_choice = "allenai/OLMo-1B-hf"

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

    model_checkpoint_shard = f"model_checkpoint.pth"
    model_cp_path = os.path.join(args.om_checkpoint_dir, model_checkpoint_shard)
    model_cp = torch.load(model_cp_path, map_location='cpu')
    model_cp_state_dict = model_cp["state_dict"]

    set_parameters_from_parallel_model(model.state_dict(), model_cp_state_dict, config, args.use_init_empty_weights)

    # Save HF model
    print(f"Saving HF checkpoint to {args.hf_checkpoint_dir}", flush=True)
    model.save_pretrained(args.hf_checkpoint_dir)
    tokenizer.save_pretrained(args.hf_checkpoint_dir)
    print(f"Saving checkpoint done")

    # python convert_olmo_om_checkpoint_to_hf.py --model_choice allenai/OLMo-1B-hf --om_checkpoint_dir ../pretrain_parallel_models/OLMo-1B-hf_olmoemix0924_om_omcached_bf16_n256xg12_gbs3072-bs1-cs2048-gas1_dp3072-pp1-ep1-tp1_omopt-shard_final/checkpoints/step1000-tokens6B/ --use_init_empty_weights

    # python convert_olmo_om_checkpoint_to_hf.py --model_choice allenai/OLMo-1B-hf --om_checkpoint_dir ../pretrain_parallel_models/OLMo-1B-hf_olmoemix0924_om_finetune_bf16_n1xg12_gbs12-bs1-cs2048-gas1_dp12-pp1-ep1-tp1_sat-uniformblock-1x16-0.3125_omopt-shard_debug/checkpoints/step2000-tokens0B/ --use_init_empty_weights