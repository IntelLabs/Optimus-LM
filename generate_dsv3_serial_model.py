import os
import time

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm, DeepseekV3TopkRouter

from optimus.config_utils import get_modified_config

# Ensuring same parameters are created each time
torch.manual_seed(0)

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

# Create dummy model
print("Creating empty init model...", flush=True)
st = time.time()
hf_model_choice = model_choice.split("--")[0]
config_hf = AutoConfig.from_pretrained(hf_model_choice)
config_hf = get_modified_config(model_choice, config_hf)
with init_empty_weights():
    model_hf = AutoModelForCausalLM.from_config(config_hf)
et = time.time()
print(f"Completed dummy model creation in {(et-st)/60:.2f} minutes", flush=True)

param_names = [n for n,p in model_hf.named_parameters()]
print(f"Number of parameter names : {len(param_names)}", flush=True)
print(f"Number of parameters : {sum(p.numel() for p in model_hf.parameters())}", flush=True)

# Initialization
print("Starting serial model creation...", flush=True)
st = time.time()

saved_param_names = []
def save_parameter(tensor, name):
    tensor_path = os.path.join(serial_model_dir, f"{name}.pt")
    if not os.path.exists(tensor_path):
        torch.save(tensor, tensor_path)
    saved_param_names.append(name)
    print(f"Processed {len(saved_param_names)} / {len(param_names)}", flush=True)
    del tensor

std = config_hf.initializer_range
for name, module in model_hf.named_modules():
    if isinstance(module, nn.Linear):
        weight = torch.empty_like(module.weight, device="cpu")
        weight.normal_(mean=0.0, std=std)
        save_parameter(weight, f"{name}.weight")

        if module.bias is not None:
            bias = torch.empty_like(module.bias, device="cpu")
            bias.zero_()
            save_parameter(bias, f"{name}.bias")
    elif isinstance(module, DeepseekV3RMSNorm):
        weight = torch.empty_like(module.weight, device="cpu")
        weight.fill_(1.0)
        save_parameter(weight, f"{name}.weight")
    elif isinstance(module, nn.Embedding):
        weight = torch.empty_like(module.weight, device="cpu")
        weight.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            weight[module.padding_idx].zero_()
        save_parameter(weight, f"{name}.weight")
    elif isinstance(module, DeepseekV3TopkRouter):
        weight = torch.empty_like(module.weight, device="cpu")
        weight.normal_(mean=0.0, std=std)
        save_parameter(weight, f"{name}.weight")
    else:
        # Check if module is leaf
        if (len(list(module.children())) == 0) and (module._parameters != {}):
            print(f"WARNING : Module {name} of type {type(module)} is a leaf module but not handled explicitly", flush=True)
            exit(-1)

et = time.time()
print(f"Time taken for serial model creation : {(et-st)/60:.2f} minutes", flush=True)

# Check all parameters are saved
missing_params = set(param_names) - set(saved_param_names)
if len(missing_params) > 0:
    print("FAILURE : Following parameters are missing in the saved serial model:", flush=True)
    for p in missing_params:
        print(p, flush=True)
else:
    print(f"SUCCESS : All parameters are saved in the serial model at {serial_model_dir}", flush=True)