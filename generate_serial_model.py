import os
import time

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeRMSNorm

from optimus.config_utils import get_modified_config

# Ensuring same parameters are created each time
torch.manual_seed(0)

# model_choice = "allenai/OLMoE-1B-7B-0924--SE1.125" # (8B)
# model_choice = "allenai/OLMoE-1B-7B-0924--SA3-SH1.5-SE2.25-SC0.5-SI2-L36" # (100B)
# model_choice = "allenai/OLMoE-1B-7B-0924--SA7-SH3.5-SE3.75-SI2-L48" # (518B)
# model_choice = "allenai/OLMoE-1B-7B-0924--SA4.5-SH2.25-SE6-SC0.75-SI2-L96" # (1T)

# OLMoE (Paper variants)
# model_choice = "allenai/OLMo-1B-hf"
# model_choice = "allenai/OLMoE-1B-7B-0924" #(OLMoE)
# model_choice = "allenai/OLMoE-1B-7B-0924--SE1.5-L32" #(OLMoE-L)
# model_choice = "allenai/OLMoE-1B-7B-0924--SA1.5-SH1.5-SE2.25-SI1.5-L48" #(OLMoE-XL)
# model_choice = "allenai/OLMoE-1B-7B-0924--SA1.5-SH1.5-SE3.75-SI1.5-L64" #(OLMoE-XXL)

import argparse

parser = argparse.ArgumentParser(description="Generate serial model for OLMoE")
parser.add_argument("--model_choice", type=str, default="allenai/OLMoE-1B-7B-0924", help="Model choice for OLMoE")
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
    elif isinstance(module, OlmoeRMSNorm):
        weight = torch.empty_like(module.weight, device="cpu")
        weight.fill_(1.0)
        save_parameter(weight, f"{name}.weight")
    elif isinstance(module, nn.Embedding):
        weight = torch.empty_like(module.weight, device="cpu")
        weight.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            weight[module.padding_idx].zero_()
        save_parameter(weight, f"{name}.weight")

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