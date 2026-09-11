import json

import torch

from transformers import AutoConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm

from optimus.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3ParallelConfig
from optimus.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ParallelRMSNorm

from optimus.utils import print_error_info

# For reproducibility
torch.manual_seed(0)

# Optimus and HF configs
config = DeepseekV3ParallelConfig()
config_hf = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3")

# Run options
with open("validation/deepseekv3/input_config.json", "r") as f:
    input_config = json.load(f)

batch_size = input_config["batch_size"]
sequence_length = input_config["sequence_length"]
dtype = getattr(torch, input_config["dtype"])
device = input_config["device"]
test_only_forward = input_config["test_only_forward"]


# Optimus and HF tensors
input = torch.randn((batch_size, sequence_length, config.hidden_size), dtype=dtype, device=device)
output_grad = torch.randn((batch_size, sequence_length, config.hidden_size), dtype=dtype, device=device)

input_hf = input.detach().clone()
output_grad_hf = output_grad.detach().clone()

if not test_only_forward:
    input.requires_grad = True
    input_hf.requires_grad = True

# Optimus and HF modules
module_hf = DeepseekV3RMSNorm(config_hf.hidden_size, eps=config_hf.rms_norm_eps)
module = DeepseekV3ParallelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
with torch.no_grad():
    module_hf.weight.data.copy_(torch.randn(config_hf.hidden_size)) # To avoid all ones in the weight which can hide some errors
module.set_parameters_from_full_module(module_hf)

# Convert modules to the correct dtype and device
module = module.to(dtype).to(device)
module_hf = module_hf.to(dtype).to(device)

# Forward pass
output = module(input)
output_hf = module_hf(input_hf)

# Backward pass
if not test_only_forward:
    output.backward(output_grad)
    output_hf.backward(output_grad_hf)

# Correctness check
print_error_info(output, output_hf, "Output")
if not test_only_forward:
    print_error_info(input.grad, input_hf.grad, "Input Grad")