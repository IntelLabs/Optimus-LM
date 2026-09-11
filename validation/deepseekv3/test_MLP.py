import torch

from transformers import AutoConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MLP

from optimus.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3ParallelConfig
from optimus.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ParallelMLP

from optimus.utils import print_error_info

# For reproducibility
torch.manual_seed(0)

# Optimus and HF configs
config = DeepseekV3ParallelConfig()
config_hf = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3")

# Run options
import json
with open("validation/deepseekv3/input_config.json", "r") as f:
    input_config = json.load(f)

batch_size = input_config["batch_size"]
sequence_length = input_config["sequence_length"]
dtype = getattr(torch, input_config["dtype"])
device = input_config["device"]
test_only_forward = input_config["test_only_forward"]

# Optimus tensors
input = torch.randn((batch_size, sequence_length, config.hidden_size), dtype=dtype, device=device)
output_grad = torch.randn((batch_size, sequence_length, config.hidden_size), dtype=dtype, device=device)

# HF tensors
input_hf = input.detach().clone()
output_grad_hf = output_grad.detach().clone()

if not test_only_forward:
    input.requires_grad = True
    input_hf.requires_grad = True

# HF and optimus modules
module_hf = DeepseekV3MLP(config_hf)
module = DeepseekV3ParallelMLP(config)
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