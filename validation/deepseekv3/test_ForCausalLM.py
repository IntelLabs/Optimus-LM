import torch

from transformers import AutoConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM

from optimus.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3ParallelConfig
from optimus.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ParallelForCausalLM

from optimus.utils import print_error_info

# Optimus and HF configs
config = DeepseekV3ParallelConfig()
config_hf = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3")
config_hf._attn_implementation = "eager"

# Making the model smaller for testing
config_hf.num_hidden_layers = 4
config_hf.hidden_size = 1024
config_hf.n_routed_experts = 32

config.num_hidden_layers = config_hf.num_hidden_layers
config.hidden_size = config_hf.hidden_size
config.n_routed_experts = config_hf.n_routed_experts

# Run options
import json
with open("validation/deepseekv3/input_config.json", "r") as f:
    input_config = json.load(f)

batch_size = input_config["batch_size"]
sequence_length = input_config["sequence_length"]
dtype = getattr(torch, input_config["dtype"])
device = input_config["device"]
test_only_forward = input_config["test_only_forward"]

# For reproducibility
torch.manual_seed(0)

# Optimus tensors
input = torch.randint(0, config_hf.vocab_size, (batch_size, sequence_length), dtype=torch.int64, device=device)
output_grad = torch.randn((batch_size, sequence_length, config_hf.vocab_size), dtype=dtype, device=device)

# HF tensors
input_hf = input.detach().clone()
output_grad_hf = output_grad.detach().clone()

# HF and Optimus modules
module_hf = DeepseekV3ForCausalLM(config_hf)
module = DeepseekV3ParallelForCausalLM(config)
module.set_parameters_from_full_module(module_hf)

module = module.to(dtype).to(device)
module_hf = module_hf.to(dtype).to(device)

# Forward pass
output = module(input).logits
output_hf = module_hf(input_hf).logits

# Backward pass
if not test_only_forward:
    output.backward(output_grad)
    output_hf.backward(output_grad_hf)

# Correctness check
print_error_info(output, output_hf, "Output")
if not test_only_forward:
    print_error_info(module.model.embed_tokens.weight.grad, module_hf.model.embed_tokens.weight.grad, "Embedding weight grad")