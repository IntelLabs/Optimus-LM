import torch

from transformers import AutoConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding

from optimus.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3ParallelConfig
from optimus.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ParallelRotaryEmbedding

from optimus.utils import print_error_info

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

# For reproducibility
torch.manual_seed(0)

# Optimus tensors
input = torch.randn((batch_size, sequence_length, config.hidden_size), dtype=dtype, device=device)
position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)

# HF tensors
input_hf = input.detach().clone()
position_ids_hf = position_ids.detach().clone()


# Optimus and HF modules
model_hf = DeepseekV3RotaryEmbedding(config_hf).to(dtype).to(device)
model = DeepseekV3ParallelRotaryEmbedding(config).to(dtype).to(device)

output_hf = model_hf(input_hf, position_ids_hf)
output = model(input, position_ids)

print_error_info(output[0], output_hf[0], "Cos")
print_error_info(output[1], output_hf[1], "Sin")