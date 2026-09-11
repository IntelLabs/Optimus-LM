import torch

from transformers import AutoConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3DecoderLayer
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding

from optimus.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3ParallelConfig
from optimus.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ParallelDecoderLayer
from optimus.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ParallelRotaryEmbedding

from optimus.utils import print_error_info

# Configuration
config = DeepseekV3ParallelConfig()
config_hf = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3")
config_hf._attn_implementation = "eager"
config.skip_causal_mask = True  # HF follows sending attention_mask from decoder. Optimus does not send an attention_mask, but we want to skip causal mask for correctness check.

# Run options
import json
with open("validation/deepseekv3/input_config.json", "r") as f:
    input_config = json.load(f)

batch_size = input_config["batch_size"]
sequence_length = input_config["sequence_length"]
dtype = getattr(torch, input_config["dtype"])
device = input_config["device"]
test_only_forward = input_config["test_only_forward"]
layer_idx = input_config["layer_idx"]

# For reproducibility
torch.manual_seed(0)

# Optimus tensors
input = torch.randn((batch_size, sequence_length, config.hidden_size), dtype=dtype, device=device)
output_grad = torch.randn((batch_size, sequence_length, config.hidden_size), dtype=dtype, device=device)

# HF tensors
input_hf = input.detach().clone()
output_grad_hf = output_grad.detach().clone()

if not test_only_forward:
    input.requires_grad = True
    input_hf.requires_grad = True

# HF and Optimus modules
module_hf = DeepseekV3DecoderLayer(config_hf, layer_idx=layer_idx).to(dtype).to(device)
module = DeepseekV3ParallelDecoderLayer(config, layer_idx=layer_idx).to(dtype).to(device)
module.set_parameters_from_full_module(module_hf)

position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)
rotary_emb = DeepseekV3ParallelRotaryEmbedding(config).to(dtype).to(device)
rotary_emb_hf = DeepseekV3RotaryEmbedding(config_hf).to(dtype).to(device)

position_embeddings = rotary_emb(input, position_ids)
position_embeddings_hf = rotary_emb_hf(input_hf, position_ids)

# Forward pass
output = module(input, position_embeddings)
output_hf = module_hf(input_hf, position_embeddings=position_embeddings_hf, attention_mask=None)

# Backward pass
if not test_only_forward:
    output.backward(output_grad)
    output_hf.backward(output_grad_hf)

# Correctness check
print_error_info(output, output_hf, "Output")
if not test_only_forward:
    print_error_info(input.grad, input_hf.grad, "Input Grad")