import os

import torch
import intel_extension_for_pytorch

# Transformer imports
from transformers import AutoConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeModel
from transformers.models.olmoe.modeling_olmoe import OlmoeRotaryEmbedding

# Optimus imports
from optimus.models.olmoe.configuration_olmoe import OlmoeParallelConfig
from optimus.models.olmoe.modeling_olmoe import OlmoeParallelModel

from optimus.utils import print_error_info
from optimus.mapper import ParallelDpPpEpTpMapper

# Reproducibility
torch.manual_seed(42)

# Input configuration
batch_size = 2
seq_length = 4096
dtype = torch.float32
device = "xpu"
run_only_forward = False

# Configurations
config = OlmoeParallelConfig()
config_hf = AutoConfig.from_pretrained("allenai/OLMoE-1B-7B-0924")

# Setting number of layers to 1 for quick check
config.num_hidden_layers = 1
config_hf.num_hidden_layers = 1

# Input and ouptut gradient tensors
input_ids = torch.randint(0, config_hf.vocab_size, (batch_size, seq_length), dtype=torch.int64, device=device)
output_grad = torch.randn((batch_size, seq_length, config.hidden_size), dtype=dtype, device=device)

print(config_hf)

# Modules
model_hf = OlmoeModel(config_hf)
model = OlmoeParallelModel(config, pmap=ParallelDpPpEpTpMapper())
with torch.no_grad():
    model.load_state_dict(model_hf.state_dict())

# Set dtype and device
model = model.to(dtype).to(device)
model_hf = model_hf.to(dtype).to(device)

# Forward pass
output = model(input_ids)[0]
output_hf = model_hf(input_ids)[0]

# Backward pass
if not run_only_forward:
    output.backward(output_grad)
    output_hf.backward(output_grad)

# Compare output
print_error_info(output, output_hf, "Output")
if not run_only_forward:
    embed_tokens_grad = model.embed_tokens.weight.grad
    embed_tokens_grad_hf = model_hf.embed_tokens.weight.grad
    unique_input_ids = torch.unique(input_ids)
    print_error_info(embed_tokens_grad[unique_input_ids], embed_tokens_grad_hf[unique_input_ids], "Embedding gradient")