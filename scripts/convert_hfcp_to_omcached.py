import os
import torch

from transformers import AutoConfig, AutoModelForCausalLM

from optimus.models.llama.configuration_llama import LlamaParallelConfig
from optimus.models.llama.modeling_llama import LlamaParallelForCausalLM
from optimus.mapper import ParallelDpPpEpTpMapper
from optimus.config_utils import get_modified_config
from optimus.utils import no_init_weights

###############################################################
# Distributed initialization
import os
rank, world_size, local_rank, local_world_size = 0,1,0,1
if int(os.getenv("PMI_SIZE", "1")) > 1:
    from optimus.dutils import setup_xpu_distributed
    rank, world_size, local_rank, local_world_size = setup_xpu_distributed(dist_backend="xccl")

if torch.distributed.is_initialized():
    torch.distributed.barrier()
if rank == 0:
    print("Distributed initialization completed", flush=True)
###############################################################

# Pretrained model
hf_model_choice = "meta-llama/Llama-3.1-8B"
model_choice = hf_model_choice
model_hf = AutoModelForCausalLM.from_pretrained(hf_model_choice)

# Optimus model
pipeline_parallelism = 2
expert_parallelism = 1
tensor_parallelism = 1
assert world_size == (pipeline_parallelism * expert_parallelism * tensor_parallelism), f"World size {world_size} does not match pp*ep*tp {pipeline_parallelism * expert_parallelism * tensor_parallelism}"

config_hf_ref = AutoConfig.from_pretrained(hf_model_choice)
config = LlamaParallelConfig(
    hidden_size=config_hf_ref.hidden_size,
    intermediate_size= config_hf_ref.intermediate_size,
    num_hidden_layers = config_hf_ref.num_hidden_layers,
    num_attention_heads = config_hf_ref.num_attention_heads,                
    num_key_value_heads = config_hf_ref.num_key_value_heads,
    # Other
    tensor_parallelism=tensor_parallelism,
    pipeline_parallelism=pipeline_parallelism
    )
config = get_modified_config(model_choice, config)

# Model
pmap = ParallelDpPpEpTpMapper(
            data_parallelism=1, 
            pipeline_parallelism=pipeline_parallelism,
            expert_parallelism=expert_parallelism,
            tensor_parallelism=tensor_parallelism, 
            rank=rank, 
            create_groups=False)

with no_init_weights():
    model = LlamaParallelForCausalLM(config, pmap=pmap).to(torch.bfloat16)

# Copying weights from HF model to Optimus model
with torch.no_grad():
    model.set_parameters_from_full_module(model_hf)

# Save sharded model
parallel_model_dir_name = f"{model_choice.replace('/','__')}_pp{pipeline_parallelism}-ep{expert_parallelism}-tp{tensor_parallelism}-vpp1"

# project = "Aurora_deployment"
project = "Intel-Aurora"
parallel_models_dump_dir = f"/lus/flare/projects/{project}/dvooturi/parallel_models"
parallel_model_dir = os.path.join(parallel_models_dump_dir, parallel_model_dir_name)
state_dict_path = os.path.join(parallel_model_dir, f"model_{pmap.mp_ind}.pth")

# Save OM cached model
os.makedirs(parallel_model_dir, exist_ok=True)
torch.save(model.state_dict(), state_dict_path)


# python scripts/convert_hfcp_to_omcached.py
# bash launch_dist.sh 2 1 python scripts/convert_hfcp_to_omcached.py