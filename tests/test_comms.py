import os
import sys

import torch
# import intel_extension_for_pytorch 
# import oneccl_bindings_for_pytorch 

# Setting seed for reproducibility
torch.manual_seed(0)

# Distributed step
dist_backend = "xccl"
rank, world_size, local_rank, local_world_size = 0,1,0,1
if int(os.getenv("PMI_SIZE", "1")) > 1:
    os.environ['RANK'] = os.getenv("PALS_RANKID", "0")
    os.environ['WORLD_SIZE'] = os.getenv("PMI_SIZE", "1")
    os.environ['LOCAL_RANK'] = os.getenv("PALS_LOCAL_RANKID", "0")
    os.environ['LOCAL_WORLD_SIZE'] = os.getenv("PALS_LOCAL_SIZE", "1")

    local_rank = int(os.getenv("LOCAL_RANK"))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE"))

    # Initialize process group
    torch.distributed.init_process_group(dist_backend, device_id=local_rank)

    # Get variables
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.xpu.set_device(local_rank) # Setting device rank

# Arguments
size = (128 * 1024 * 1024)
dtype = torch.float32
device = "xpu"

# Input
tensor = torch.ones(size, dtype=dtype, device=device) + rank # Each rank's array has RANK+1 value.

if rank == 0:
    print_str = "*************** Arguments *******************\n"
    print_str += f"Size               : {size}\n"
    print_str += f"Dtype              : {dtype}\n"
    print_str += f"Memory             : {(size * tensor.element_size() / (1024 * 1024))} MB\n"
    print_str += f"Dist Backend       : {dist_backend}\n"
    print_str += f"World Size         : {world_size}\n"
    print_str += f"Local World Size   : {local_world_size}\n"
    print_str += f"*******************************************\n"
    print(print_str, flush=True)


# Allreduce operation
torch.distributed.all_reduce(tensor)
torch.distributed.barrier()

if rank == 0:
    ref_val = (world_size * (world_size + 1))/2
    if (ref_val == tensor).all():
        print("SUCCESS")
    else:
        print("FAILED")

## Single node
# bash launch_dist.sh 2 1 python tests/test_comms.py
# bash launch_dist.sh 12 1 python tests/test_comms.py

## Multi node
# bash launch_dist.sh 1 2 python tests/test_comms.py
# bash launch_dist.sh 12 2 python tests/test_comms.py

