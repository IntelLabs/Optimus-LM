import os

import torch

def print_rank_0(string, flush=True, skip=False):
    if skip:
        return
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(string, flush=flush)
    else:
        print(string, flush=flush)


def setup_xpu_distributed(dist_backend="ccl"):
    if dist_backend == "xccl":
        pass
    elif dist_backend == "ccl":
        import oneccl_bindings_for_pytorch
    elif dist_backend == "mpi":
        import oneccl_bindings_for_pytorch
        import torch_mpi
        torch.distributed.Backend.backend_capability['mpi'].append('xpu')
    else:
        print(f"Unsupported distributed backend {dist_backend} for XPU")
        exit(-1)

    assert dist_backend in ["xccl", "ccl", "mpi"], f"Unsupported distributed backend {dist_backend} for XPU"
    
    os.environ['RANK'] = os.getenv("PALS_RANKID", "0")
    os.environ['WORLD_SIZE'] = os.getenv("PMI_SIZE", "1")
    os.environ['LOCAL_RANK'] = os.getenv("PALS_LOCAL_RANKID", "0")
    os.environ['LOCAL_WORLD_SIZE'] = os.getenv("PALS_LOCAL_SIZE", "1")
    os.environ['MPI_LOCALRANKID'] = os.environ['LOCAL_RANK']
    
    local_rank = int(os.getenv("LOCAL_RANK"))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE"))

    if dist_backend == "xccl":
        torch.distributed.init_process_group(dist_backend, device_id=local_rank)
    else:
        torch.distributed.init_process_group(dist_backend)

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    torch.xpu.set_device(local_rank) # Setting device rank

    return rank, world_size, local_rank, local_world_size


def setup_distributed(dist_backend="ccl", local_rank_to_device_map=None, num_ranks_per_gpu=1):
    assert ((local_rank_to_device_map != None) and (num_ranks_per_gpu > 1)) == False
    if dist_backend == "nccl":
        # print("Using NCCL distributed backend")
        pass
    elif dist_backend == "ccl":
        # print("Using CCL distributed backend")
        if torch.cuda.is_available():
            assert False, "ccl distributed backend cannoted be used on NV GPU"
        else:
            import oneccl_bindings_for_pytorch
    elif dist_backend == "mpi":
        # print("Using MPI distributed backend")
        if torch.cuda.is_available():
            import torch_mpi
        else:
            import intel_extension_for_pytorch
            import oneccl_bindings_for_pytorch
            import torch_mpi
            torch.distributed.Backend.backend_capability['mpi'].append('xpu')
    elif dist_backend == "gloo":
        pass
    else:
        print(f"Unsupported distributed backend {dist_backend}")
        exit(-1)

    def env_to_int(keys, default=None):
        value = default
        for key in keys:
            if key in os.environ:
                value = int(os.environ[key])
                break
        return value

    # Setting up distributed
    os.environ['RANK'] = str(env_to_int(["RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", "PALS_RANKID"], 0))
    os.environ['WORLD_SIZE'] = str(env_to_int(["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], 1))
    local_rank = env_to_int(
        ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK", "PALS_LOCAL_RANKID", "TPI_LOCALRANKID"], 0
    )
    local_world_size = env_to_int(
        ["MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE", "PALS_LOCAL_SIZE", "TPI_LOCALNRANKS"], 1
    )
    
    os.environ['MPI_LOCALRANKID'] = str(local_rank)
    torch.distributed.init_process_group(dist_backend)

    # Get variables
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    # Setting device
    from .utils import get_device, set_device
    if local_rank_to_device_map == None:
        if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ['CUDA_VISIBLE_DEVICES'].startswith("MIG"):
            device = torch.device(get_device(), 0)
            assert num_ranks_per_gpu == 1
        else:
            device = torch.device(get_device(), local_rank//num_ranks_per_gpu)
    else:
        device = torch.device(get_device(), local_rank_to_device_map[local_rank])
    set_device(device)

    # print(f"Rank {rank} :: world_size = {world_size} local_rank = {local_rank} local_world_size = {local_world_size}, device = {device}", flush=True)

    return rank, world_size, local_rank, local_world_size, device