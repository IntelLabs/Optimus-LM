import math
import torch
from .dutils import print_rank_0


import torch
import torch.nn as nn
from contextlib import contextmanager

from typing import Optional, Union

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"

def device_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.xpu.is_available():
        torch.xpu.synchronize()
    else:
        print("Cannot synchronize")
        exit(-1)


def get_shape_string(tensor):
    return "(" + "x".join(str(dim) for dim in tensor.shape) + ")"

def get_pg_world_size(group):
    return len(torch.distributed.get_process_group_ranks(group))

def get_comms_profile_info_string(tensor, group):
    shape_str = get_shape_string(tensor)
    dtype_str = str(tensor.dtype).split(".")[1]
    ws_str = get_pg_world_size(group)
    return f"{shape_str}-{dtype_str}-ws{ws_str}"

###################################################################################################################
# TEMPORARY fix for "module load frameworks" huggingface issue
# Load balancing loss
def load_balancing_loss_func_4_56_1_mod(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, routing_weights.shape[1]))
            .reshape(-1, routing_weights.shape[1])
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    # device_index = routing_weights.device.index if routing_weights.device.index is not None else 0
    # rank = routing_weights.shape[1] * int(device_index)
    rank = 0
    overall_loss = torch.sum(
        tokens_per_expert[:, rank : rank + routing_weights.shape[1]] * router_prob_per_expert.unsqueeze(0)
    )
    return overall_loss * num_experts
###################################################################################################################

def get_load_balanced_expert_mask(sequence_length, num_experts, num_experts_per_tok, ep_ind=0):
    S = sequence_length
    E = num_experts
    C = num_experts_per_tok

    # Create tensor
    mask = torch.zeros((S, E), dtype=torch.bool)

    for s in range(S):
        se = (ep_ind*S + s) % E
        ee = se + (C-1)

        # Patch 1
        se1 = se
        ee1 = ee if ee < E else E-1
        mask[s, se1:(ee1+1)] = True

        # Patch 2
        if ee >= E:
            se2 = 0
            ee2 = ee - E
            mask[s, se2:(ee2+1)] = True

    return mask


@contextmanager
def no_init_weights():
    """
    Temporarily disable parameter initialization (reset_parameters)
    for all newly created nn.Modules inside this context.
    Useful for loading large pretrained checkpoints efficiently.
    """
    old_reset_parameters = {}

    def disable_reset_parameters(module):
        if hasattr(module, "reset_parameters"):
            old_reset_parameters[module] = module.reset_parameters
            module.reset_parameters = lambda *args, **kwargs: None

    # Patch the Module constructor to disable init for all new modules
    old_init = nn.Module.__init__

    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        disable_reset_parameters(self)

    nn.Module.__init__ = new_init

    try:
        yield
    finally:
        # Restore everything
        for module, old_fn in old_reset_parameters.items():
            module.reset_parameters = old_fn
        nn.Module.__init__ = old_init


def print_debug(string, enabled=True):
    # if torch.distributed.get_rank() != 1:
    #     return
    if enabled:
        print(string, flush=True)

def get_torch_dtype(dtype_str):
    assert dtype_str in ["float32", "bfloat16"], "Invalid dtype"
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "bfloat16":
        return torch.bfloat16

def get_layer_info_of_pipeline_stage(pp_ind, num_layers, pipeline_parallelism):
    ideal_num_layers_in_stage = math.ceil(num_layers / pipeline_parallelism)
    layer_start_id = pp_ind * ideal_num_layers_in_stage
    layer_end_id = min((pp_ind+1)*ideal_num_layers_in_stage, num_layers) - 1
    num_layers_in_stage = layer_end_id - layer_start_id + 1
    return layer_start_id, num_layers_in_stage


def is_xpu_available():
    return torch.xpu.is_available()

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:        
        if is_xpu_available():
            return "xpu"
        else:
            return "cpu"
    # elif torch.xpu.is_available():
    #     return "xpu"
    # else:
    #     return "cpu"

def get_dist_backend():
    import torch.distributed as dist
    if torch.cuda.is_available() and dist.is_nccl_available():
        return "nccl"
    elif is_xpu_available():
        return "xccl"
    else:
        print("Cannot choose backend")
        exit(-1)


def set_device(device):
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    elif torch.xpu.is_available():
        torch.xpu.set_device(device)

def device_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.xpu.is_available():
        torch.xpu.synchronize()
    else:
        print("Cannot synchronize")
        exit(-1)

def print_pytorch_memory_summary(info="", enabled=True):
    if enabled:
        print_rank_0(f"#{info}\n{torch.xpu.memory_summary()}")

def get_partition_lists(data, num_partitions):
    # Divide list of numbers(data) into P partitions.
    size = sum(data)
    ideal_partition_size = size // num_partitions # Ideal partition size

    # To return
    partition_lists = []
    for i in range(num_partitions):
        partition_lists.append([])

    cur_partition_size = 0
    cur_partition_id = 0
    for i,value in enumerate(data):
        cur_partition_size += value
        partition_lists[cur_partition_id].append(i)
        if cur_partition_size >= ideal_partition_size:
            cur_partition_id += 1
            cur_partition_size = 0

    return partition_lists

def get_model_and_optimizer_memory_usage(model, optimizer):
    # Model (weights and gradients)
    model_memory_in_bytes = 0
    for key,tensor in model.state_dict().items():
        model_memory_in_bytes += 2*(tensor.numel() * tensor.element_size())

    # Optimizer (States)
    optimizer_memory_in_bytes = 0
    if optimizer != None:
        for key,tensor in optimizer.state_dict().items():
            if (tensor != None) and (type(tensor) != int):
                optimizer_memory_in_bytes += (tensor.numel() * tensor.element_size())

    memory_in_bytes = (model_memory_in_bytes + optimizer_memory_in_bytes)

    # Memory in GB
    model_memory_in_GB = model_memory_in_bytes / (1024 * 1024 * 1024)
    optimizer_memory_in_GB = optimizer_memory_in_bytes / (1024 * 1024 * 1024)
    memory_in_GB = memory_in_bytes / (1024 * 1024 * 1024)

    return (model_memory_in_GB, optimizer_memory_in_GB, memory_in_GB)
    

"""
def compare_tensors(tensor, tensor_ref, tensor_name="Tensor", print_sample_size=10):
    print_string = ""
    print_string += "*"*30 + "\n"
    # Detach tensor
    tensor = tensor.detach() if isinstance(tensor, torch.Tensor) else tensor[0].detach()
    tensor_ref = tensor_ref[0].detach() if isinstance(tensor_ref, tuple) else tensor_ref.detach()
    if tensor_name is not None:
        print_string += f"Error information for {tensor_name}" + "\n"
    error = torch.abs(tensor - tensor_ref)
    norm = torch.norm(error)
    avg_error = torch.sum(error)/error.numel()
    print_string += f"Average error : {avg_error}" + "\n"
    print_string += f"Norm : {norm}" + "\n"
    if print_sample_size > 0:
        flat_tensor = tensor.flatten()
        flat_tensor_ref = tensor_ref.flatten()
        for i in range(print_sample_size):
            print_string += f"{flat_tensor[i].item()}, {flat_tensor_ref[i].item()}" + "\n"
    print_string += "*"*30 + "\n"

    print(print_string, flush=True)
"""


def print_error_info(tensor, tensor_ref, name=None, print_sample=True, sort=False):
    print_str = "*"*40 + "\n"
    tensor = tensor.detach().clone() if isinstance(tensor, torch.Tensor) else tensor[0].detach().clone()
    tensor_ref = tensor_ref.detach().clone() if isinstance(tensor_ref, torch.Tensor) else tensor_ref[0].detach().clone()
    
    if name != None:
        print_str += f"Error information for ({name})\n"
    # Converting them into arrays
    tensor = tensor.flatten()
    tensor_ref = tensor_ref.flatten()

    error_tensor = torch.abs(tensor.to(torch.float32) - tensor_ref.to(torch.float32))
    norm = torch.norm(error_tensor)
    min_error_val = torch.min(error_tensor).item()
    max_error_val = torch.max(error_tensor).item()
    avg_error_val = torch.mean(error_tensor)

    tensor_norm = torch.norm(tensor)
    tensor_ref_norm = torch.norm(tensor_ref)

    print_str += f"Obs tensor norm : {tensor_norm:10g}\n"
    print_str += f"Ref tensor norm : {tensor_ref_norm:10g}\n"
    print_str += f"Err tensor norm : {norm:10g}\n"
    print_str += f"Min error value : {min_error_val:10g}\n"
    print_str += f"Max error value : {max_error_val:10g}\n"
    print_str += f"Avg error value : {avg_error_val:10g}\n"

    diff_count = torch.sum(error_tensor > 0.0).item()
    print_str += f"Diff count      : {diff_count} / {tensor.numel()} ({(diff_count / tensor.numel() * 100):.2f} %)\n"

    # Sort the values based on the reference tensor
    if sort:
        indices = torch.argsort(error_tensor, descending=True)
    else:
        indices = torch.arange(tensor.numel(), device=tensor.device)
    
    if print_sample:
      num_samples = min(tensor.numel(), 8)
      print_str += f"First {num_samples} values\n"
      for i in range(num_samples):
          print_str += f"{indices[i]}, {tensor[indices[i]].item():10g}, {tensor_ref[indices[i]].item():10g}\n"
    print_str += "*"*40 + "\n"
    print(print_str, flush=True)


import functools
import types
import typing
import warnings
from typing import cast, Optional, Union
from typing_extensions import deprecated

import torch
from torch import Tensor
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

_tensor_or_tensors = Union[
    torch.Tensor,
    typing.Iterable[torch.Tensor],  # noqa: UP006 - needed until XLA's patch is updated
]


def _no_grad(func):
    """
    This wrapper is needed to avoid a circular import when using @torch.no_grad on the exposed functions
    clip_grad_norm_ and clip_grad_value_ themselves.
    """

    def _no_grad_wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    functools.update_wrapper(_no_grad_wrapper, func)
    return _no_grad_wrapper

@_no_grad
def get_total_norm(
    tensors: _tensor_or_tensors,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    r"""Compute the norm of an iterable of tensors.

    The norm is computed over the norms of the individual tensors, as if the norms of
    the individual tensors were concatenated into a single vector.

    Args:
        tensors (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will be normalized
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of :attr:`tensors` is ``nan``, ``inf``, or ``-inf``.
            Default: ``False``
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the tensors (viewed as a single vector).
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)
    first_device = tensors[0].device
    grouped_tensors: dict[
        tuple[torch.device, torch.dtype], tuple[list[list[Tensor]], list[int]]
    ] = _group_tensors_by_device_and_dtype(
        [tensors]  # type: ignore[list-item]
    )  # type: ignore[assignment]

    norms: list[Tensor] = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend(
                [torch.linalg.vector_norm(g, norm_type) for g in device_tensors]
            )

    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm

def dump_weight_gradients(model, config, pmap, opt_key=None, prefix=None):
        from collections import OrderedDict
        grad_dict = OrderedDict()
        for name,param in model.named_parameters():
            name_mod = name
            if "experts" in name:
                tokens = name.split(".")
                tokens[5] = str(pmap.ep_ind * (config.num_experts // pmap.expert_parallelism) + int(tokens[5]))
                name_mod = ".".join(tokens)
            if param.grad is not None:
                grad_dict[name_mod] = param.grad
            
        grad_name = f"grads_dp{pmap.data_parallelism}-ep{pmap.expert_parallelism}-tp{pmap.tensor_parallelism}_rank{pmap.rank}"
        if opt_key != None:
            grad_name = f"{opt_key}_" + grad_name
        if prefix != None:
            grad_name = f"{prefix}_" + grad_name
        grad_path = f"weight_grads/{grad_name}.pt"
        
        torch.save(grad_dict, grad_path)

def save_tensor(tensor, name):
    tensor_path = f"tensors/{name}.pt"
    torch.save(tensor, tensor_path)