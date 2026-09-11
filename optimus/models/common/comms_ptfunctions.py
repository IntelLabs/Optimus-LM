import time
import math

import torch

from ...profilers import record_pcl_function

"""
import os
save_tensors = False
tensor_dump_dir = "reference"
fwd_id = 0
bwd_id = 0
"""

# orig_allreduce = torch.distributed.all_reduce
# def all_reduce_wrapper(tensor, *args, **kwargs):
#     flat_tensor = tensor.view(-1)
#     splits = flat_tensor.split(128 * 1024 * 1024) # Limiting to 512 MB tensor size
#     for split in splits:
#         orig_allreduce(split, *args, **kwargs)

# torch.distributed.all_reduce = all_reduce_wrapper

# Tensor Parallelism
class forward_identity_backward_allreduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group=None):
        ctx.group = group
        return x
    
    @staticmethod
    def backward(ctx, grad):        
        """
        global bwd_id
        if save_tensors:
            rank = torch.distributed.get_rank()
            torch.save(grad, os.path.join(tensor_dump_dir, f"rank{rank}_tensor_bwd_{bwd_id}_before.pt"))
        """

        with record_pcl_function("2su-comms--2bwd--1jitter"):
            torch.xpu.synchronize()
            torch.distributed.barrier(group=ctx.group)

        with record_pcl_function("2su-comms--2bwd--2allreduce("+"x".join(str(dim) for dim in grad.shape)+")"):
            group = ctx.group
            torch.distributed.all_reduce(grad, group=group)

        """
        if save_tensors:            
            rank = torch.distributed.get_rank()
            torch.save(grad, os.path.join(tensor_dump_dir, f"rank{rank}_tensor_bwd_{bwd_id}_after.pt"))
            bwd_id += 1
        """

        return grad, None

class forward_allreduce_backward_identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group=None):
        """
        global fwd_id
        if save_tensors:
            rank = torch.distributed.get_rank()
            torch.save(x, os.path.join(tensor_dump_dir, f"rank{rank}_tensor_fwd_{fwd_id}_before.pt"))
        """

        with record_pcl_function("2su-comms--1fwd--1jitter"):
            torch.xpu.synchronize()
            torch.distributed.barrier(group=group)

        with record_pcl_function("2su-comms--1fwd--2("+"x".join(str(dim) for dim in x.shape)+")"):
            torch.distributed.all_reduce(x, group=group)

        """
        if save_tensors:
            rank = torch.distributed.get_rank()
            torch.save(x, os.path.join(tensor_dump_dir, f"rank{rank}_tensor_fwd_{fwd_id}_after.pt"))
            fwd_id += 1
        """
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None

class forward_inner_split_backward_allgather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pid=0, num_processes=1, group=None):
        size_p = x.shape[-1]//num_processes
        xp = x[...,pid*size_p:(pid+1)*size_p].contiguous()

        ctx.pid = pid
        ctx.num_processes = num_processes
        ctx.group = group
        ctx.size_p = size_p
        return xp
    
    @staticmethod
    def backward(ctx, grad):
        pid = ctx.pid
        num_processes = ctx.num_processes
        group = ctx.group
        size_p = ctx.size_p
        
        """
        x_grad_list = [torch.empty_like(grad) for _ in range(num_processes)]
        print(f"Rank {pid} - Allgathering gradients of shape {grad.shape} into list of size {len(x_grad_list)}", flush=True)
        torch.distributed.all_gather(x_grad_list, grad, group=group)
        # Concatenate the gradients from all processes
        x_grad = torch.cat(x_grad_list, dim=-1)
        """

        x_grad = torch.zeros((grad.shape[0], grad.shape[1], num_processes * size_p), dtype=grad.dtype, device=grad.device)
        x_grad[...,pid*size_p:(pid+1)*size_p].copy_(grad)
        
        """
        global bwd_id
        if save_tensors:
            rank = torch.distributed.get_rank()
            torch.save(x_grad, os.path.join(tensor_dump_dir, f"rank{rank}_tensor_bwd_{bwd_id}_before.pt"))
        """

        with record_pcl_function("2su-comms--2bwd--1jitter"):
            torch.xpu.synchronize()
            torch.distributed.barrier(group=group)

        with record_pcl_function("2su-comms--2bwd--2("+"x".join(str(dim) for dim in x_grad.shape)+")"):
            torch.distributed.all_reduce(x_grad, group=group)
        
        """
        if save_tensors:
            rank = torch.distributed.get_rank()
            torch.save(x_grad, os.path.join(tensor_dump_dir, f"rank{rank}_tensor_bwd_{bwd_id}_after.pt"))
            bwd_id += 1
        """

        return x_grad, None, None, None


# Expert Parallelism
class forward_allgather_backward_reducescatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pid=0, num_processes=1, group=None, use_allreduce_for_reducescatter=False):        
        xf_shape = (num_processes * x.shape[0],) + x.size()[1:]
        xf = torch.zeros(xf_shape, dtype=x.dtype, device=x.device)
        torch.distributed.all_gather_into_tensor(xf, x, group=group)
        
        ctx.pid = pid
        ctx.num_processes = num_processes
        ctx.group = group
        ctx.use_allreduce_for_reducescatter = use_allreduce_for_reducescatter
        return xf
    
    @staticmethod
    def backward(ctx, grad):
        pid = ctx.pid
        num_processes = ctx.num_processes
        group = ctx.group
        use_allreduce_for_reducescatter = ctx.use_allreduce_for_reducescatter
        
        if not use_allreduce_for_reducescatter:
            x_grad_shape = (grad.shape[0]//num_processes,) + grad.size()[1:]
            x_grad = torch.zeros(x_grad_shape, dtype=grad.dtype, device=grad.device)
            torch.distributed.reduce_scatter_tensor(x_grad, grad, group=group)
        else:
            torch.distributed.all_reduce(grad, group=group)
            size_p = grad.shape[0]//num_processes
            x_grad = grad[pid*size_p:(pid+1)*size_p]

        return x_grad, None, None, None, None

class forward_allgather_backward_split(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pid=0, num_processes=1, group=None):
        use_allreduce = False
        if not use_allreduce:
            xf_shape = (num_processes * x.shape[0],) + x.size()[1:]
            xf = torch.zeros(xf_shape, dtype=x.dtype, device=x.device)

            with record_pcl_function("2su-comms--1fwd--1jitter"):
                torch.xpu.synchronize()
                torch.distributed.barrier(group=group)

            with record_pcl_function("2su-comms--1fwd--2allgather("+"x".join(str(dim) for dim in xf.shape)+")"):
                torch.distributed.all_gather_into_tensor(xf, x, group=group)
        else:
            xf_shape = (num_processes * x.shape[0],) + x.size()[1:]
            xf = torch.zeros(xf_shape, dtype=x.dtype, device=x.device)
            xf[pid*x.shape[0]:(pid+1)*x.shape[0]].copy_(x)

            with record_pcl_function("2su-comms--1fwd--1jitter"):
                torch.xpu.synchronize()
                torch.distributed.barrier(group=group)

            with record_pcl_function("2su-comms--1fwd--2allreduce("+"x".join(str(dim) for dim in xf.shape)+")"):
                torch.distributed.all_reduce(xf, group=group)
        
        ctx.pid = pid
        ctx.num_processes = num_processes
        ctx.group = group
        return xf
    
    @staticmethod
    def backward(ctx, grad):
        pid = ctx.pid
        num_processes = ctx.num_processes
        group = ctx.group

        size_p = grad.shape[0]//num_processes
        x_grad = grad[pid*size_p:(pid+1)*size_p]
        return x_grad, None, None, None

class forward_reducescatter_backward_allgather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pid=0, num_processes=1, group=None, use_allreduce_for_reducescatter=False, use_allreduce_for_allgather=False):
        if not use_allreduce_for_reducescatter:
            xp_shape = (x.shape[0]//num_processes,) + x.size()[1:]
            xp = torch.zeros(xp_shape, dtype=x.dtype, device=x.device)

            with record_pcl_function("2su-comms--1fwd--1jitter"):
                torch.xpu.synchronize()
                torch.distributed.barrier(group=group)

            with record_pcl_function("2su-comms--1fwd--2reducescatter("+"x".join(str(dim) for dim in x.shape)+")"):
                torch.distributed.reduce_scatter_tensor(xp, x, group=group)
        else:
            with record_pcl_function("2su-comms--1fwd--1jitter"):
                torch.xpu.synchronize()
                torch.distributed.barrier(group=group)

            with record_pcl_function("2su-comms--1fwd--2allreduce("+"x".join(str(dim) for dim in x.shape)+")"):
                torch.distributed.all_reduce(x, group=group)
            x_d0 = x.shape[0]//num_processes
            xp = x[pid*x_d0:(pid+1)*x_d0]

        ctx.pid = pid
        ctx.num_processes = num_processes
        ctx.group = group
        ctx.use_allreduce_for_allgather = use_allreduce_for_allgather
        return xp
    
    @staticmethod
    def backward(ctx, grad):
        pid = ctx.pid
        num_processes = ctx.num_processes
        group = ctx.group
        use_allreduce_for_allgather = ctx.use_allreduce_for_allgather
        
        if not use_allreduce_for_allgather:
            x_grad_shape = (num_processes * grad.shape[0],) + grad.size()[1:]
            x_grad = torch.zeros(x_grad_shape, dtype=grad.dtype, device=grad.device)

            with record_pcl_function("2su-comms--2bwd--1jitter"):
                torch.xpu.synchronize()
                torch.distributed.barrier(group=group)

            with record_pcl_function("2su-comms--2bwd--2allgather("+"x".join(str(dim) for dim in x_grad.shape)+")"):
                torch.distributed.all_gather_into_tensor(x_grad, grad, group=group)
        else:
            x_grad_shape = (num_processes * grad.shape[0],) + grad.size()[1:]
            x_grad = torch.zeros(x_grad_shape, dtype=grad.dtype, device=grad.device)
            x_grad[pid*grad.shape[0]:(pid+1)*grad.shape[0]].copy_(grad)

            with record_pcl_function("2su-comms--2bwd--1jitter"):
                torch.xpu.synchronize()
                torch.distributed.barrier(group=group)

            with record_pcl_function("2su-comms--2bwd--2allreduce("+"x".join(str(dim) for dim in x_grad.shape)+")"):
                torch.distributed.all_reduce(x_grad, group=group)

        return x_grad, None, None, None, None, None


# Profiling
class forward_identity_backward_synchronize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group=None):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        with record_pcl_function("5moe-imbalance--2bwd"):
            torch.xpu.synchronize()
            torch.distributed.barrier(group=ctx.group)
        return grad, None

class forward_synchronize_backward_identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group=None):
        with record_pcl_function("5moe-imbalance--1fwd"):
            torch.xpu.synchronize()
            torch.distributed.barrier(group=group)
        return None

    @staticmethod
    def backward(ctx, grad):        
        return None


# Debugging
class forward_identity_backward_identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tag=""):
        ctx.tag = tag
        r = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()
        torch.save(x, f"tensors/{ctx.tag}_fwd_r{r}.pt")
        return x

    @staticmethod
    def backward(ctx, grad):
        r = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()
        torch.save(grad, f"tensors/{ctx.tag}_bwd_r{r}.pt")
        return grad, None


# Others
class forward_split_backward_allgather_pad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pid=0, num_processes=1, group=None):
        size = x.shape[0]
        size_p = math.ceil(size / num_processes)
        start_ind = pid * size_p
        end_ind = min((pid+1)*size_p, size)
        xp = x[start_ind:end_ind]

        ctx.pid = pid
        ctx.num_processes = num_processes
        ctx.group = group

        ctx.size = size
        ctx.size_p = size_p
        ctx.last_size_p = size_p
        ctx.is_last_split_partial = False
        if size % num_processes != 0:
            ctx.last_size_p = (end_ind - start_ind)
            ctx.is_last_split_partial = True

        return xp

    @staticmethod
    def backward(ctx, grad):
        if (ctx.pid == (ctx.num_processes - 1)) and ctx.is_last_split_partial:
            grad_padded_shape = (ctx.size_p,) + grad.size()[1:]
            grad_padded = torch.zeros(grad_padded_shape, dtype=grad.dtype, device=grad.device)
            grad_padded[:ctx.last_size_p].copy_(grad)
        else:
            grad_padded = grad

        # Full output
        x_grad_pad_shape = (ctx.size_p * ctx.num_processes,) + grad.size()[1:]
        x_grad_pad = torch.zeros(x_grad_pad_shape, dtype=grad.dtype, device=grad.device)
        torch.distributed.all_gather_into_tensor(x_grad_pad, grad_padded, group=ctx.group)
        x_grad = x_grad_pad[:ctx.size]

        return x_grad, None, None, None

class forward_allgather_backward_split_pad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xp, size, pid=0, num_processes=1, group=None):
        xp_pad = xp
        size_p = math.ceil(size / num_processes)
        if xp.shape[0] != size_p:
            x_pad_shape = (size_p,) + xp.size()[1:]
            xp_pad = torch.zeros(x_pad_shape, dtype=xp.dtype, device=xp.device)
            xp_pad[:xp.shape[0]].copy_(xp)

        x_pad_shape = (num_processes * size_p,) + xp.size()[1:]
        x_pad = torch.zeros(x_pad_shape, dtype=xp.dtype, device=xp.device)

        torch.distributed.all_gather_into_tensor(x_pad, xp_pad, group=group)
        x = x_pad[:size]

        ctx.pid = pid
        ctx.num_processes = num_processes
        ctx.group = group

        ctx.size = size
        ctx.size_p = size_p

        return x
    
    @staticmethod
    def backward(ctx, grad):
        start_ind = ctx.pid * ctx.size_p
        end_ind = min((ctx.pid+1)*ctx.size_p, ctx.size)
        xp_grad = grad[start_ind:end_ind]

        return xp_grad, None, None, None, None

class forward_split_backward_allgather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pid=0, num_processes=1, group=None):
        size_p = x.shape[0]//num_processes
        xp = x[pid*size_p:(pid+1)*size_p]

        ctx.pid = pid
        ctx.num_processes = num_processes
        ctx.group = group
        return xp
    
    @staticmethod
    def backward(ctx, grad):
        pid = ctx.pid
        num_processes = ctx.num_processes
        group = ctx.group
        
        x_grad_shape = (grad.shape[0]*num_processes,) + grad.size()[1:]
        x_grad = torch.zeros(x_grad_shape, dtype=grad.dtype, device=grad.device)
        torch.distributed.all_gather_into_tensor(x_grad, grad, group=group)

        return x_grad, None, None, None

class forward_allgather_backward_pick(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rank=0, world_size=1, group=None):
        x_list = [torch.empty_like(x) for i in range(world_size)]
        torch.distributed.all_gather(x_list, x, group=group)
        logits = torch.cat(x_list, -1)

        # Saving information for backward
        ctx.size_per_rank = x.shape[-1]
        ctx.rank = rank
        ctx.group = group
        return logits
    
    @staticmethod
    def backward(ctx, grad):
        batch_size, seq_length, _ = grad.shape
        size_per_rank = ctx.size_per_rank
        rank = ctx.rank
        group = ctx.group

        x_grad = grad[:,:,rank*size_per_rank:(rank+1)*size_per_rank].contiguous()
        return x_grad, None, None, None
