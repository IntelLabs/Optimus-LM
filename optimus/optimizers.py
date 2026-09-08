import os
import math

import torch
from torch.optim import Optimizer

from optimus.profilers import record_pcl_function
from optimus.utils import dump_weight_gradients
from optimus.utils import get_comms_profile_info_string

def chunked_allreduce_wrapper(tensor, group, barrier=False):
    flat_tensor_parts = tensor.split(128 * 1024 * 1024) # Limiting to 512 MB tensor size
    for flat_tensor_part in flat_tensor_parts:
        with record_pcl_function("3so-comms--1allreduce--" + get_comms_profile_info_string(flat_tensor_part, group)):
            torch.distributed.all_reduce(flat_tensor_part, group=group)
            if barrier:
                torch.distributed.barrier(group=group)

class MixedPrecisionAdamW(Optimizer):
    """
    Stores all parameters, gradients, and optimizer states into a flat buffer.
    Optimizer states include exp_avg and exp_avg_sq which are in FP32
    Master weights are stored in FP32
    ## Limitations
    1. Only one param group
    """
    def __init__(self, 
        params, model,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01, warmup_steps=0,
        dtype=torch.float32, device=None,
        num_shards=1, shard_id=0, group=None,
        opt_disable_delayed_grad_clipping=False,
        opt_use_two_step_grad_clipping=False,
        opt_use_chunked_allreduce=False,
        profiler=None,
        tensor_parallelism=1, tp_ind=0, tp_group=None,
        expert_parallelism=1, ep_ind=0, ep_group=None,
        dpep_ind=0, dpep_group=None,
        skip_optimizer_step=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        assert device != None

        self.model = model
        self.warmup_steps = warmup_steps
        self.dtype = dtype
        self.device = device
        self.opt_disable_delayed_grad_clipping = opt_disable_delayed_grad_clipping
        self.opt_use_two_step_grad_clipping = opt_use_two_step_grad_clipping
        self.opt_use_chunked_allreduce = opt_use_chunked_allreduce
        self.profiler = profiler

        self.tensor_parallelism = tensor_parallelism
        self.tp_ind = tp_ind
        self.tp_group = tp_group

        self.expert_parallelism = expert_parallelism
        self.ep_ind = ep_ind
        self.ep_group = ep_group

        assert self.expert_parallelism == 1, "This optimizer does not support expert parallelism."

        self.dpep_ind = dpep_ind
        self.dpep_group = dpep_group
        self.skip_optimizer_step = skip_optimizer_step

        self.step_id = 0

        # Sharding options
        self.num_shards = num_shards
        self.dp_ind = shard_id
        self.group = group

        self.param_list = [p for pg in self.param_groups for p in pg["params"]]
        assert len([pg for pg in self.param_groups]) == 1, "Only one parameter group is supported"

        self.param_sizes = [p.numel() for p in self.param_list]
        self.param_start_inds = [sum(self.param_sizes[0:i]) for i in range(len(self.param_sizes))]
        self.flat_param_size = sum(self.param_sizes)

        # Flat buffers
        self.flat_param = torch.empty(self.flat_param_size, dtype=self.dtype, device=self.device)
        self.flat_grad = torch.zeros(self.flat_param_size, dtype=self.dtype, device=self.device)
        if not self.skip_optimizer_step:
            self.flat_exp_avg = torch.zeros(self.flat_param_size, dtype=torch.float32, device=self.device)
            self.flat_exp_avg_sq = torch.zeros(self.flat_param_size, dtype=torch.float32, device=self.device)
        else:
            self.flat_exp_avg = None
            self.flat_exp_avg_sq = None
        
        if self.dtype != torch.float32:
            if not self.skip_optimizer_step:
                self.flat_fp32_master_param = torch.zeros(self.flat_param_size, dtype=torch.float32, device=self.device)
            else:
                self.flat_fp32_master_param = None
        else:
            # In case of FP32 optimizer, we can use the same buffer for master param and model param to save memory
            self.flat_fp32_master_param = self.flat_param
        
        # Mapping parameter into flat parameter.
        for param_id, p in enumerate(self.param_list):
            start_ind = self.param_start_inds[param_id]
            end_ind = self.param_start_inds[param_id] + self.param_sizes[param_id]
            # Parameter
            p.data = self.flat_param[start_ind:end_ind].view_as(p.data).copy_(p.data) # Override
            # Gradient
            p.grad = self.flat_grad[start_ind:end_ind].view_as(p.data)

            # Master parameter
            if self.dtype != torch.float32 and (not self.skip_optimizer_step):
                self.flat_fp32_master_param[start_ind:end_ind].view_as(p.data).copy_(p.data)

    def zero_grad(self):
        self.flat_grad.zero_()

    def step(self, closure=None):
        with record_pcl_function("3dp-comms-1grad_acc-block"):
            if torch.distributed.is_initialized() and self.num_shards > 1:
                if self.opt_use_chunked_allreduce:
                    # Do it in parts to avoid large parallel allreduce bug (https://jira.devtools.intel.com/browse/MLSL-3776)
                    flat_grad_parts = self.flat_grad.split(128 * 1024 * 1024) # Limiting to 512 MB tensor size
                    for flat_grad_part in flat_grad_parts:
                        with record_pcl_function("3dp-comms--1grad_acc--part"):
                            torch.distributed.all_reduce(flat_grad_part, group=self.group)
                else:
                    # Bug https://jira.devtools.intel.com/browse/MLSL-3776 is fixed now
                    with record_pcl_function("3dp-comms--1grad_acc"):
                        torch.distributed.all_reduce(self.flat_grad, group=self.group)
                # Average the gradients
                self.flat_grad.div_(self.num_shards)
         
        with record_pcl_function("4opt-step_grad-clip"):
            # Do gradient clipping only after warmup steps
            if self.opt_disable_delayed_grad_clipping or \
                ((self.opt_disable_delayed_grad_clipping == False) and (self.step_id >= self.warmup_steps)):

                if not self.opt_use_two_step_grad_clipping:
                    # Single step gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    if self.tensor_parallelism > 1 and (self.dp_ind == 0 and self.tp_ind == 0) and (self.step_id == 0):
                        print(f"Warning : Using local gradient clipping", flush=True)
                else:
                    if self.tensor_parallelism == 1:
                        # Two step gradient clipping (Serial)
                        grad_norm = torch.nn.utils.get_total_norm([p.grad for p in self.model.parameters() if p.grad is not None])
                        torch.nn.utils.clip_grads_with_norm_(self.model.parameters(), 1.0, grad_norm)
                    else:
                        # Two step gradient clipping (Parallel)
                        grad_norm = self.model.calculate_grad_norm()
                        torch.nn.utils.clip_grads_with_norm_(self.model.parameters(), 1.0, grad_norm)
            else:
                # Calculate gradient norm even when gradient clipping is not applicable in this step.            
                if self.tensor_parallelism == 1:
                    grad_norm = torch.nn.utils.get_total_norm([p.grad for p in self.model.parameters() if p.grad is not None])
                else:
                    grad_norm = self.model.calculate_grad_norm()

        with record_pcl_function("4opt-step_compute"):
            # Hyper parameters
            beta1, beta2 = self.defaults["betas"]
            eps = self.defaults["eps"]
            lr = self.param_groups[0]["lr"]
            step_size = lr
            weight_decay = self.defaults["weight_decay"]

            # Tensors
            size = self.flat_param_size # Number of parameters
            data = self.flat_param # Weights
            grad = self.flat_grad  # Gradients
            data_master = self.flat_fp32_master_param # OS master weights
            exp_avg = self.flat_exp_avg # OS moment1
            exp_avg_sq = self.flat_exp_avg_sq # OS moment2

            # Doing in chunks to reduce memory requirement
            chunk_size = 128 * 1024 * 1024
            num_chunks = math.ceil(size / chunk_size)
            tensor_list = [data, grad, data_master, exp_avg, exp_avg_sq]
            for chunk_id in range(num_chunks):
                start_ind = chunk_id * chunk_size
                end_ind = min((chunk_id+1)*chunk_size, size)

                # Getting chunks
                data_chunk, grad_chunk, data_master_chunk, exp_avg_chunk, exp_avg_sq_chunk = [tensor[start_ind:end_ind] for tensor in tensor_list]

                # Using FP32 gradient for optimizer step
                grad_chunk_fp32 = grad_chunk.to(torch.float32)

                # Optimizer compute
                exp_avg_chunk.mul_(beta1).add_(grad_chunk_fp32, alpha=(1.0 - beta1))
                exp_avg_sq_chunk.mul_(beta2).addcmul_(grad_chunk_fp32, grad_chunk_fp32, value=1.0 - beta2)
                exp_avg_hat_chunk = exp_avg_chunk / (1.0 - math.pow(beta1, (self.step_id+1))) 
                exp_avg_sq_hat_chunk = exp_avg_sq_chunk / (1.0 - math.pow(beta2, (self.step_id+1)))
                denom_chunk = exp_avg_sq_hat_chunk.sqrt().add_(eps)
                data_master_chunk.addcdiv_(exp_avg_hat_chunk, denom_chunk, value=-step_size)

                # Handling weight decay
                if weight_decay > 0.0:
                    data_master_chunk.add_(data_master_chunk, alpha=(-lr * weight_decay)) #[data = data - (lr * weight_decay)]

                # Convert master weights to model weights dtype
                data_chunk.copy_(data_master_chunk.to(data_chunk.dtype))
        # Update step
        self.step_id += 1

        return grad_norm
    
    def state_dict(self):
        opt_state_dict = {
            "flat_exp_avg" : self.flat_exp_avg,
            "flat_exp_avg_sq" : self.flat_exp_avg_sq,
            "flat_fp32_master_param" : self.flat_fp32_master_param,
            "step_id" : self.step_id
        }

        return opt_state_dict
    
    def load_state_dict(self, opt_state_dict):
        self.flat_exp_avg.copy_(opt_state_dict["flat_exp_avg"].to(self.device))
        self.flat_exp_avg_sq.copy_(opt_state_dict["flat_exp_avg_sq"].to(self.device))
        self.flat_fp32_master_param.copy_(opt_state_dict["flat_fp32_master_param"].to(self.device))
        self.step_id = opt_state_dict["step_id"]

class ShardedMixedPrecisionAdamW(Optimizer):
    """
    Stores all parameters, gradients, and optimizer states into a flat buffer.
    Optimizer states include exp_avg and exp_avg_sq which are in FP32
    Master weights are stored in FP32
    Master weights and optimizer states are sharded across multiple GPUs
    ## Limitations
    1. Only one param group
    """
    def __init__(self, 
        params, model,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01, warmup_steps=0,
        dtype=torch.float32, device=None, 
        num_shards=1, shard_id=0, group=None,
        opt_disable_delayed_grad_clipping=False,
        profiler=None,
        tensor_parallelism=1, tp_ind=0, tp_group=None,
        expert_parallelism=1, ep_ind=0, ep_group=None,
        dpep_ind=0, dpep_group=None,
        shard_divisible_by=512, 
        opt_use_chunked_allreduce=False,
        opt_use_allreduce_for_grad_acc=False, 
        opt_use_allreduce_for_param_gather=False,
        skip_optimizer_step=False,
        disable_optimizer_state_checkpointing=False,
        skip_optimizer_step_weight_update=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.opt_key = "shard"

        assert device != None

        self.model = model
        self.warmup_steps = warmup_steps
        self.dtype = dtype
        self.device = device
        self.opt_disable_delayed_grad_clipping = opt_disable_delayed_grad_clipping
        self.profiler = profiler

        self.tensor_parallelism = tensor_parallelism
        self.tp_ind = tp_ind
        self.tp_group = tp_group

        self.expert_parallelism = expert_parallelism
        self.ep_ind = ep_ind
        self.ep_group = ep_group

        self.step_id = 0

        # Sharding options
        self.data_parallelism = num_shards
        self.dp_ind = shard_id
        self.dp_group = group
        
        self.shard_divisible_by = shard_divisible_by
        self.opt_use_chunked_allreduce = opt_use_chunked_allreduce
        self.opt_use_allreduce_for_grad_acc = opt_use_allreduce_for_grad_acc
        self.opt_use_allreduce_for_param_gather = opt_use_allreduce_for_param_gather

        self.param_list = [p for pg in self.param_groups for p in pg["params"]]
        assert len([pg for pg in self.param_groups]) == 1, "Only one parameter group is supported"

        self.param_sizes = [p.numel() for p in self.param_list]
        self.param_start_inds = [sum(self.param_sizes[0:i]) for i in range(len(self.param_sizes))]
        self.flat_param_size = sum(self.param_sizes)

        self.shard_size =  (math.ceil(self.flat_param_size / (self.data_parallelism * self.shard_divisible_by))) * self.shard_divisible_by
        self.flat_buffer_size = self.data_parallelism * self.shard_size

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            ideal_shard_size = (self.flat_param_size // self.data_parallelism)
            info_str = "********************************************************************************\n"
            info_str += "Total number of parameters : {}\n".format(self.flat_param_size)
            info_str += "Number of shards           : {}\n".format(self.data_parallelism)
            info_str += "Ideal shard size           : {}\n".format(ideal_shard_size)
            info_str += "Actual Shard size          : {} (Padding : {})\n".format(self.shard_size, self.shard_size - ideal_shard_size)
            info_str += "Padding percentage         : {:.3f} %\n".format((1.0 - ideal_shard_size/self.shard_size) * 100)
            info_str += "Flat parameter size        : {}\n".format(self.flat_buffer_size)
            info_str += "********************************************************************************"
            print(info_str, flush=True)
            
        # Flat buffers
        self.flat_param = torch.empty(self.flat_buffer_size, dtype=dtype, device=device)
        self.flat_grad = torch.zeros(self.flat_buffer_size, dtype=self.dtype, device=self.device)
        if skip_optimizer_step == False:
            self.flat_exp_avg_shard = torch.zeros(self.shard_size, dtype=torch.float32, device=self.device)
            self.flat_exp_avg_sq_shard = torch.zeros(self.shard_size, dtype=torch.float32, device=self.device)
            self.flat_fp32_master_param_shard = torch.zeros(self.shard_size, dtype=torch.float32, device=self.device)
        else:
            self.flat_exp_avg_shard = None
            self.flat_exp_avg_sq_shard = None
            self.flat_fp32_master_param_shard = None

        # Current shard view (Useful for reduce_scatter and all_gather_into_tensor comms)
        self.flat_param_shard = self.flat_param[self.dp_ind*self.shard_size:(self.dp_ind+1)*self.shard_size]
        self.flat_grad_shard = self.flat_grad[self.dp_ind*self.shard_size:(self.dp_ind+1)*self.shard_size]

        # Mapping parameter into flat parameter.
        for param_id, p in enumerate(self.param_list):
            start_ind = self.param_start_inds[param_id]
            end_ind = self.param_start_inds[param_id] + self.param_sizes[param_id]
            # Parameter
            p.data = self.flat_param[start_ind:end_ind].view_as(p.data).copy_(p.data) # Override
            # Gradient
            p.grad = self.flat_grad[start_ind:end_ind].view_as(p.data)

        # Setting master parameter
        if (not skip_optimizer_step):
            self.flat_fp32_master_param_shard.copy_(self.flat_param_shard)

        self.disable_optimizer_state_checkpointing = disable_optimizer_state_checkpointing
        self.skip_optimizer_step_weight_update = skip_optimizer_step_weight_update

    def zero_grad(self):
        self.flat_grad.zero_()

    def step(self, closure=None):
        if self.dump_weight_gradients:
            dump_weight_gradients(self.model, self.config, self.pmap, self.opt_key, "beforegradacc")

        # If it is distributed, accumulate gradients
        with record_pcl_function("4opt-step_1gradacc"):
            # EP grad_acc
            if torch.distributed.is_initialized() and self.expert_parallelism > 1:
                replicated_param_list = self.model.get_ep_replicated_param_list()
                for name, param in self.model.named_parameters():
                    if name in replicated_param_list:
                        torch.distributed.all_reduce(param.grad, group=self.ep_group)
                    param.grad.div_(self.expert_parallelism)

            # DP grad_acc
            if torch.distributed.is_initialized() and self.data_parallelism > 1:
                if not self.opt_use_allreduce_for_grad_acc:
                    with record_pcl_function("3so-comms--2reduce_scatter--" + get_comms_profile_info_string(self.flat_grad, self.dp_group)):
                        torch.distributed.reduce_scatter_tensor(self.flat_grad_shard, self.flat_grad, group=self.dp_group)
                    # Average the gradients
                    self.flat_grad_shard.div_(self.data_parallelism)
                else:
                    if not self.opt_use_chunked_allreduce:
                        with record_pcl_function("3so-comms--1allreduce--" + get_comms_profile_info_string(self.flat_grad, self.dp_group)):
                            torch.distributed.all_reduce(self.flat_grad, group=self.dp_group)
                    else:
                        chunked_allreduce_wrapper(self.flat_grad, self.dp_group)
                    # Average the gradients
                    self.flat_grad.div_(self.data_parallelism)

            # print(f"Rank : {torch.distributed.get_rank()}, Grad norm : {torch.nn.utils.get_total_norm(self.flat_grad)}", flush=True)

        if self.dump_weight_gradients:
            dump_weight_gradients(self.model, self.config, self.pmap, self.opt_key, "aftergradacc")

        # Global gradient norm
        with record_pcl_function("4opt-step_3global-gradnorm"):
            if self.expert_parallelism > 1:
                # Scale down EP replicated parameters gradients
                ep_replicated_param_list = self.model.get_ep_replicated_param_list()
                for name, param in self.model.named_parameters():
                    if name in ep_replicated_param_list:
                        param.grad.data.div_(math.sqrt(self.expert_parallelism))
            
            if self.tensor_parallelism > 1:
                # Scale down TP replicated parameters gradients
                tp_replicated_param_list = self.model.get_tp_replicated_param_list()
                for name, param in self.model.named_parameters():
                    if name in tp_replicated_param_list:
                        param.grad.data.div_(math.sqrt(self.tensor_parallelism))

            grad_shard_norm = torch.nn.utils.get_total_norm(self.flat_grad_shard)
            if torch.distributed.is_initialized():
                grad_shard_norm_fp32 = grad_shard_norm.to(torch.float32)
                grad_shard_norm_sq_fp32 = torch.pow(grad_shard_norm_fp32, 2)
                torch.distributed.all_reduce(grad_shard_norm_sq_fp32)
                grad_norm = torch.sqrt(grad_shard_norm_sq_fp32)
            else:
                grad_norm = grad_shard_norm

            if self.expert_parallelism > 1:
                # Undo scale down
                ep_replicated_param_list = self.model.get_ep_replicated_param_list()
                for name, param in self.model.named_parameters():
                    if name in ep_replicated_param_list:
                        param.grad.data.mul_(math.sqrt(self.expert_parallelism))
            
            if self.tensor_parallelism > 1:
                # Undo scale down
                tp_replicated_param_list = self.model.get_tp_replicated_param_list()
                for name, param in self.model.named_parameters():
                    if name in tp_replicated_param_list:
                        param.grad.data.mul_(math.sqrt(self.tensor_parallelism))
            
        # Gradient clipping
        with record_pcl_function("4opt-step_4clip-grads"):
            if self.opt_disable_delayed_grad_clipping or \
                ((self.opt_disable_delayed_grad_clipping == False) and (self.step_id >= self.warmup_steps)):
                max_norm = 1.0
                use_custom_grad_clipping = True
                if not use_custom_grad_clipping:
                    torch.nn.utils.clip_grads_with_norm_(self.model.parameters(), max_norm, grad_norm)
                else:
                    clip_coef = max_norm / (grad_norm + 1e-6)  # avoid divide-by-zero
                    if clip_coef < 1.0:
                        self.flat_grad_shard.mul_(clip_coef)  # In-place scaling

        with record_pcl_function("4opt-step_5compute"):
            if not self.skip_optimizer_step_weight_update:
                # Hyper parameters
                beta1, beta2 = self.defaults["betas"]
                eps = self.defaults["eps"]
                lr = self.param_groups[0]["lr"]
                step_size = lr
                weight_decay = self.defaults["weight_decay"]

                # Tensors
                size = self.shard_size # Sum of sizes of parameters
                data = self.flat_param_shard # Weights
                grad = self.flat_grad_shard # Gradients
                data_master = self.flat_fp32_master_param_shard # OS master weights
                exp_avg = self.flat_exp_avg_shard # OS moment1
                exp_avg_sq = self.flat_exp_avg_sq_shard # OS moment2

                # Doing in chunks to reduce memory requirement
                chunk_size = 128 * 1024 * 1024
                num_chunks = math.ceil(size / chunk_size)
                tensor_list = [data, grad, data_master, exp_avg, exp_avg_sq]
                for chunk_id in range(num_chunks):
                    start_ind = chunk_id * chunk_size
                    end_ind = min((chunk_id+1)*chunk_size, size)

                    # Getting chunks
                    data_chunk, grad_chunk, data_master_chunk, exp_avg_chunk, exp_avg_sq_chunk = [tensor[start_ind:end_ind] for tensor in tensor_list]

                    # Using FP32 gradient for optimizer step
                    grad_chunk_fp32 = grad_chunk.to(torch.float32)

                    # Optimizer compute
                    exp_avg_chunk.mul_(beta1).add_(grad_chunk_fp32, alpha=(1.0 - beta1))
                    exp_avg_sq_chunk.mul_(beta2).addcmul_(grad_chunk_fp32, grad_chunk_fp32, value=1.0 - beta2)
                    exp_avg_hat_chunk = exp_avg_chunk / (1.0 - math.pow(beta1, (self.step_id+1))) 
                    exp_avg_sq_hat_chunk = exp_avg_sq_chunk / (1.0 - math.pow(beta2, (self.step_id+1)))
                    denom_chunk = exp_avg_sq_hat_chunk.sqrt().add_(eps)
                    data_master_chunk.addcdiv_(exp_avg_hat_chunk, denom_chunk, value=-step_size)

                    # Handling weight decay
                    if weight_decay > 0.0:
                        data_master_chunk.add_(data_master_chunk, alpha=(-lr * weight_decay)) #[data = data - (lr * weight_decay)]

                    # Convert master weights to model weights dtype
                    if not self.opt_use_allreduce_for_param_gather:
                        data_chunk.copy_(data_master_chunk)

        if not self.opt_use_allreduce_for_param_gather:
            if torch.distributed.is_initialized() and self.data_parallelism > 1:
                with record_pcl_function("3so-comms--3allgather--"+get_comms_profile_info_string(self.flat_param, self.dp_group)):
                    torch.distributed.all_gather_into_tensor(self.flat_param, self.flat_param_shard, group=self.dp_group)
        else:
            with record_pcl_function("4opt-step_6weightupdate"):
                self.flat_param.mul_(0)
                self.flat_param_shard.copy_(self.flat_fp32_master_param_shard)
            if torch.distributed.is_initialized() and self.data_parallelism > 1:
                if not self.opt_use_chunked_allreduce:
                    with record_pcl_function("3so-comms--1allreduce--"+get_comms_profile_info_string(self.flat_param, self.dp_group)):
                        torch.distributed.all_reduce(self.flat_param, group=self.dp_group)
                else:
                    chunked_allreduce_wrapper(self.flat_param, self.dp_group)

        # Update step
        self.step_id += 1

        return grad_norm

    def state_dict(self):
        if not self.disable_optimizer_state_checkpointing:
            opt_state_dict = {
                "flat_exp_avg_shard" : self.flat_exp_avg_shard,
                "flat_exp_avg_sq_shard" : self.flat_exp_avg_sq_shard,
                "flat_fp32_master_param_shard" : self.flat_fp32_master_param_shard,
                "step_id" : self.step_id
            }
        else:
            opt_state_dict = {
                "step_id" : self.step_id
            }
        return opt_state_dict
    
    def load_state_dict(self, opt_state_dict):
        if "flat_exp_avg_shard" in opt_state_dict:
            self.flat_exp_avg_shard.copy_(opt_state_dict["flat_exp_avg_shard"])
        if "flat_exp_avg_sq_shard" in opt_state_dict:
            self.flat_exp_avg_sq_shard.copy_(opt_state_dict["flat_exp_avg_sq_shard"])
        if "flat_fp32_master_param_shard" in opt_state_dict:
            self.flat_fp32_master_param_shard.copy_(opt_state_dict["flat_fp32_master_param_shard"])
        self.step_id = opt_state_dict["step_id"]

class ParamGroupShardedMixedPrecisionAdamW(Optimizer):
    """
    Stores all parameters, gradients, and optimizer states into a flat buffer.
    Optimizer states include exp_avg and exp_avg_sq which are in FP32
    Master weights are stored in FP32
    Master weights and optimizer states are sharded across multiple GPUs
    ## Limitations
    1. Only one param group
    """
    def __init__(self, 
        params, model,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01, warmup_steps=0,
        dtype=torch.float32, device=None, 
        num_shards=1, shard_id=0, group=None,
        opt_disable_delayed_grad_clipping=False,
        profiler=None,
        tensor_parallelism=1, tp_ind=0, tp_group=None,
        expert_parallelism=1, ep_ind=0, ep_group=None, 
        dpep_ind=0, dpep_group=None,
        shard_divisible_by=512, 
        opt_use_chunked_allreduce=False,
        opt_use_allreduce_for_grad_acc=True, 
        opt_use_allreduce_for_param_gather=False,
        skip_optimizer_step = False,
        verbose=False,
        disable_optimizer_state_checkpointing=False,
        skip_optimizer_step_weight_update=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        assert tensor_parallelism == 1, "ParamGroupShardedMixedPrecisionAdamW does not support tensor parallelism."
        assert device != None
        self.opt_key = "pgshard"

        self.model = model
        self.warmup_steps = warmup_steps
        self.dtype = dtype
        self.device = device
        self.opt_disable_delayed_grad_clipping = opt_disable_delayed_grad_clipping
        self.profiler = profiler

        self.expert_parallelism = expert_parallelism
        self.ep_ind = ep_ind
        self.ep_group = ep_group

        self.dpep_ind = dpep_ind
        self.dpep_group = dpep_group

        self.step_id = 0

        # Sharding options
        self.data_parallelism = num_shards
        self.dp_ind = shard_id
        self.dp_group = group
        self.shard_divisible_by = shard_divisible_by
        self.opt_use_chunked_allreduce = opt_use_chunked_allreduce
        self.opt_use_allreduce_for_grad_acc = opt_use_allreduce_for_grad_acc
        self.opt_use_allreduce_for_param_gather = opt_use_allreduce_for_param_gather
        self.opt_use_fp32_allreduce_for_param_gather = False

        self.param_list = [p for pg in self.param_groups for p in pg["params"]]
        assert len([pg for pg in self.param_groups]) == 1, "Only one parameter group is supported"

        # Param grouping
        replicated_param_names = self.model.get_ep_replicated_param_list() 
        param_list_replicated = [p for n,p in self.model.named_parameters() if n in replicated_param_names]
        param_list_divided   =  [p for n,p in self.model.named_parameters() if not n in replicated_param_names]

        # if torch.distributed.get_rank() == 0:
        #     print("Replicated pararameters", flush=True)
        #     for n,p in self.model.named_parameters():
        #         if n in replicated_param_names:
        #             print(n, p.numel(), flush=True)

        #     print("Duplicated pararameters", flush=True)
        #     for n,p in self.model.named_parameters():
        #         if not n in replicated_param_names:
        #             print(n, p.numel(), flush=True)    

        # Size of flat buffer and shard
        param_sizes_replicated = [p.numel() for p in param_list_replicated]
        param_start_inds_replicated = [sum(param_sizes_replicated[0:i]) for i in range(len(param_sizes_replicated))]
        flat_param_size_replicated = sum(param_sizes_replicated)
        self.shard_size_replicated = (math.ceil(flat_param_size_replicated / (self.data_parallelism * self.expert_parallelism * self.shard_divisible_by))) * self.shard_divisible_by
        flat_buffer_size_replicated = (self.data_parallelism * self.expert_parallelism) * self.shard_size_replicated

        param_sizes_divided = [p.numel() for p in param_list_divided]
        param_start_inds_divided = [sum(param_sizes_divided[0:i]) for i in range(len(param_sizes_divided))]
        flat_param_size_divided = sum(param_sizes_divided)
        self.shard_size_divided = (math.ceil(flat_param_size_divided / (self.data_parallelism * self.shard_divisible_by))) * self.shard_divisible_by
        flat_buffer_size_divided = self.data_parallelism * self.shard_size_divided
        
        if torch.distributed.is_initialized() and ((self.dp_ind == 0) and (self.ep_ind == 0)):
            ideal_shard_size_replicated = (flat_param_size_replicated // (self.data_parallelism * self.expert_parallelism))
            ideal_shard_size_divided = (flat_param_size_divided) // (self.data_parallelism)
            info_str = "********************************************************************************\n"
            info_str += "Total number of parameters : {}\n".format(flat_param_size_replicated + flat_param_size_divided)
            info_str += "Replicated parameters      : {}\n".format(flat_param_size_replicated)
            info_str += "Number of shards           : {}\n".format(self.data_parallelism * self.expert_parallelism)
            info_str += "Ideal shard size           : {}\n".format(ideal_shard_size_replicated)
            info_str += "Actual Shard size          : {} (Padding : {})\n".format(self.shard_size_replicated, self.shard_size_replicated - ideal_shard_size_replicated)
            info_str += "Padding percentage         : {:.3f} %\n".format((1.0 - ideal_shard_size_replicated/self.shard_size_replicated) * 100)
            info_str += "Flat parameter size        : {}\n".format(flat_buffer_size_replicated)
            info_str += "\n"
            info_str += "Divided parameters         : {}\n".format(flat_param_size_divided)
            info_str += "Number of shards           : {}\n".format(self.data_parallelism)
            info_str += "Ideal shard size           : {}\n".format(ideal_shard_size_divided)
            info_str += "Actual Shard size          : {} (Padding : {})\n".format(self.shard_size_divided, self.shard_size_divided - ideal_shard_size_divided)
            info_str += "Padding percentage         : {:.3f} %\n".format((1.0 - ideal_shard_size_divided/self.shard_size_divided) * 100)
            info_str += "Flat parameter size        : {}\n".format(flat_buffer_size_divided)

            info_str += "********************************************************************************"
            print(info_str, flush=True) 

        # Flat buffers
        self.flat_param_replicated = torch.empty(flat_buffer_size_replicated, dtype=self.dtype, device=self.device)
        self.flat_grad_replicated = torch.zeros(flat_buffer_size_replicated, dtype=self.dtype, device=self.device)

        self.flat_param_divided = torch.empty(flat_buffer_size_divided, dtype=self.dtype, device=self.device)
        self.flat_grad_divided = torch.zeros(flat_buffer_size_divided, dtype=self.dtype, device=self.device)

        if skip_optimizer_step == False:
            self.flat_exp_avg_shard_replicated = torch.zeros(self.shard_size_replicated, dtype=torch.float32, device=self.device)
            self.flat_exp_avg_sq_shard_replicated = torch.zeros(self.shard_size_replicated, dtype=torch.float32, device=self.device)

            self.flat_exp_avg_shard_divided = torch.zeros(self.shard_size_divided, dtype=torch.float32, device=self.device)
            self.flat_exp_avg_sq_shard_divided = torch.zeros(self.shard_size_divided, dtype=torch.float32, device=self.device)

            self.flat_fp32_master_param_shard_replicated = torch.zeros(self.shard_size_replicated, dtype=torch.float32, device=self.device)
            self.flat_fp32_master_param_shard_divided = torch.zeros(self.shard_size_divided, dtype=torch.float32, device=self.device)
        else:
            self.flat_exp_avg_shard_replicated = None
            self.flat_exp_avg_sq_shard_replicated = None

            self.flat_exp_avg_shard_divided = None
            self.flat_exp_avg_sq_shard_divided = None

            self.flat_fp32_master_param_shard_replicated = None
            self.flat_fp32_master_param_shard_divided = None         

        # Current shard view (Useful for reduce_scatter and all_gather_into_tensor comms)
        self.flat_param_shard_replicated = self.flat_param_replicated[self.dpep_ind*self.shard_size_replicated:(self.dpep_ind+1)*self.shard_size_replicated]
        self.flat_grad_shard_replicated = self.flat_grad_replicated[self.dpep_ind*self.shard_size_replicated:(self.dpep_ind+1)*self.shard_size_replicated]

        self.flat_param_shard_divided = self.flat_param_divided[self.dp_ind*self.shard_size_divided:(self.dp_ind+1)*self.shard_size_divided]
        self.flat_grad_shard_divided = self.flat_grad_divided[self.dp_ind*self.shard_size_divided:(self.dp_ind+1)*self.shard_size_divided]

        # Mapping parameter into flat parameter.
        for param_id, p in enumerate(param_list_replicated):
            start_ind = param_start_inds_replicated[param_id]
            end_ind = param_start_inds_replicated[param_id] + param_sizes_replicated[param_id]
            # Parameter
            p.data = self.flat_param_replicated[start_ind:end_ind].view_as(p.data).copy_(p.data) # Override
            # Gradient
            p.grad = self.flat_grad_replicated[start_ind:end_ind].view_as(p.data)

        for param_id, p in enumerate(param_list_divided):
            start_ind = param_start_inds_divided[param_id]
            end_ind = param_start_inds_divided[param_id] + param_sizes_divided[param_id]
            # Parameter
            p.data = self.flat_param_divided[start_ind:end_ind].view_as(p.data).copy_(p.data) # Override
            # Gradient
            p.grad = self.flat_grad_divided[start_ind:end_ind].view_as(p.data)

        # Setting master parameter
        if (not skip_optimizer_step):
            self.flat_fp32_master_param_shard_replicated.copy_(self.flat_param_shard_replicated)
            self.flat_fp32_master_param_shard_divided.copy_(self.flat_param_shard_divided)

        self.verbose = verbose
        self.disable_optimizer_state_checkpointing = disable_optimizer_state_checkpointing
        self.skip_optimizer_step_weight_update = skip_optimizer_step_weight_update

    def zero_grad(self):
        self.flat_grad_replicated.zero_()
        self.flat_grad_divided.zero_()

    def step(self, closure=None):
        if self.verbose and (self.pmap.dp_ind == 0):
            print(f"Rank {self.pmap.rank} : Started optimizer step", flush=True)

        # print(f"Beginning of the step (flat_grad*). Rank : {torch.distributed.get_rank()}, PR norm : {torch.nn.utils.get_total_norm(self.flat_grad_replicated)}, PD norm : {torch.nn.utils.get_total_norm(self.flat_grad_divided)}")
        if self.dump_weight_gradients:
            dump_weight_gradients(self.model, self.config, self.pmap, self.opt_key, "beforegradacc")
        
        # print(f"Before gradient accumulation (flat_grad_shard*). Rank : {torch.distributed.get_rank()}, PR norm : {torch.nn.utils.get_total_norm(self.flat_grad_shard_replicated)}, PD norm : {torch.nn.utils.get_total_norm(self.flat_grad_shard_divided)}")
        # If it is distributed, accumulate gradients
        with record_pcl_function("4opt-step_1gradacc"):
            # Replicated grad_acc (DP+EP)
            if torch.distributed.is_initialized() and ((self.data_parallelism > 1) or (self.expert_parallelism > 1)):
                if self.verbose and (self.pmap.dp_ind == 0):
                    print(f"Rank {self.pmap.rank} : Before gradient accumulation for replicated parameters", flush=True)
                if not self.opt_use_allreduce_for_grad_acc:
                    with record_pcl_function("3so-comms--2reduce_scatter--" + get_comms_profile_info_string(self.flat_grad_replicated, self.dpep_group)):
                        torch.distributed.reduce_scatter_tensor(self.flat_grad_shard_replicated, self.flat_grad_replicated, group=self.dpep_group)
                        torch.distributed.barrier(group=self.dpep_group)
                    # Average the gradients
                    self.flat_grad_shard_replicated.div_(self.data_parallelism * self.expert_parallelism)
                else:
                    if not self.opt_use_chunked_allreduce:
                        with record_pcl_function("3so-comms--1allreduce--" + get_comms_profile_info_string(self.flat_grad_replicated, self.dpep_group)):
                            torch.distributed.all_reduce(self.flat_grad_replicated, group=self.dpep_group)
                            torch.distributed.barrier(group=self.dpep_group)
                    else:
                        chunked_allreduce_wrapper(self.flat_grad_replicated, self.dpep_group, barrier=True)
                    # Average the gradients
                    self.flat_grad_replicated.div_(self.data_parallelism * self.expert_parallelism)

            # Divided grad_acc
            if torch.distributed.is_initialized() and (self.data_parallelism > 1):
                if self.verbose and (self.pmap.dp_ind == 0):
                    print(f"Rank {self.pmap.rank} : Before gradient accumulation for divided parameters", flush=True)
                if not self.opt_use_allreduce_for_grad_acc:
                    with record_pcl_function("3so-comms--2reduce_scatter--" + get_comms_profile_info_string(self.flat_grad_divided, self.dp_group)):
                        torch.distributed.reduce_scatter_tensor(self.flat_grad_shard_divided, self.flat_grad_divided, group=self.dp_group)
                        torch.distributed.barrier(group=self.dp_group)
                    # Average the gradients
                    self.flat_grad_shard_divided.div_(self.data_parallelism)
                else:
                    if not self.opt_use_chunked_allreduce:
                        with record_pcl_function("3so-comms--1allreduce--" + get_comms_profile_info_string(self.flat_grad_divided, self.dp_group)):
                            torch.distributed.all_reduce(self.flat_grad_divided, group=self.dp_group)
                            torch.distributed.barrier(group=self.dp_group)
                    else:
                        chunked_allreduce_wrapper(self.flat_grad_divided, self.dp_group, barrier=True)
                    # Average the gradients
                    self.flat_grad_divided.div_(self.data_parallelism)                    

        if self.dump_weight_gradients:
            dump_weight_gradients(self.model, self.config, self.pmap, self.opt_key, "aftergradacc")

        # print(f"Rank : {torch.distributed.get_rank()} Grad norm : {torch.nn.utils.get_total_norm([self.flat_grad_replicated, self.flat_grad_divided])}")

        # Global gradient norm
        with record_pcl_function("4opt-step_3global-gradnorm"):
            if self.verbose and (self.pmap.dp_ind == 0):
                print(f"Rank {self.pmap.rank} : Before grad norm calculation", flush=True)
            grad_shard_norm_replicated = torch.nn.utils.get_total_norm(self.flat_grad_shard_replicated)
            grad_shard_norm_divided = torch.nn.utils.get_total_norm(self.flat_grad_shard_divided)
            if torch.distributed.is_initialized():
                grad_shard_norm_replicated_fp32 = grad_shard_norm_replicated.to(torch.float32)
                grad_shard_norm_sq_replicated_fp32 = torch.pow(grad_shard_norm_replicated_fp32, 2)
                torch.distributed.all_reduce(grad_shard_norm_sq_replicated_fp32)

                grad_shard_norm_divided_fp32 = grad_shard_norm_divided.to(torch.float32)
                grad_shard_norm_sq_divided_fp32 = torch.pow(grad_shard_norm_divided_fp32, 2)
                torch.distributed.all_reduce(grad_shard_norm_sq_divided_fp32)

                grad_shard_norm_sq_fp32 = grad_shard_norm_sq_replicated_fp32 + grad_shard_norm_sq_divided_fp32
                grad_norm = torch.sqrt(grad_shard_norm_sq_fp32)
            else:
                grad_norm = torch.nn.utils.get_total_norm([self.flat_grad_shard_replicated, self.flat_grad_shard_divided])
            
        # Gradient clipping
        with record_pcl_function("4opt-step_4clip-grads"):
            if self.opt_disable_delayed_grad_clipping or \
                ((self.opt_disable_delayed_grad_clipping == False) and (self.step_id >= self.warmup_steps)):
                max_norm = 1.0
                use_custom_grad_clipping = True
                if not use_custom_grad_clipping:
                    torch.nn.utils.clip_grads_with_norm_(self.model.parameters(), max_norm, grad_norm)
                else:
                    clip_coef = max_norm / (grad_norm + 1e-6)  # avoid divide-by-zero
                    if clip_coef < 1.0:
                        self.flat_grad_shard_replicated.mul_(clip_coef)  # In-place scaling
                        self.flat_grad_shard_divided.mul_(clip_coef)

        with record_pcl_function("4opt-step_5compute"):
            if not self.skip_optimizer_step_weight_update:
                if self.verbose and (self.pmap.dp_ind == 0):
                    print(f"Rank {self.pmap.rank} : Before optimizer step compute", flush=True)
                opt_info_list = [
                    {
                        "shard_size" : self.shard_size_replicated,
                        "flat_param_shard" : self.flat_param_shard_replicated,
                        "flat_grad_shard" : self.flat_grad_shard_replicated,
                        "flat_fp32_master_param_shard" : self.flat_fp32_master_param_shard_replicated,
                        "flat_exp_avg_shard" : self.flat_exp_avg_shard_replicated,
                        "flat_exp_avg_sq_shard" : self.flat_exp_avg_sq_shard_replicated
                    },
                    {
                        "shard_size" : self.shard_size_divided,
                        "flat_param_shard" : self.flat_param_shard_divided,
                        "flat_grad_shard" : self.flat_grad_shard_divided,
                        "flat_fp32_master_param_shard" : self.flat_fp32_master_param_shard_divided,
                        "flat_exp_avg_shard" : self.flat_exp_avg_shard_divided,
                        "flat_exp_avg_sq_shard" : self.flat_exp_avg_sq_shard_divided
                    }
                ]

                for opt_info in opt_info_list:
                    # Hyper parameters
                    beta1, beta2 = self.defaults["betas"]
                    eps = self.defaults["eps"]
                    lr = self.param_groups[0]["lr"]
                    step_size = lr
                    weight_decay = self.defaults["weight_decay"]

                    # Tensors
                    size = opt_info["shard_size"] # Sum of sizes of parameters
                    data = opt_info["flat_param_shard"] # Weights
                    grad = opt_info["flat_grad_shard"] # Gradients
                    data_master = opt_info["flat_fp32_master_param_shard"] # OS master weights
                    exp_avg = opt_info["flat_exp_avg_shard"] # OS moment1
                    exp_avg_sq = opt_info["flat_exp_avg_sq_shard"] # OS moment2

                    # Doing in chunks to reduce memory requirement
                    chunk_size = 128 * 1024 * 1024
                    num_chunks = math.ceil(size / chunk_size)
                    tensor_list = [data, grad, data_master, exp_avg, exp_avg_sq]
                    for chunk_id in range(num_chunks):
                        start_ind = chunk_id * chunk_size
                        end_ind = min((chunk_id+1)*chunk_size, size)

                        # Getting chunks
                        data_chunk, grad_chunk, data_master_chunk, exp_avg_chunk, exp_avg_sq_chunk = [tensor[start_ind:end_ind] for tensor in tensor_list]

                        # Using FP32 gradient for optimizer step
                        grad_chunk_fp32 = grad_chunk.to(torch.float32)

                        # Optimizer compute
                        exp_avg_chunk.mul_(beta1).add_(grad_chunk_fp32, alpha=(1.0 - beta1))
                        exp_avg_sq_chunk.mul_(beta2).addcmul_(grad_chunk_fp32, grad_chunk_fp32, value=1.0 - beta2)
                        exp_avg_hat_chunk = exp_avg_chunk / (1.0 - math.pow(beta1, (self.step_id+1))) 
                        exp_avg_sq_hat_chunk = exp_avg_sq_chunk / (1.0 - math.pow(beta2, (self.step_id+1)))
                        denom_chunk = exp_avg_sq_hat_chunk.sqrt().add_(eps)
                        data_master_chunk.addcdiv_(exp_avg_hat_chunk, denom_chunk, value=-step_size)

                        # Handling weight decay
                        if weight_decay > 0.0:
                            data_master_chunk.add_(data_master_chunk, alpha=(-lr * weight_decay)) #[data = data - (lr * weight_decay)]

                        # Convert master weights to model weights dtype
                        if not self.opt_use_allreduce_for_param_gather:
                            data_chunk.copy_(data_master_chunk)


        if self.verbose and (self.pmap.dp_ind == 0):
            print(f"Rank {self.pmap.rank} : Before parameter gather", flush=True)
        if not self.opt_use_allreduce_for_param_gather:
            with record_pcl_function("4opt-step_2paramgather"):
                if torch.distributed.is_initialized() and (self.expert_parallelism > 1 or self.data_parallelism > 1):
                    with record_pcl_function("3so-comms--3allgather--"+get_comms_profile_info_string(self.flat_param_replicated, self.dpep_group)):
                        torch.distributed.all_gather_into_tensor(self.flat_param_replicated, self.flat_param_shard_replicated, group=self.dpep_group)
                
                if torch.distributed.is_initialized() and (self.data_parallelism > 1):
                    with record_pcl_function("3so-comms--3allgather--"+get_comms_profile_info_string(self.flat_param_divided, self.dp_group)):
                        torch.distributed.all_gather_into_tensor(self.flat_param_divided, self.flat_param_shard_divided, group=self.dp_group)
        else:
            with record_pcl_function("4opt-step_6weightupdate"):
                self.flat_param_replicated.mul_(0)
                self.flat_param_shard_replicated.copy_(self.flat_fp32_master_param_shard_replicated)

                self.flat_param_divided.mul_(0)
                self.flat_param_shard_divided.copy_(self.flat_fp32_master_param_shard_divided)

            with record_pcl_function("4opt-step_2paramgather"):
                if torch.distributed.is_initialized() and (self.expert_parallelism > 1 or self.data_parallelism > 1):
                    if not self.opt_use_chunked_allreduce:
                        if not self.opt_use_fp32_allreduce_for_param_gather:
                            with record_pcl_function("3so-comms--1allreduce--" + get_comms_profile_info_string(self.flat_param_replicated, self.dpep_group)):
                                torch.distributed.all_reduce(self.flat_param_replicated, group=self.dpep_group)
                                torch.distributed.barrier(group=self.dpep_group)
                        else:
                            with record_pcl_function("4opt-step_7bf16tofp32_conversion"):
                                with torch.no_grad():
                                    flat_param_replicated_fp32 = self.flat_param_replicated.to(torch.float32)
                            with record_pcl_function("3so-comms--1allreduce--" + get_comms_profile_info_string(flat_param_replicated_fp32, self.dpep_group)):
                                torch.distributed.all_reduce(flat_param_replicated_fp32, group=self.dpep_group)
                                torch.distributed.barrier(group=self.dpep_group)
                            with record_pcl_function("4opt-step_8fp32tobf16_conversion"):
                                with torch.no_grad():
                                    self.flat_param_replicated.copy_(flat_param_replicated_fp32)
                    else:
                        chunked_allreduce_wrapper(self.flat_param_replicated, self.dpep_group, barrier=True)
        
                if torch.distributed.is_initialized() and (self.data_parallelism > 1):
                    if not self.opt_use_chunked_allreduce:
                        if not self.opt_use_fp32_allreduce_for_param_gather:
                            with record_pcl_function("3so-comms--1allreduce--" + get_comms_profile_info_string(self.flat_param_divided, self.dp_group)):
                                torch.distributed.all_reduce(self.flat_param_divided, group=self.dp_group)
                                torch.distributed.barrier(group=self.dp_group)
                        else:
                            with record_pcl_function("4opt-step_7bf16tofp32_conversion"):
                                with torch.no_grad():
                                    flat_param_divided_fp32 = self.flat_param_divided.to(torch.float32)
                            with record_pcl_function("3so-comms--1allreduce--" + get_comms_profile_info_string(flat_param_divided_fp32, self.dp_group)):
                                torch.distributed.all_reduce(flat_param_divided_fp32, group=self.dp_group)
                                torch.distributed.barrier(group=self.dp_group)
                            with record_pcl_function("4opt-step_8fp32tobf16_conversion"):
                                with torch.no_grad():
                                    self.flat_param_divided.copy_(flat_param_divided_fp32)
                    else:
                        chunked_allreduce_wrapper(self.flat_param_divided, self.dp_group, barrier=True)

        if self.verbose and (self.pmap.dp_ind == 0):
            print(f"Rank {self.pmap.rank} : Completed optimizer step", flush=True)

        # Update step
        self.step_id += 1

        return grad_norm

    def state_dict(self):
        if not self.disable_optimizer_state_checkpointing:
            opt_state_dict = {
                "flat_exp_avg_shard_replicated" : self.flat_exp_avg_shard_replicated,
                "flat_exp_avg_sq_shard_replicated" : self.flat_exp_avg_sq_shard_replicated,
                "flat_fp32_master_param_shard_replicated" : self.flat_fp32_master_param_shard_replicated,
                "flat_exp_avg_shard_divided" : self.flat_exp_avg_shard_divided,
                "flat_exp_avg_sq_shard_divided" : self.flat_exp_avg_sq_shard_divided,
                "flat_fp32_master_param_shard_divided" : self.flat_fp32_master_param_shard_divided,
                "step_id" : self.step_id
            }
        else:
            opt_state_dict = {
                "step_id" : self.step_id
            }
        return opt_state_dict
    
    def load_state_dict(self, opt_state_dict):
        if "flat_exp_avg_shard_replicated" in opt_state_dict:
            self.flat_exp_avg_shard_replicated.copy_(opt_state_dict["flat_exp_avg_shard_replicated"])
        if "flat_exp_avg_sq_shard_replicated" in opt_state_dict:
            self.flat_exp_avg_sq_shard_replicated.copy_(opt_state_dict["flat_exp_avg_sq_shard_replicated"])
        if "flat_fp32_master_param_shard_replicated" in opt_state_dict:
            self.flat_fp32_master_param_shard_replicated.copy_(opt_state_dict["flat_fp32_master_param_shard_replicated"])
        if "flat_exp_avg_shard_divided" in opt_state_dict:
            self.flat_exp_avg_shard_divided.copy_(opt_state_dict["flat_exp_avg_shard_divided"])
        if "flat_exp_avg_sq_shard_divided" in opt_state_dict:
            self.flat_exp_avg_sq_shard_divided.copy_(opt_state_dict["flat_exp_avg_sq_shard_divided"])
        if "flat_fp32_master_param_shard_divided" in opt_state_dict:
            self.flat_fp32_master_param_shard_divided.copy_(opt_state_dict["flat_fp32_master_param_shard_divided"])
            
        self.step_id = opt_state_dict["step_id"]

class SubShardedMixedPrecisionAdamW(Optimizer):
    """
    Stores all parameters, gradients, and optimizer states into a flat buffer.
    Optimizer states include exp_avg and exp_avg_sq which are in FP32
    Master weights are stored in FP32
    Master weights and optimizer states are sharded across multiple GPUs
    The flat param is divided into chunks, and each chunk is processed separately.

    ## Limitations
    1. Only one param group
    """
    def __init__(self,
        params, model,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01, warmup_steps=0,
        dtype=torch.float32, device=None, 
        num_shards=1, shard_id=0, group=None,
        opt_disable_delayed_grad_clipping=False,
        profiler=None,
        tensor_parallelism=1, tp_ind=0, tp_group=None,
        expert_parallelism=1, ep_ind=0, ep_group=None,
        dpep_ind=0, dpep_group=None,
        skip_optimizer_step=False,
        chunk_size_limit=134217728,
        opt_use_chunked_allreduce=False,
        opt_use_allreduce_for_grad_acc=True, 
        opt_use_allreduce_for_param_gather=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        assert device != None

        assert opt_use_chunked_allreduce == False, "SubShardedMixedPrecisionAdamW does not support chunked allreduce yet."

        self.model = model
        self.warmup_steps = warmup_steps
        self.dtype = dtype
        self.device = device
        self.opt_disable_delayed_grad_clipping = opt_disable_delayed_grad_clipping
        self.profiler = profiler

        self.data_parallelism = num_shards
        self.dp_ind = shard_id
        self.dp_group = group

        self.tensor_parallelism = tensor_parallelism
        self.tp_ind = tp_ind
        self.tp_group = tp_group

        self.expert_parallelism = expert_parallelism
        self.ep_ind = ep_ind
        self.ep_group = ep_group

        # Current implementation specific
        self.chunk_size_limit = chunk_size_limit
        self.opt_use_chunked_allreduce = opt_use_chunked_allreduce
        self.opt_use_allreduce_for_grad_acc = opt_use_allreduce_for_grad_acc
        self.opt_use_allreduce_for_param_gather = opt_use_allreduce_for_param_gather

        self.step_id = 0

        self.param_list = [p for pg in self.param_groups for p in pg["params"]]
        assert len([pg for pg in self.param_groups]) == 1, "Only one parameter group is supported"

        self.param_sizes = [p.numel() for p in self.param_list]
        self.param_start_inds = [sum(self.param_sizes[0:i]) for i in range(len(self.param_sizes))]
        self.all_param_size = sum(self.param_sizes) # Total number of parameters

        # Choose the biggest possible value that is less than 256 M size and is multiple of DP
        self.chunk_size = ((self.chunk_size_limit // self.data_parallelism) * self.data_parallelism)
        self.num_chunks = math.ceil(self.all_param_size / self.chunk_size)
        self.flat_buffer_size = self.num_chunks * self.chunk_size

        self.sub_chunk_size = self.chunk_size // self.data_parallelism
        self.shard_size = self.num_chunks * self.sub_chunk_size
        
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            ideal_shard_size = (self.all_param_size // self.data_parallelism)
            info_str = "********************************************************************************\n"
            info_str += "Total number of parameters : {}\n".format(self.all_param_size)
            info_str += "Flat buffer size           : {} (Padding : {})\n".format(self.flat_buffer_size, self.flat_buffer_size - self.all_param_size)
            info_str += "Chunk size limit           : {}\n".format(self.chunk_size_limit)
            info_str += "Chunk size                 : {}\n".format(self.chunk_size)
            info_str += "Sub chunk size             : {}\n".format(self.sub_chunk_size)
            info_str += "Number of chunks           : {}\n".format(self.num_chunks)
            info_str += "********************************************************************************"
            print(info_str, flush=True)
            
        # Flat buffers
        self.flat_param = torch.empty(self.flat_buffer_size, dtype=dtype, device=device)
        self.flat_grad = torch.zeros(self.flat_buffer_size, dtype=self.dtype, device=self.device)
        if skip_optimizer_step == False:
            self.flat_exp_avg_shard = torch.zeros(self.shard_size, dtype=torch.float32, device=self.device)
            self.flat_exp_avg_sq_shard = torch.zeros(self.shard_size, dtype=torch.float32, device=self.device)
            self.flat_fp32_master_param_shard = torch.zeros(self.shard_size, dtype=torch.float32, device=self.device)
        else:
            self.flat_exp_avg_shard = None
            self.flat_exp_avg_sq_shard = None
            self.flat_fp32_master_param_shard = None
        
        # Mapping parameter into flat parameter.
        for param_id, p in enumerate(self.param_list):
            start_ind = self.param_start_inds[param_id]
            end_ind = self.param_start_inds[param_id] + self.param_sizes[param_id]
            # Parameter
            p.data = self.flat_param[start_ind:end_ind].view_as(p.data).copy_(p.data) # Override
            # Gradient
            p.grad = self.flat_grad[start_ind:end_ind].view_as(p.data)

        # Setting master parameter
        if skip_optimizer_step == False:
            for chunk_id in range(self.num_chunks):
                flat_param_chunk = self.flat_param[chunk_id*self.chunk_size:(chunk_id+1)*self.chunk_size]
                flat_param_shard_sub_chunk = flat_param_chunk[self.dp_ind*self.sub_chunk_size:(self.dp_ind+1)*self.sub_chunk_size]
                flat_fp32_master_param_shard_sub_chunk = self.flat_fp32_master_param_shard[chunk_id*self.sub_chunk_size:(chunk_id+1)*self.sub_chunk_size]
                flat_fp32_master_param_shard_sub_chunk.copy_(flat_param_shard_sub_chunk)

    def zero_grad(self):
        self.flat_grad.zero_()

    def step(self, closure=None):
        flat_param_chunks = [self.flat_param[chunk_id*self.chunk_size:(chunk_id+1)*self.chunk_size] for chunk_id in range(self.num_chunks)]
        flat_grad_chunks = [self.flat_grad[chunk_id*self.chunk_size:(chunk_id+1)*self.chunk_size] for chunk_id in range(self.num_chunks)]
        flat_param_shard_sub_chunks = [flat_param_chunks[chunk_id][self.dp_ind*self.sub_chunk_size:(self.dp_ind+1)*self.sub_chunk_size] for chunk_id in range(self.num_chunks)]
        flat_grad_shard_sub_chunks  = [ flat_grad_chunks[chunk_id][self.dp_ind*self.sub_chunk_size:(self.dp_ind+1)*self.sub_chunk_size] for chunk_id in range(self.num_chunks)]
        flat_exp_avg_shard_sub_chunks = [self.flat_exp_avg_shard[chunk_id*self.sub_chunk_size:(chunk_id+1)*self.sub_chunk_size] for chunk_id in range(self.num_chunks)]
        flat_exp_avg_sq_shard_sub_chunks = [self.flat_exp_avg_sq_shard[chunk_id*self.sub_chunk_size:(chunk_id+1)*self.sub_chunk_size] for chunk_id in range(self.num_chunks)]
        flat_fp32_master_param_shard_sub_chunks = [self.flat_fp32_master_param_shard[chunk_id*self.sub_chunk_size:(chunk_id+1)*self.sub_chunk_size] for chunk_id in range(self.num_chunks)]

        # If it is distributed, accumulate gradients
        with record_pcl_function("4opt-step_1gradacc"):
            # EP grad_acc
            if torch.distributed.is_initialized() and self.expert_parallelism > 1:
                replicated_param_list = self.model.get_ep_replicated_param_list()
                for name, param in self.model.named_parameters():
                    if name in replicated_param_list:
                        torch.distributed.all_reduce(param.grad, group=self.ep_group)
                    param.grad.div_(self.expert_parallelism)

            # DP grad_acc
            for chunk_id in range(self.num_chunks):
                if torch.distributed.is_initialized() and self.data_parallelism > 1:
                    if not self.opt_use_allreduce_for_grad_acc:
                        with record_pcl_function("3so-comms--2reduce_scatter--" + get_comms_profile_info_string(flat_grad_chunks[chunk_id], self.dp_group)):
                            torch.distributed.reduce_scatter_tensor(flat_grad_shard_sub_chunks[chunk_id], flat_grad_chunks[chunk_id], group=self.dp_group)
                        flat_grad_shard_sub_chunks[chunk_id].div_(self.data_parallelism)
                    else:
                        with record_pcl_function("3so-comms--1allreduce--" + get_comms_profile_info_string(flat_grad_chunks[chunk_id], self.dp_group)):
                            torch.distributed.all_reduce(flat_grad_chunks[chunk_id], group=self.dp_group)
                        flat_grad_chunks[chunk_id].div_(self.data_parallelism)

        # Clip gradients
        with record_pcl_function("4opt-step_3global-gradnorm"):
            if self.expert_parallelism > 1:
                # Scale down EP replicated parameters gradients
                ep_replicated_param_list = self.model.get_ep_replicated_param_list()
                for name, param in self.model.named_parameters():
                    if name in ep_replicated_param_list:
                        param.grad.data.div_(math.sqrt(self.expert_parallelism))

            if self.tensor_parallelism > 1:
                # Scale down TP replicated parameters gradients
                tp_replicated_param_list = self.model.get_tp_replicated_param_list()
                for name,param in self.model.named_parameters():
                    if name in replicated_params:
                        param.grad.data.div_(math.sqrt(self.tensor_parallelism))

            grad_shard_norm = torch.nn.utils.get_total_norm(flat_grad_shard_sub_chunks)
            if torch.distributed.is_initialized():
                grad_shard_norm_fp32= grad_shard_norm.to(torch.float32)
                grad_shard_norm_sq_fp32 = torch.pow(grad_shard_norm_fp32, 2)
                torch.distributed.all_reduce(grad_shard_norm_sq_fp32)
                grad_norm = torch.sqrt(grad_shard_norm_sq_fp32)
            else:
                grad_norm = grad_shard_norm
            
            if self.expert_parallelism > 1:
                # Undo scale down
                ep_replicated_param_list = self.model.get_ep_replicated_param_list()
                for name, param in self.model.named_parameters():
                    if name in ep_replicated_param_list:
                        param.grad.data.mul_(math.sqrt(self.expert_parallelism))
            
            if self.tensor_parallelism > 1:
                # Undo scale down
                tp_replicated_param_list = self.model.get_tp_replicated_param_list()
                for name, param in self.model.named_parameters():
                    if name in tp_replicated_param_list:
                        param.grad.data.mul_(math.sqrt(self.tensor_parallelism))

        with record_pcl_function("4opt-step_4clip-grads"):
            if self.opt_disable_delayed_grad_clipping or \
                ((self.opt_disable_delayed_grad_clipping == False) and (self.step_id >= self.warmup_steps)):
                max_norm = 1.0
                use_custom_grad_clipping = True
                if not use_custom_grad_clipping:
                    torch.nn.utils.clip_grads_with_norm_(self.model.parameters(), max_norm, grad_norm)
                else:
                    clip_coef = max_norm / (grad_norm + 1e-6)  # avoid divide-by-zero
                    if clip_coef < 1.0:
                        for chunk_id in range(self.num_chunks):
                            flat_grad_shard_sub_chunks[chunk_id].mul_(clip_coef)

        # Weight update
        for chunk_id in range(self.num_chunks):
            # Weight update
            with record_pcl_function("4opt-step_5compute"):
                # Hyper parameters
                beta1, beta2 = self.defaults["betas"]
                eps = self.defaults["eps"]
                lr = self.param_groups[0]["lr"]
                step_size = lr
                weight_decay = self.defaults["weight_decay"]

                # Getting chunks
                flat_param_shard_sub_chunk      = flat_param_shard_sub_chunks[chunk_id]
                flat_grad_shard_sub_chunk       = flat_grad_shard_sub_chunks[chunk_id]
                flat_exp_avg_shard_sub_chunk    = flat_exp_avg_shard_sub_chunks[chunk_id]
                flat_exp_avg_sq_shard_sub_chunk = flat_exp_avg_sq_shard_sub_chunks[chunk_id]
                flat_fp32_master_param_shard_sub_chunk = flat_fp32_master_param_shard_sub_chunks[chunk_id]

                # Using FP32 gradient for optimizer step
                flat_grad_shard_sub_chunk_fp32 = flat_grad_shard_sub_chunk.to(torch.float32)

                # Optimizer compute
                flat_exp_avg_shard_sub_chunk.mul_(beta1).add_(flat_grad_shard_sub_chunk_fp32, alpha=(1.0 - beta1))
                flat_exp_avg_sq_shard_sub_chunk.mul_(beta2).addcmul_(flat_grad_shard_sub_chunk_fp32, flat_grad_shard_sub_chunk_fp32, value=1.0 - beta2)
                flat_exp_avg_hat_sub_chunk = flat_exp_avg_shard_sub_chunk / (1.0 - math.pow(beta1, (self.step_id+1))) 
                flat_exp_avg_sq_hat_sub_chunk = flat_exp_avg_sq_shard_sub_chunk / (1.0 - math.pow(beta2, (self.step_id+1)))
                denom_sub_chunk = flat_exp_avg_sq_hat_sub_chunk.sqrt().add_(eps)
                flat_fp32_master_param_shard_sub_chunk.addcdiv_(flat_exp_avg_hat_sub_chunk, denom_sub_chunk, value=-step_size)

                # Handling weight decay
                if weight_decay > 0.0:
                    flat_fp32_master_param_shard_sub_chunk.add_(flat_fp32_master_param_shard_sub_chunk, alpha=(-lr * weight_decay)) #[data = data - (lr * weight_decay)]

                if not self.opt_use_allreduce_for_param_gather:
                    # Convert master weights to model weights dtype
                    flat_param_shard_sub_chunk.copy_(flat_fp32_master_param_shard_sub_chunk)

            # When using allreduce for param gather, it is immediately done after weight update
            if self.opt_use_allreduce_for_param_gather:
                with record_pcl_function("4opt-step_6weightupdate"):
                    flat_param_shard_sub_chunk = flat_param_shard_sub_chunks[chunk_id]
                    flat_fp32_master_param_shard_sub_chunk = flat_fp32_master_param_shard_sub_chunks[chunk_id]

                    flat_param_chunks[chunk_id].mul_(0.0) # Reset parameters to zero
                    flat_param_shard_sub_chunk.copy_(flat_fp32_master_param_shard_sub_chunk)

                if torch.distributed.is_initialized() and self.data_parallelism > 1:
                    with record_pcl_function("3so-comms--1allreduce--"+get_comms_profile_info_string(flat_param_chunks[chunk_id], self.dp_group)):
                        torch.distributed.all_reduce(flat_param_chunks[chunk_id], group=self.dp_group)

        ## Do param gather
        if not self.opt_use_allreduce_for_param_gather:
            for chunk_id in range(self.num_chunks):
                if torch.distributed.is_initialized() and self.data_parallelism > 1:
                    with record_pcl_function("3so-comms--3allgather--"+get_comms_profile_info_string(flat_param_chunks[chunk_id], self.dp_group)):
                        torch.distributed.all_gather_into_tensor(flat_param_chunks[chunk_id], flat_param_shard_sub_chunks[chunk_id], group=self.dp_group)

        # Update step
        self.step_id += 1

        return grad_norm

    def state_dict(self):
        opt_state_dict = {
            "flat_exp_avg_shard" : self.flat_exp_avg_shard,
            "flat_exp_avg_sq_shard" : self.flat_exp_avg_sq_shard,
            "flat_fp32_master_param_shard" : self.flat_fp32_master_param_shard,
            "step_id" : self.step_id
        }
        return opt_state_dict
    
    def load_state_dict(self, opt_state_dict):
        self.flat_exp_avg_shard.copy_(opt_state_dict["flat_exp_avg_shard"])
        self.flat_exp_avg_sq_shard.copy_(opt_state_dict["flat_exp_avg_sq_shard"])
        self.flat_fp32_master_param_shard.copy_(opt_state_dict["flat_fp32_master_param_shard"])
        self.step_id = opt_state_dict["step_id"]

class ParamGroupSubShardedMixedPrecisionAdamW(Optimizer):
    """
    Stores all parameters, gradients, and optimizer states into a flat buffer.
    Optimizer states include exp_avg and exp_avg_sq which are in FP32
    Master weights are stored in FP32
    Master weights and optimizer states are sharded across multiple GPUs
    The flat param is divided into chunks, and each chunk is processed separately.

    ## Limitations
    1. Only one param group
    """
    def __init__(self,
        params, model,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01, warmup_steps=0,
        dtype=torch.float32, device=None, 
        num_shards=1, shard_id=0, group=None,
        opt_disable_delayed_grad_clipping=False,
        profiler=None,
        tensor_parallelism=1, tp_ind=0, tp_group=None,
        expert_parallelism=1, ep_ind=0, ep_group=None,
        dpep_ind=0, dpep_group=None,
        skip_optimizer_step=False,
        shard_divisible_by=512,
        chunk_size_limit=134217728,
        opt_use_fp32_for_grad_acc=False,
        opt_use_chunked_allreduce=False,
        opt_use_allreduce_for_grad_acc=True, 
        opt_use_allreduce_for_param_gather=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        assert tensor_parallelism == 1, "ParamGroupSubShardMixedPrecisionAdamW optimizer does not support tensor parallelism."
        assert opt_use_chunked_allreduce == False, "ParamGroupSubShardedMixedPrecisionAdamW does not support chunked allreduce."
        assert device != None

        self.model = model
        self.warmup_steps = warmup_steps
        self.dtype = dtype
        self.device = device
        self.opt_disable_delayed_grad_clipping = opt_disable_delayed_grad_clipping
        self.profiler = profiler

        self.data_parallelism = num_shards
        self.dp_ind = shard_id
        self.dp_group = group

        self.tensor_parallelism = tensor_parallelism
        self.tp_ind = tp_ind
        self.tp_group = tp_group

        self.expert_parallelism = expert_parallelism
        self.ep_ind = ep_ind
        self.ep_group = ep_group

        self.dpep_ind = dpep_ind
        self.dpep_group = dpep_group

        # Current implementation specific
        self.shard_divisible_by = shard_divisible_by
        self.chunk_size_limit = chunk_size_limit
        self.opt_use_fp32_for_grad_acc = opt_use_fp32_for_grad_acc
        self.opt_use_chunked_allreduce = opt_use_chunked_allreduce
        self.opt_use_allreduce_for_grad_acc = opt_use_allreduce_for_grad_acc
        self.opt_use_allreduce_for_param_gather = opt_use_allreduce_for_param_gather
        self.opt_alternate_compute_and_comms = True

        self.step_id = 0

        self.param_list = [p for pg in self.param_groups for p in pg["params"]]
        assert len([pg for pg in self.param_groups]) == 1, "Only one parameter group is supported"

        # Grouping parameters into replicated and divided groups
        param_names_rep = self.model.get_ep_replicated_param_list() 
        param_list_rep = [p for n,p in self.model.named_parameters() if n in param_names_rep]
        param_list_div = [p for n,p in self.model.named_parameters() if not n in param_names_rep]

        # Replicated parameters buffer size calculation
        param_sizes_rep = [p.numel() for p in param_list_rep]
        param_start_inds_rep = [sum(param_sizes_rep[0:i]) for i in range(len(param_sizes_rep))]
        all_param_size_rep = sum(param_sizes_rep)
        # self.chunk_size_rep = ((self.chunk_size_limit // (self.data_parallelism * self.expert_parallelism * self.shard_divisible_by)) * (self.data_parallelism * self.expert_parallelism * self.shard_divisible_by))
        self.chunk_size_rep = math.ceil(self.chunk_size_limit / (self.data_parallelism * self.expert_parallelism * self.shard_divisible_by)) * (self.data_parallelism * self.expert_parallelism * self.shard_divisible_by)
        self.num_chunks_rep = math.ceil(all_param_size_rep / self.chunk_size_rep)
        self.flat_buffer_size_rep = self.num_chunks_rep * self.chunk_size_rep
        self.sub_chunk_size_rep = self.chunk_size_rep // (self.data_parallelism * self.expert_parallelism)
        self.shard_size_rep = self.num_chunks_rep * self.sub_chunk_size_rep

        # Divided parameters buffer size calculation
        param_sizes_div = [p.numel() for p in param_list_div]
        param_start_inds_div = [sum(param_sizes_div[0:i]) for i in range(len(param_sizes_div))]
        all_param_size_div = sum(param_sizes_div)
        #self.chunk_size_div = ((self.chunk_size_limit // (self.data_parallelism * self.shard_divisible_by)) * (self.data_parallelism * self.shard_divisible_by))
        self.chunk_size_div = math.ceil(self.chunk_size_limit / (self.data_parallelism * self.shard_divisible_by)) * (self.data_parallelism * self.shard_divisible_by)
        self.num_chunks_div = math.ceil(all_param_size_div / self.chunk_size_div)
        self.flat_buffer_size_div = self.num_chunks_div * self.chunk_size_div
        self.sub_chunk_size_div = self.chunk_size_div // (self.data_parallelism)
        self.shard_size_div = self.num_chunks_div * self.sub_chunk_size_div

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        if torch.distributed.is_initialized() and ((self.dp_ind == 0) and (self.ep_ind == 0)):
            info_str = "********************************************************************************\n"
            info_str += "Index (dp_ind, ep_ind)     : ({}, {})\n".format(self.dp_ind, self.ep_ind)
            info_str += "Total number of parameters : {}\n".format(all_param_size_rep + all_param_size_div)
            info_str += "\n"
            info_str += "Replicated parameters      : {}\n".format(all_param_size_rep)
            info_str += "Flat buffer size           : {} (Padding : {})\n".format(self.flat_buffer_size_rep, self.flat_buffer_size_rep - all_param_size_rep)
            info_str += "Shard size                 : {}\n".format(self.shard_size_rep)
            info_str += "Chunk size limit           : {}\n".format(self.chunk_size_limit)
            info_str += "Chunk size                 : {}\n".format(self.chunk_size_rep)
            info_str += "Number of chunks           : {}\n".format(self.num_chunks_rep)
            info_str += "Sub chunk size             : {}\n".format(self.sub_chunk_size_rep)
            info_str += "Number of shards           : {}\n".format(self.data_parallelism * self.expert_parallelism)
            info_str += "\n"
            info_str += "Divided parameters         : {}\n".format(all_param_size_div)
            info_str += "Flat buffer size           : {} (Padding : {})\n".format(self.flat_buffer_size_div, self.flat_buffer_size_div - all_param_size_div)
            info_str += "Shard size                 : {}\n".format(self.shard_size_div)
            info_str += "Chunk size limit           : {}\n".format(self.chunk_size_limit)
            info_str += "Chunk size                 : {}\n".format(self.chunk_size_div)
            info_str += "Number of chunks           : {}\n".format(self.num_chunks_div)
            info_str += "Sub chunk size             : {}\n".format(self.sub_chunk_size_div)
            info_str += "Number of shards           : {}\n".format(self.data_parallelism)
            info_str += "********************************************************************************"
            print(info_str, flush=True) 

        # Flat buffers
        self.flat_param_rep = torch.empty(self.flat_buffer_size_rep, dtype=dtype, device=device)
        self.flat_grad_rep = torch.zeros(self.flat_buffer_size_rep, dtype=self.dtype, device=self.device)

        self.flat_param_div = torch.empty(self.flat_buffer_size_div, dtype=self.dtype, device=self.device)
        self.flat_grad_div = torch.zeros(self.flat_buffer_size_div, dtype=self.dtype, device=self.device)

        if skip_optimizer_step == False:
            self.flat_exp_avg_shard_rep = torch.zeros(self.shard_size_rep, dtype=torch.float32, device=self.device)
            self.flat_exp_avg_sq_shard_rep = torch.zeros(self.shard_size_rep, dtype=torch.float32, device=self.device)
            self.flat_fp32_master_param_shard_rep = torch.zeros(self.shard_size_rep, dtype=torch.float32, device=self.device)

            self.flat_exp_avg_shard_div = torch.zeros(self.shard_size_div, dtype=torch.float32, device=self.device)
            self.flat_exp_avg_sq_shard_div = torch.zeros(self.shard_size_div, dtype=torch.float32, device=self.device)
            self.flat_fp32_master_param_shard_div = torch.zeros(self.shard_size_div, dtype=torch.float32, device=self.device)
        else:
            self.flat_exp_avg_shard_rep = None
            self.flat_exp_avg_sq_shard_rep = None
            self.flat_fp32_master_param_shard_rep = None

            self.flat_exp_avg_shard_div = None
            self.flat_exp_avg_sq_shard_div = None
            self.flat_fp32_master_param_shard_div = None
        
        # Mapping parameter into flat parameter. (Replicated)
        for param_id, p in enumerate(param_list_rep):
            start_ind = param_start_inds_rep[param_id]
            end_ind = param_start_inds_rep[param_id] + param_sizes_rep[param_id]
            # Parameter
            p.data = self.flat_param_rep[start_ind:end_ind].view_as(p.data).copy_(p.data) # Override
            # Gradient
            p.grad = self.flat_grad_rep[start_ind:end_ind].view_as(p.data)
        
        # Mapping parameter into flat parameter. (Divided)
        for param_id, p in enumerate(param_list_div):
            start_ind = param_start_inds_div[param_id]
            end_ind = param_start_inds_div[param_id] + param_sizes_div[param_id]
            # Parameter
            p.data = self.flat_param_div[start_ind:end_ind].view_as(p.data).copy_(p.data) # Override
            # Gradient
            p.grad = self.flat_grad_div[start_ind:end_ind].view_as(p.data)

        if skip_optimizer_step == False:
            # Setting master parameter (Replicated)
            for chunk_id in range(self.num_chunks_rep):
                flat_param_chunk = self.flat_param_rep[chunk_id*self.chunk_size_rep:(chunk_id+1)*self.chunk_size_rep]
                flat_param_shard_sub_chunk = flat_param_chunk[self.dpep_ind*self.sub_chunk_size_rep:(self.dpep_ind+1)*self.sub_chunk_size_rep]
                flat_fp32_master_param_shard_sub_chunk = self.flat_fp32_master_param_shard_rep[chunk_id*self.sub_chunk_size_rep:(chunk_id+1)*self.sub_chunk_size_rep]
                flat_fp32_master_param_shard_sub_chunk.copy_(flat_param_shard_sub_chunk)

            # Setting master parameter (Divided)
            for chunk_id in range(self.num_chunks_div):
                flat_param_chunk = self.flat_param_div[chunk_id*self.chunk_size_div:(chunk_id+1)*self.chunk_size_div]
                flat_param_shard_sub_chunk = flat_param_chunk[self.dp_ind*self.sub_chunk_size_div:(self.dp_ind+1)*self.sub_chunk_size_div]
                flat_fp32_master_param_shard_sub_chunk = self.flat_fp32_master_param_shard_div[chunk_id*self.sub_chunk_size_div:(chunk_id+1)*self.sub_chunk_size_div]
                flat_fp32_master_param_shard_sub_chunk.copy_(flat_param_shard_sub_chunk)

        # Handy lists for clean code (rep)
        lists_rep = self.get_lists_dict(self.flat_param_rep, self.flat_grad_rep,
                self.flat_exp_avg_shard_rep, self.flat_exp_avg_sq_shard_rep, self.flat_fp32_master_param_shard_rep, 
                self.chunk_size_rep, self.sub_chunk_size_rep, self.num_chunks_rep, self.dpep_ind)
        self.flat_param_chunks_rep = lists_rep["flat_param_chunks"]
        self.flat_grad_chunks_rep = lists_rep["flat_grad_chunks"]
        self.flat_param_shard_sub_chunks_rep = lists_rep["flat_param_shard_sub_chunks"]
        self.flat_grad_shard_sub_chunks_rep  = lists_rep["flat_grad_shard_sub_chunks"]
        self.flat_exp_avg_shard_sub_chunks_rep = lists_rep["flat_exp_avg_shard_sub_chunks"]
        self.flat_exp_avg_sq_shard_sub_chunks_rep = lists_rep["flat_exp_avg_sq_shard_sub_chunks"]
        self.flat_fp32_master_param_shard_sub_chunks_rep = lists_rep["flat_fp32_master_param_shard_sub_chunks"]

        # Handy lists for clean code (div)
        lists_div = self.get_lists_dict(self.flat_param_div, self.flat_grad_div,
                self.flat_exp_avg_shard_div, self.flat_exp_avg_sq_shard_div, self.flat_fp32_master_param_shard_div, 
                self.chunk_size_div, self.sub_chunk_size_div, self.num_chunks_div, self.dp_ind)
        self.flat_param_chunks_div = lists_div["flat_param_chunks"]
        self.flat_grad_chunks_div = lists_div["flat_grad_chunks"]
        self.flat_param_shard_sub_chunks_div = lists_div["flat_param_shard_sub_chunks"]
        self.flat_grad_shard_sub_chunks_div  = lists_div["flat_grad_shard_sub_chunks"]
        self.flat_exp_avg_shard_sub_chunks_div = lists_div["flat_exp_avg_shard_sub_chunks"]
        self.flat_exp_avg_sq_shard_sub_chunks_div = lists_div["flat_exp_avg_sq_shard_sub_chunks"]
        self.flat_fp32_master_param_shard_sub_chunks_div = lists_div["flat_fp32_master_param_shard_sub_chunks"]

    def get_lists_dict(self, flat_param, flat_grad, 
                flat_exp_avg_shard, flat_exp_avg_sq_shard, flat_fp32_master_param_shard, 
                chunk_size, sub_chunk_size, num_chunks, shard_id):
            flat_param_chunks = [flat_param[chunk_id*chunk_size:(chunk_id+1)*chunk_size] for chunk_id in range(num_chunks)]
            flat_grad_chunks = [flat_grad[chunk_id*chunk_size:(chunk_id+1)*chunk_size] for chunk_id in range(num_chunks)]
            flat_param_shard_sub_chunks = [flat_param_chunks[chunk_id][shard_id*sub_chunk_size:(shard_id+1)*sub_chunk_size] for chunk_id in range(num_chunks)]
            flat_grad_shard_sub_chunks  = [ flat_grad_chunks[chunk_id][shard_id*sub_chunk_size:(shard_id+1)*sub_chunk_size] for chunk_id in range(num_chunks)]
            flat_exp_avg_shard_sub_chunks = [flat_exp_avg_shard[chunk_id*sub_chunk_size:(chunk_id+1)*sub_chunk_size] for chunk_id in range(num_chunks)]
            flat_exp_avg_sq_shard_sub_chunks = [flat_exp_avg_sq_shard[chunk_id*sub_chunk_size:(chunk_id+1)*sub_chunk_size] for chunk_id in range(num_chunks)]
            flat_fp32_master_param_shard_sub_chunks = [flat_fp32_master_param_shard[chunk_id*sub_chunk_size:(chunk_id+1)*sub_chunk_size] for chunk_id in range(num_chunks)]

            # Packing as lists_dict
            lists_dict = {}
            lists_dict["flat_param_chunks"] = flat_param_chunks
            lists_dict["flat_grad_chunks"] = flat_grad_chunks
            lists_dict["flat_param_shard_sub_chunks"] = flat_param_shard_sub_chunks
            lists_dict["flat_grad_shard_sub_chunks"] = flat_grad_shard_sub_chunks
            lists_dict["flat_exp_avg_shard_sub_chunks"] = flat_exp_avg_shard_sub_chunks
            lists_dict["flat_exp_avg_sq_shard_sub_chunks"] = flat_exp_avg_sq_shard_sub_chunks
            lists_dict["flat_fp32_master_param_shard_sub_chunks"] = flat_fp32_master_param_shard_sub_chunks

            return lists_dict

    def zero_grad(self):
        self.flat_grad_rep.zero_()
        self.flat_grad_div.zero_()

    def step(self, closure=None):
        with record_pcl_function("4opt-step_1gradacc"):
            # Replicated grad_acc (DP+EP)
            for chunk_id in range(self.num_chunks_rep):
                if torch.distributed.is_initialized() and ((self.data_parallelism > 1) or (self.expert_parallelism > 1)):
                    if not self.opt_use_allreduce_for_grad_acc:
                        flat_grad_chunk = self.flat_grad_chunks_rep[chunk_id]
                        flat_grad_shard_sub_chunk = self.flat_grad_shard_sub_chunks_rep[chunk_id]
                        if not self.opt_use_fp32_for_grad_acc:    
                            with record_pcl_function("3so-comms--2reduce_scatter--" + get_comms_profile_info_string(flat_grad_chunk, self.dpep_group)):
                                torch.distributed.reduce_scatter_tensor(flat_grad_shard_sub_chunk, flat_grad_chunk, group=self.dpep_group)
                        else:
                            with record_pcl_function("4opt-step_6fp32ga-precomms"):
                                flat_grad_chunk_fp32 = flat_grad_chunk.to(torch.float32)
                                flat_grad_shard_sub_chunk_fp32 = flat_grad_chunk_fp32[self.dpep_ind*self.sub_chunk_size_rep:(self.dpep_ind+1)*self.sub_chunk_size_rep]
                            with record_pcl_function("3so-comms--2reduce_scatter--" + get_comms_profile_info_string(flat_grad_chunk_fp32, self.dpep_group)):
                                torch.distributed.reduce_scatter_tensor(flat_grad_shard_sub_chunk_fp32, flat_grad_chunk_fp32, group=self.dpep_group)
                            with record_pcl_function("4opt-step_7fp32ga-postcomms"):
                                flat_grad_shard_sub_chunk.copy_(flat_grad_shard_sub_chunk_fp32)

                        flat_grad_shard_sub_chunk.div_((self.data_parallelism * self.expert_parallelism))
                    else:
                        flat_grad_chunk = self.flat_grad_chunks_rep[chunk_id]
                        if not self.opt_use_fp32_for_grad_acc:
                            with record_pcl_function("3so-comms--1allreduce--" + get_comms_profile_info_string(flat_grad_chunk, self.dpep_group)):
                                torch.distributed.all_reduce(flat_grad_chunk, group=self.dpep_group)
                        else:
                            with record_pcl_function("4opt-step_6fp32ga-precomms"):
                                flat_grad_chunk_fp32 = flat_grad_chunk.to(torch.float32)
                            with record_pcl_function("3so-comms--1allreduce--" + get_comms_profile_info_string(flat_grad_chunk_fp32, self.dpep_group)):
                                torch.distributed.all_reduce(flat_grad_chunk_fp32, group=self.dpep_group)
                            with record_pcl_function("4opt-step_7fp32ga-postcomms"):
                                flat_grad_chunk.copy_(flat_grad_chunk_fp32)

                        flat_grad_chunk.div_((self.data_parallelism * self.expert_parallelism))

            # Divided grad_acc (DP)
            for chunk_id in range(self.num_chunks_div):
                if torch.distributed.is_initialized() and (self.data_parallelism > 1):
                    if not self.opt_use_allreduce_for_grad_acc:
                        flat_grad_chunk = self.flat_grad_chunks_div[chunk_id]
                        flat_grad_shard_sub_chunk = self.flat_grad_shard_sub_chunks_div[chunk_id]
                        if not self.opt_use_fp32_for_grad_acc:
                            with record_pcl_function("3so-comms--2reduce_scatter--" + get_comms_profile_info_string(flat_grad_chunk, self.dp_group)):
                                torch.distributed.reduce_scatter_tensor(flat_grad_shard_sub_chunk, flat_grad_chunk, group=self.dp_group)
                        else:
                            with record_pcl_function("4opt-step_6fp32ga-precomms"):
                                flat_grad_chunk_fp32 = flat_grad_chunk.to(torch.float32)
                                flat_grad_shard_sub_chunk_fp32 = flat_grad_chunk_fp32[self.dp_ind*self.sub_chunk_size_div:(self.dp_ind+1)*self.sub_chunk_size_div]
                            with record_pcl_function("3so-comms--2reduce_scatter--" + get_comms_profile_info_string(flat_grad_chunk_fp32, self.dp_group)):
                                torch.distributed.reduce_scatter_tensor(flat_grad_shard_sub_chunk_fp32, flat_grad_chunk_fp32, group=self.dp_group)
                            with record_pcl_function("4opt-step_7fp32ga-postcomms"):
                                flat_grad_shard_sub_chunk.copy_(flat_grad_shard_sub_chunk_fp32)

                        flat_grad_shard_sub_chunk.div_(self.data_parallelism)
                    else:
                        flat_grad_chunk = self.flat_grad_chunks_div[chunk_id]
                        if not self.opt_use_fp32_for_grad_acc:
                            with record_pcl_function("3so-comms--1allreduce--" + get_comms_profile_info_string(flat_grad_chunk, self.dp_group)):
                                torch.distributed.all_reduce(flat_grad_chunk, group=self.dp_group)
                        else:
                            with record_pcl_function("4opt-step_6fp32ga-precomms"):
                                flat_grad_chunk_fp32 = flat_grad_chunk.to(torch.float32)
                            with record_pcl_function("3so-comms--1allreduce--" + get_comms_profile_info_string(flat_grad_chunk_fp32, self.dp_group)):
                                torch.distributed.all_reduce(flat_grad_chunk_fp32, group=self.dp_group)
                            with record_pcl_function("4opt-step_7fp32ga-postcomms"):
                                flat_grad_chunk.copy_(flat_grad_chunk_fp32)
                        flat_grad_chunk.div_(self.data_parallelism)

        # Global gradient norm
        with record_pcl_function("4opt-step_3global-gradnorm"):
            grad_shard_norm_rep = torch.nn.utils.get_total_norm(self.flat_grad_shard_sub_chunks_rep)
            grad_shard_norm_div = torch.nn.utils.get_total_norm(self.flat_grad_shard_sub_chunks_div)
            if torch.distributed.is_initialized():
                grad_shard_norm_rep_fp32 = grad_shard_norm_rep.to(torch.float32)
                grad_shard_norm_sq_rep_fp32 = torch.pow(grad_shard_norm_rep_fp32, 2)
                torch.distributed.all_reduce(grad_shard_norm_sq_rep_fp32)

                grad_shard_norm_div_fp32 = grad_shard_norm_div.to(torch.float32)
                grad_shard_norm_sq_div_fp32 = torch.pow(grad_shard_norm_div_fp32, 2)
                torch.distributed.all_reduce(grad_shard_norm_sq_div_fp32)

                grad_shard_norm_sq_fp32 = grad_shard_norm_sq_rep_fp32 + grad_shard_norm_sq_div_fp32
                grad_norm = torch.sqrt(grad_shard_norm_sq_fp32)
            else:
                grad_norm = torch.nn.utils.get_total_norm(self.flat_grad_shard_sub_chunks_rep + self.flat_grad_shard_sub_chunks_div)

        # Gradient clipping
        with record_pcl_function("4opt-step_4clip-grads"):
            if self.opt_disable_delayed_grad_clipping or \
                ((self.opt_disable_delayed_grad_clipping == False) and (self.step_id >= self.warmup_steps)):
                max_norm = 1.0
                use_custom_grad_clipping = True
                if not use_custom_grad_clipping:
                    torch.nn.utils.clip_grads_with_norm_(self.model.parameters(), max_norm, grad_norm)
                else:
                    clip_coef = max_norm / (grad_norm + 1e-6)  # avoid divide-by-zero
                    if clip_coef < 1.0:
                        for chunk_id in range(self.num_chunks_rep):
                            self.flat_grad_shard_sub_chunks_rep[chunk_id].mul_(clip_coef)                    
                        for chunk_id in range(self.num_chunks_div):
                            self.flat_grad_shard_sub_chunks_div[chunk_id].mul_(clip_coef)

        #############################################################################################################
        # Optimizer compute (Replicated)
        for chunk_id in range(self.num_chunks_rep):
            # Getting chunks
            flat_param_chunk                = self.flat_param_chunks_rep[chunk_id]
            flat_param_shard_sub_chunk      = self.flat_param_shard_sub_chunks_rep[chunk_id]
            flat_grad_shard_sub_chunk       = self.flat_grad_shard_sub_chunks_rep[chunk_id]
            flat_exp_avg_shard_sub_chunk    = self.flat_exp_avg_shard_sub_chunks_rep[chunk_id]
            flat_exp_avg_sq_shard_sub_chunk = self.flat_exp_avg_sq_shard_sub_chunks_rep[chunk_id]
            flat_fp32_master_param_shard_sub_chunk = self.flat_fp32_master_param_shard_sub_chunks_rep[chunk_id]

            with record_pcl_function("4opt-step_5compute"):
                # Hyper parameters
                beta1, beta2 = self.defaults["betas"]
                eps = self.defaults["eps"]
                lr = self.param_groups[0]["lr"]
                step_size = lr
                weight_decay = self.defaults["weight_decay"]    

                # Using FP32 gradient for optimizer step
                flat_grad_shard_sub_chunk_fp32 = flat_grad_shard_sub_chunk.to(torch.float32)

                # Optimizer compute
                flat_exp_avg_shard_sub_chunk.mul_(beta1).add_(flat_grad_shard_sub_chunk_fp32, alpha=(1.0 - beta1))
                flat_exp_avg_sq_shard_sub_chunk.mul_(beta2).addcmul_(flat_grad_shard_sub_chunk_fp32, flat_grad_shard_sub_chunk_fp32, value=1.0 - beta2)
                flat_exp_avg_hat_sub_chunk = flat_exp_avg_shard_sub_chunk / (1.0 - math.pow(beta1, (self.step_id+1))) 
                flat_exp_avg_sq_hat_sub_chunk = flat_exp_avg_sq_shard_sub_chunk / (1.0 - math.pow(beta2, (self.step_id+1)))
                denom_sub_chunk = flat_exp_avg_sq_hat_sub_chunk.sqrt().add_(eps)
                flat_fp32_master_param_shard_sub_chunk.addcdiv_(flat_exp_avg_hat_sub_chunk, denom_sub_chunk, value=-step_size)

                # Handling weight decay
                if weight_decay > 0.0:
                    flat_fp32_master_param_shard_sub_chunk.add_(flat_fp32_master_param_shard_sub_chunk, alpha=(-lr * weight_decay)) #[data = data - (lr * weight_decay)]

                if self.opt_use_allreduce_for_param_gather:
                    flat_param_chunk.mul_(0.0) # Reset parameters to zero
                
                # Convert master weights to model weights dtype
                flat_param_shard_sub_chunk.copy_(flat_fp32_master_param_shard_sub_chunk)

            # Param gather (Replicated)
            if self.opt_alternate_compute_and_comms:
                with record_pcl_function("4opt-step_2paramgather"):
                    if self.opt_use_allreduce_for_param_gather:
                        if torch.distributed.is_initialized() and ((self.data_parallelism > 1) or (self.expert_parallelism > 1)):
                            with record_pcl_function("3so-comms--1allreduce--"+get_comms_profile_info_string(flat_param_chunk, self.dpep_group)):
                                torch.distributed.all_reduce(flat_param_chunk, group=self.dpep_group)
                    else:
                        if torch.distributed.is_initialized() and ((self.data_parallelism > 1) or (self.expert_parallelism > 1)):
                            with record_pcl_function("3so-comms--3allgather--"+get_comms_profile_info_string(flat_param_chunk, self.dpep_group)):
                                torch.distributed.all_gather_into_tensor(flat_param_chunk, flat_param_shard_sub_chunk, group=self.dpep_group)

        # Param gather (Replicated)
        if not self.opt_alternate_compute_and_comms:
            with record_pcl_function("4opt-step_2paramgather"):
                for chunk_id in range(self.num_chunks_rep):
                    flat_param_chunk = self.flat_param_chunks_rep[chunk_id]
                    flat_param_shard_sub_chunk = self.flat_param_shard_sub_chunks_rep[chunk_id]
                    if self.opt_use_allreduce_for_param_gather:
                        if torch.distributed.is_initialized() and ((self.data_parallelism > 1) or (self.expert_parallelism > 1)):
                            with record_pcl_function("3so-comms--1allreduce--"+get_comms_profile_info_string(flat_param_chunk, self.dpep_group)):
                                torch.distributed.all_reduce(flat_param_chunk, group=self.dpep_group)
                    else:
                        if torch.distributed.is_initialized() and ((self.data_parallelism > 1) or (self.expert_parallelism > 1)):
                            with record_pcl_function("3so-comms--3allgather--"+get_comms_profile_info_string(flat_param_chunk, self.dpep_group)):
                                torch.distributed.all_gather_into_tensor(flat_param_chunk, flat_param_shard_sub_chunk, group=self.dpep_group)
        #############################################################################################################################

        #############################################################################################################
        # Optimizer compute (Divided)
        for chunk_id in range(self.num_chunks_div):
            # Getting chunks
            flat_param_chunk                = self.flat_param_chunks_div[chunk_id]
            flat_param_shard_sub_chunk      = self.flat_param_shard_sub_chunks_div[chunk_id]
            flat_grad_shard_sub_chunk       = self.flat_grad_shard_sub_chunks_div[chunk_id]
            flat_exp_avg_shard_sub_chunk    = self.flat_exp_avg_shard_sub_chunks_div[chunk_id]
            flat_exp_avg_sq_shard_sub_chunk = self.flat_exp_avg_sq_shard_sub_chunks_div[chunk_id]
            flat_fp32_master_param_shard_sub_chunk = self.flat_fp32_master_param_shard_sub_chunks_div[chunk_id]

            with record_pcl_function("4opt-step_5compute"):
                # Hyper parameters
                beta1, beta2 = self.defaults["betas"]
                eps = self.defaults["eps"]
                lr = self.param_groups[0]["lr"]
                step_size = lr
                weight_decay = self.defaults["weight_decay"]

                # Using FP32 gradient for optimizer step
                flat_grad_shard_sub_chunk_fp32 = flat_grad_shard_sub_chunk.to(torch.float32)

                # Optimizer compute
                flat_exp_avg_shard_sub_chunk.mul_(beta1).add_(flat_grad_shard_sub_chunk_fp32, alpha=(1.0 - beta1))
                flat_exp_avg_sq_shard_sub_chunk.mul_(beta2).addcmul_(flat_grad_shard_sub_chunk_fp32, flat_grad_shard_sub_chunk_fp32, value=1.0 - beta2)
                flat_exp_avg_hat_sub_chunk = flat_exp_avg_shard_sub_chunk / (1.0 - math.pow(beta1, (self.step_id+1))) 
                flat_exp_avg_sq_hat_sub_chunk = flat_exp_avg_sq_shard_sub_chunk / (1.0 - math.pow(beta2, (self.step_id+1)))
                denom_sub_chunk = flat_exp_avg_sq_hat_sub_chunk.sqrt().add_(eps)
                flat_fp32_master_param_shard_sub_chunk.addcdiv_(flat_exp_avg_hat_sub_chunk, denom_sub_chunk, value=-step_size)

                # Handling weight decay
                if weight_decay > 0.0:
                    flat_fp32_master_param_shard_sub_chunk.add_(flat_fp32_master_param_shard_sub_chunk, alpha=(-lr * weight_decay)) #[data = data - (lr * weight_decay)]

                if self.opt_use_allreduce_for_param_gather:
                    flat_param_chunk.mul_(0.0) # Reset parameters to zero
                
                # Convert master weights to model weights dtype
                flat_param_shard_sub_chunk.copy_(flat_fp32_master_param_shard_sub_chunk)

            # Param gather (Divided)
            if self.opt_alternate_compute_and_comms:
                with record_pcl_function("4opt-step_2paramgather"):
                    if self.opt_use_allreduce_for_param_gather:
                        if torch.distributed.is_initialized() and (self.data_parallelism > 1):
                            with record_pcl_function("3so-comms--1allreduce--"+get_comms_profile_info_string(flat_param_chunk, self.dp_group)):
                                torch.distributed.all_reduce(flat_param_chunk, group=self.dp_group)
                    else:
                        if torch.distributed.is_initialized() and (self.data_parallelism > 1):
                            with record_pcl_function("3so-comms--3allgather--"+get_comms_profile_info_string(flat_param_chunk, self.dp_group)):
                                torch.distributed.all_gather_into_tensor(flat_param_chunk, flat_param_shard_sub_chunk, group=self.dp_group)

        # Param gather (Divided)
        if not self.opt_alternate_compute_and_comms:
            with record_pcl_function("4opt-step_2paramgather"):
                for chunk_id in range(self.num_chunks_div):
                    flat_param_chunk = self.flat_param_chunks_div[chunk_id]
                    flat_param_shard_sub_chunk = self.flat_param_shard_sub_chunks_div[chunk_id]
                    if self.opt_use_allreduce_for_param_gather:
                        if torch.distributed.is_initialized() and (self.data_parallelism > 1):
                            with record_pcl_function("3so-comms--1allreduce--"+get_comms_profile_info_string(flat_param_chunk, self.dp_group)):
                                torch.distributed.all_reduce(flat_param_chunk, group=self.dp_group)
                    else:
                        if torch.distributed.is_initialized() and (self.data_parallelism > 1):
                            with record_pcl_function("3so-comms--3allgather--"+get_comms_profile_info_string(flat_param_chunk, self.dp_group)):
                                torch.distributed.all_gather_into_tensor(flat_param_chunk, flat_param_shard_sub_chunk, group=self.dp_group)
        #############################################################################################################################

        # Update step
        self.step_id += 1

        return grad_norm

    def state_dict(self):
        opt_state_dict = {
            "flat_exp_avg_shard_replicated" : self.flat_exp_avg_shard_rep,
            "flat_exp_avg_sq_shard_replicated" : self.flat_exp_avg_sq_shard_rep,
            "flat_fp32_master_param_shard_replicated" : self.flat_fp32_master_param_shard_rep,
            "flat_exp_avg_shard_divided" : self.flat_exp_avg_shard_div,
            "flat_exp_avg_sq_shard_divided" : self.flat_exp_avg_sq_shard_div,
            "flat_fp32_master_param_shard_divided" : self.flat_fp32_master_param_shard_div,
            "step_id" : self.step_id
        }
        return opt_state_dict
    
    def load_state_dict(self, opt_state_dict):
        self.flat_exp_avg_shard_rep.copy_(opt_state_dict["flat_exp_avg_shard_replicated"])
        self.flat_exp_avg_sq_shard_rep.copy_(opt_state_dict["flat_exp_avg_sq_shard_replicated"])
        self.flat_fp32_master_param_shard_rep.copy_(opt_state_dict["flat_fp32_master_param_shard_replicated"])
        self.flat_exp_avg_shard_div.copy_(opt_state_dict["flat_exp_avg_shard_divided"])
        self.flat_exp_avg_sq_shard_div.copy_(opt_state_dict["flat_exp_avg_sq_shard_divided"])
        self.flat_fp32_master_param_shard_div.copy_(opt_state_dict["flat_fp32_master_param_shard_divided"])
        self.step_id = opt_state_dict["step_id"]


"""
# AdamW Computation
exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1)) # [exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad]
exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2) # [exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad]
exp_avg_hat = exp_avg / (1.0 - math.pow(beta1, (self.step_id+1))) 
exp_avg_sq_hat = exp_avg_sq / (1.0 - math.pow(beta2, (self.step_id+1)))
denom = exp_avg_sq_hat.sqrt().add_(eps) # [(torch.sqrt(exp_avg_sq) + eps))]
data_master.addcdiv_(exp_avg_hat.to(torch.float32), denom.to(torch.float32), value=-step_size) # [data = data - step_size * (exp_avg / (torch.sqrt(exp_avg_sq) + eps))]

# Handling weight decay
if weight_decay > 0.0:
    data_master.add_(data_master, alpha=(-lr * weight_decay)) #[data = data - (lr * weight_decay)]

# Convert master weights to model weights dtype
data.copy_(data_master.to(data.dtype))
"""

"""
Constraints:
1. A parameter has to be contiguous in memory.

Notation

all_param_size   : Sum of sizes of all parameters
flat_buffer_size : all_param_size + padding
"""