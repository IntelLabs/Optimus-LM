import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_activation_checkpoint

from ...profilers import record_pcl_function

from .comms_ptfunctions import forward_allgather_backward_split
from .comms_ptfunctions import forward_identity_backward_allreduce
from .comms_ptfunctions import forward_reducescatter_backward_allgather
from .comms_ptfunctions import forward_identity_backward_synchronize
from .comms_ptfunctions import forward_synchronize_backward_identity
from .comms_ptfunctions import forward_allgather_backward_reducescatter

from ..olmoe.modeling_olmoe import OlmoeParallelMLP

from ...kernels.grouped_linear_forward import triton_grouped_linear_forward
from ...kernels.grouped_linear_backward_input import triton_grouped_linear_backward_input
from ...kernels.grouped_linear_backward_input_add import triton_grouped_linear_backward_input_add
from ...kernels.grouped_linear_backward_weight import triton_grouped_linear_backward_weight

from ...kernels.merged_mlp_triton_kernels import triton_grouped_linear_forward_wrapper
from ...kernels.merged_mlp_triton_kernels import triton_grouped_linear_backward_input_wrapper
from ...kernels.merged_mlp_triton_kernels import triton_grouped_linear_backward_weight_wrapper


def get_tensor_list(tensor, cum_token_counts):
    N = len(cum_token_counts)-1
    tensor_list = []
    for n in range(N):
        start_index = cum_token_counts[n]
        end_index = cum_token_counts[n+1]
        tensor_list.append(tensor[start_index:end_index])
    return tensor_list

@torch.compile
def silu_backward_grad_input(x, grad_output):
    sig = torch.sigmoid(x)
    grad_input = grad_output * (sig * (1 + x * (1 - sig)))
    return grad_input

class GroupMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, cum_token_counts, use_triton_path, *weight_tuple):
        with record_pcl_function("1moeblock--7gmlp_fwd"):
            # Short hands
            N = len(cum_token_counts)-1
            T,E = input.shape
            I = weight_tuple[0].shape[0]

            # Activation tensors
            output_gproj   = torch.empty((T, I), dtype=input.dtype, device=input.device)
            output_uproj   = torch.empty((T, I), dtype=input.dtype, device=input.device)
            output_mul     = torch.empty((T, I), dtype=input.dtype, device=input.device)
            output         = torch.empty((T, E), dtype=input.dtype, device=input.device)

            # Tensor lists for convenience
            input_list = get_tensor_list(input, cum_token_counts)
            output_gproj_list = get_tensor_list(output_gproj, cum_token_counts)
            output_uproj_list = get_tensor_list(output_uproj, cum_token_counts)
            output_mul_list = get_tensor_list(output_mul, cum_token_counts)
            output_list = get_tensor_list(output, cum_token_counts)

            if not use_triton_path:
                # Gate and Up proj
                for n in range(N):
                    torch.matmul(input_list[n], weight_tuple[3*n].t(), out=output_gproj_list[n])
                    torch.matmul(input_list[n], weight_tuple[3*n+1].t(), out=output_uproj_list[n])
            else:
                triton_grouped_linear_forward(input, [weight_tuple[3*n] for n in range(N)], output_gproj, cum_token_counts)
                triton_grouped_linear_forward(input, [weight_tuple[3*n+1] for n in range(N)], output_uproj, cum_token_counts)

            # Silu
            output_silu = torch.nn.functional.silu(output_gproj) # out= is not supported

            # Mul
            torch.mul(output_silu, output_uproj, out=output_mul)

            # Down proj
            if not use_triton_path:
                for n in range(N):
                    torch.matmul(output_mul_list[n], weight_tuple[3*n+2].t(), out=output_list[n])
            else:
                triton_grouped_linear_forward(output_mul, [weight_tuple[3*n+2] for n in range(N)], output, cum_token_counts)

            ctx.save_for_backward(input, output_gproj, output_silu, output_uproj, output_mul, cum_token_counts, *weight_tuple)
            ctx.triton_path = use_triton_path
        return output
        
    @staticmethod
    def backward(ctx, output_grad):
        with record_pcl_function("1moeblock--7gmlp_bwd"):
            input, output_gproj, output_silu, output_uproj, output_mul, cum_token_counts, *weight_tuple = ctx.saved_tensors
            use_triton_path = ctx.triton_path

            # Short hands
            N = len(cum_token_counts)-1
            T,E = input.shape
            I = weight_tuple[0].shape[0]
            
            # Allocation
            output_mul_grad = torch.empty_like(output_mul)
            output_uproj_grad = torch.empty_like(output_uproj)
            output_silu_grad = torch.empty_like(output_silu)
            output_gproj_grad = torch.empty_like(output_gproj)
            input_grad = torch.empty_like(input)
            
            weight_grad_list = []
            for n in range(N):
                weight_grad_list.append(torch.empty_like(weight_tuple[3*n]))
                weight_grad_list.append(torch.empty_like(weight_tuple[3*n+1]))
                weight_grad_list.append(torch.empty_like(weight_tuple[3*n+2]))

            # Tensor lists for convenience
            output_mul_list = get_tensor_list(output_mul, cum_token_counts)
            output_grad_list = get_tensor_list(output_grad, cum_token_counts)
            output_mul_grad_list = get_tensor_list(output_mul_grad, cum_token_counts)

            input_list = get_tensor_list(input, cum_token_counts)
            output_uproj_grad_list = get_tensor_list(output_uproj_grad, cum_token_counts)
            input_grad_list = get_tensor_list(input_grad, cum_token_counts)            

            # DownProj gradients
            if not use_triton_path:
                for n in range(N):
                    torch.matmul(output_grad_list[n], weight_tuple[3*n+2], out=output_mul_grad_list[n])
                    torch.matmul(output_grad_list[n].t(), output_mul_list[n], out=weight_grad_list[3*n+2])
            else:
                triton_grouped_linear_backward_input(output_grad, [weight_tuple[3*n+2] for n in range(N)], output_mul_grad, cum_token_counts)
                triton_grouped_linear_backward_weight(output_grad, output_mul, [weight_grad_list[3*n+2] for n in range(N)], cum_token_counts)

            # Mul gradients
            torch.mul(output_mul_grad, output_silu, out=output_uproj_grad)
            torch.mul(output_mul_grad, output_uproj, out=output_silu_grad)

            # UpProj gradients
            if not use_triton_path:
                for n in range(N):
                    torch.matmul(output_uproj_grad_list[n], weight_tuple[3*n+1], out=input_grad_list[n])
                    torch.matmul(output_uproj_grad_list[n].t(), input_list[n], out=weight_grad_list[3*n+1])
            else:
                triton_grouped_linear_backward_input(output_uproj_grad, [weight_tuple[3*n+1] for n in range(N)], input_grad, cum_token_counts)
                triton_grouped_linear_backward_weight(output_uproj_grad, input, [weight_grad_list[3*n+1] for n in range(N)], cum_token_counts)

            # Silu gradient
            output_gproj_grad = silu_backward_grad_input(output_gproj, output_silu_grad)

            # GateProj gradients
            output_gproj_grad_list = get_tensor_list(output_gproj_grad, cum_token_counts)
            if not use_triton_path:
                for n in range(N):
                    torch.addmm(input_grad_list[n], output_gproj_grad_list[n], weight_tuple[3*n], out=input_grad_list[n])
                    torch.matmul(output_gproj_grad_list[n].t(), input_list[n], out=weight_grad_list[3*n])
            else:
                triton_grouped_linear_backward_input_add(output_gproj_grad, [weight_tuple[3*n] for n in range(N)], input_grad, cum_token_counts)
                triton_grouped_linear_backward_weight(output_gproj_grad, input, [weight_grad_list[3*n] for n in range(N)], cum_token_counts)

            grad_weight_tuple = tuple(weight_grad_list)

        return input_grad, None, None, *grad_weight_tuple

class GroupedGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, merged_weight, cum_token_counts, max_tokens_per_expert, use_triton_path):
        num_tokens = input.shape[0]
        num_experts, ofm, ifm = merged_weight.shape

        # Output allocation
        output = torch.empty((num_tokens, ofm), dtype=input.dtype, device=input.device)

        if not use_triton_path:
            # Tensor lists for convenience
            input_list = get_tensor_list(input, cum_token_counts)
            output_list = get_tensor_list(output, cum_token_counts)
            # For loop over experts
            for n in range(num_experts):
                torch.matmul(input_list[n], merged_weight[n].t(), out=output_list[n])
        else:
            cum_token_counts = cum_token_counts.to(input.device)
            triton_grouped_linear_forward_wrapper(input, merged_weight, output, cum_token_counts, max_tokens_per_expert)

        ctx.save_for_backward(input, merged_weight, cum_token_counts)
        ctx.max_tokens_per_expert = max_tokens_per_expert
        ctx.use_triton_path = use_triton_path

        return output

    @staticmethod
    def backward(ctx, output_grad):
        input, merged_weight, cum_token_counts = ctx.saved_tensors
        max_tokens_per_expert = ctx.max_tokens_per_expert
        use_triton_path = ctx.use_triton_path

        # Memory allocation
        input_grad = torch.empty_like(input)
        merged_weight_grad = torch.empty_like(merged_weight)

        if not use_triton_path:
            # Tensor lists for convenience
            input_list = get_tensor_list(input, cum_token_counts)
            input_grad_list = get_tensor_list(input_grad, cum_token_counts)
            output_grad_list = get_tensor_list(output_grad, cum_token_counts)

            # For loop over experts
            for n in range(merged_weight.shape[0]):
                # Input grad
                torch.matmul(output_grad_list[n], merged_weight[n], out=input_grad_list[n])
                # Weight grad
                torch.matmul(output_grad_list[n].t(), input_list[n], out=merged_weight_grad[n])
        else:
            triton_grouped_linear_backward_input_wrapper(output_grad, merged_weight, input_grad, cum_token_counts, max_tokens_per_expert)
            triton_grouped_linear_backward_weight_wrapper(output_grad, input, merged_weight_grad, cum_token_counts, max_tokens_per_expert)
        
        return input_grad, merged_weight_grad, None, None, None

class PrepareInputForGroupGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
        hidden_states, input_indices, 
        cum_expert_counts_ep, output_indices, 
        use_ref_path_fwd=False, use_ref_path_bwd=False,
        use_triton_path_fwd=False, use_triton_path_bwd=False):

        num_tokens, hidden_dim = hidden_states.shape
        num_routed_tokens = input_indices.shape[0]
        with record_pcl_function("1moeblock--6gmlp_input_prep_fwd"):
            if not use_ref_path_fwd:
                ggemm_input = torch.empty((num_routed_tokens, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
                if use_triton_path_fwd:
                    from optimus.kernels.moe_nongemm import index_gather_triton_kernel
                    index_gather_triton_kernel[num_routed_tokens,](hidden_states, input_indices, ggemm_input, hidden_dim)
                else:
                    import pcl_xpu_customops
                    pcl_xpu_customops.index_gather(hidden_states, input_indices, ggemm_input)
            else:
                ggemm_input = hidden_states[input_indices]
        
        ctx.save_for_backward(cum_expert_counts_ep, output_indices)
        ctx.use_ref_path_bwd = use_ref_path_bwd
        ctx.use_triton_path_bwd = use_triton_path_bwd
        ctx.num_tokens = num_tokens
        ctx.num_routed_tokens = num_routed_tokens
        ctx.hidden_dim = hidden_dim
        
        return ggemm_input

    @staticmethod
    def backward(ctx, ggemm_input_grad):
        cum_expert_counts_ep, output_indices = ctx.saved_tensors
        use_ref_path_bwd = ctx.use_ref_path_bwd
        use_triton_path_bwd = ctx.use_triton_path_bwd
        num_tokens = ctx.num_tokens
        num_routed_tokens = ctx.num_routed_tokens
        hidden_dim = ctx.hidden_dim

        with record_pcl_function("1moeblock--6gmlp_input_prep_bwd"):
            if not use_ref_path_bwd:
                hidden_states_grad = torch.empty((num_tokens, hidden_dim), dtype=ggemm_input_grad.dtype, device=ggemm_input_grad.device)
                if use_triton_path_bwd:
                    from optimus.kernels.moe_nongemm import shuffle_sub_matrix_reduce_triton_kernel
                    shuffle_sub_matrix_reduce_triton_kernel[num_tokens,](
                        ggemm_input_grad, 
                        cum_expert_counts_ep, 
                        output_indices, 
                        hidden_states_grad,
                        hidden_dim)
                else:
                    import pcl_xpu_customops
                    pcl_xpu_customops.shuffle_sub_matrix_reduce(ggemm_input_grad, cum_expert_counts_ep, output_indices, hidden_states_grad)
            else:
                hidden_states_grad = torch.empty((num_tokens, hidden_dim), dtype=ggemm_input_grad.dtype, device=ggemm_input_grad.device)
                T = num_tokens
                E = hidden_dim
                for t in range(T):
                    for e in range(E):
                        acc = 0.0
                        for i in range(cum_expert_counts_ep[t], cum_expert_counts_ep[t+1]):
                            index = output_indices[i]
                            acc += ggemm_input_grad[index, e]
                        hidden_states_grad[t, e] = acc
        return hidden_states_grad, None, None, None, None, None, None, None

class GroupGemmOutputReductionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ggemm_output, routing_weights, cum_expert_counts_ep, selected_experts_indices, input_indices, output_indices, 
        use_ref_path_fwd=False, use_ref_path_bwd=False,
        use_triton_path_fwd=False, use_triton_path_bwd=False):
        ctx.save_for_backward(ggemm_output, routing_weights, cum_expert_counts_ep, selected_experts_indices, input_indices, output_indices)
        ctx.use_ref_path_bwd = use_ref_path_bwd
        ctx.use_triton_path_bwd = use_triton_path_bwd
        num_tokens, num_experts_per_tok = routing_weights.shape
        hidden_dim = ggemm_output.shape[1]
        
        with record_pcl_function("1moeblock--8gmlp_output_reduction_fwd"):
            if not use_ref_path_fwd:
                final_hidden_states = torch.empty((num_tokens, hidden_dim), dtype=ggemm_output.dtype, device=ggemm_output.device)
                if use_triton_path_fwd:
                    from optimus.kernels.moe_nongemm import reduce_ggemm_output_triton_kernel
                    reduce_ggemm_output_triton_kernel[num_tokens,]( 
                        ggemm_output, routing_weights, 
                        cum_expert_counts_ep, selected_experts_indices, 
                        output_indices, final_hidden_states,
                        num_experts_per_tok, hidden_dim)
                else:
                    import pcl_xpu_customops
                    pcl_xpu_customops.reduce_ggemm_output(
                        ggemm_output, routing_weights, 
                        cum_expert_counts_ep, selected_experts_indices, 
                        output_indices, final_hidden_states, False)
            else:
                final_hidden_states = torch.empty((num_tokens, hidden_dim), dtype=ggemm_output.dtype, device=ggemm_output.device)
                T = routing_weights.shape[0]
                K = routing_weights.shape[1]
                E = ggemm_output.shape[1]

                for t in range(T):
                    for e in range(E):
                        acc = 0.0
                        for i in range(cum_expert_counts_ep[t], cum_expert_counts_ep[t+1]):
                            k = selected_experts_indices[i]
                            index = output_indices[i]
                            scaling = routing_weights[t, k]
                            acc += ggemm_output[index, e] * scaling
                        final_hidden_states[t, e] = acc
        
        return final_hidden_states

    @staticmethod
    def backward(ctx, final_hidden_states_grad):
        ggemm_output, routing_weights, cum_expert_counts_ep, selected_experts_indices, input_indices, output_indices = ctx.saved_tensors
        use_ref_path_bwd = ctx.use_ref_path_bwd
        use_triton_path_bwd = ctx.use_triton_path_bwd

        ggemm_output_grad = torch.empty_like(ggemm_output)
        routing_weights_grad = torch.zeros_like(routing_weights)

        with record_pcl_function("1moeblock--8gmlp_output_reduction_bwd"):
            if not use_ref_path_bwd:
                if use_triton_path_bwd:
                    from optimus.kernels.moe_nongemm import reduce_ggemm_output_backward_triton_kernel
                    reduce_ggemm_output_backward_triton_kernel[ggemm_output.shape[0],]( 
                        final_hidden_states_grad,
                        ggemm_output,
                        routing_weights,
                        cum_expert_counts_ep,
                        selected_experts_indices,
                        input_indices,
                        output_indices,
                        ggemm_output_grad,
                        routing_weights_grad,
                        routing_weights.shape[1], final_hidden_states_grad.shape[1])
                else:
                    import pcl_xpu_customops
                    pcl_xpu_customops.reduce_ggemm_output_backward(
                        final_hidden_states_grad,
                        ggemm_output,
                        routing_weights,
                        cum_expert_counts_ep,
                        selected_experts_indices,
                        input_indices,
                        output_indices,
                        ggemm_output_grad,
                        routing_weights_grad)
            else:
                # ggemm_output_grad
                T,E = final_hidden_states_grad.shape[0], final_hidden_states_grad.shape[1]
                K = routing_weights.shape[1]
                final_hidden_states_grad_expand = final_hidden_states_grad[:,None,:].expand(T,K,E)
                scaled_final_hidden_states_grad_expand = routing_weights.unsqueeze(2) * final_hidden_states_grad_expand
                scaled_final_hidden_states_grad_expand = scaled_final_hidden_states_grad_expand.view(T*K, E)
                ggemm_output_grad = scaled_final_hidden_states_grad_expand[output_indices]

                # routing_weights_grad
                ggemm_output_shuffled = ggemm_output[output_indices].view(T, K, E)
                routing_weights_grad = torch.sum(ggemm_output_shuffled * final_hidden_states_grad_expand, dim=2)
        return ggemm_output_grad, routing_weights_grad, None, None, None, None, None, None, None, None

class MergedMLP(nn.Module):
    def __init__(self, config):
        super(MergedMLP, self).__init__()
        self.config = config

        self.num_experts_per_rank = config.num_experts // config.expert_parallelism

        self.gate_proj_merged_weight = torch.nn.Parameter(
            torch.empty((self.num_experts_per_rank, config.intermediate_size, config.hidden_size))
        )
        self.up_proj_merged_weight = torch.nn.Parameter(
            torch.empty((self.num_experts_per_rank, config.intermediate_size, config.hidden_size))
        )
        self.down_proj_merged_weight = torch.nn.Parameter(
            torch.empty((self.num_experts_per_rank, config.hidden_size, config.intermediate_size))
        )

    def forward(self, input, cum_token_counts):
        max_tokens_per_expert = torch.max(cum_token_counts[1:] - cum_token_counts[:-1]).item()
        output_gproj = GroupedGemmFunction.apply(input, self.gate_proj_merged_weight, cum_token_counts, max_tokens_per_expert, self.config.use_triton_path_for_gemm_in_fast_moe)
        output_uproj = GroupedGemmFunction.apply(input, self.up_proj_merged_weight, cum_token_counts, max_tokens_per_expert, self.config.use_triton_path_for_gemm_in_fast_moe)

        output_silu = torch.nn.functional.silu(output_gproj)
        output_mul = output_silu * output_uproj

        output = GroupedGemmFunction.apply(output_mul, self.down_proj_merged_weight, cum_token_counts, max_tokens_per_expert, self.config.use_triton_path_for_gemm_in_fast_moe)
        return output

class FastOlmoeParallelSparseMoeBlock(nn.Module):
    def __init__(self, config, pmap=None):
        super().__init__()
        # Parallelism related
        self.config = config
        self.pmap = pmap
        self.num_experts_per_rank = config.num_experts // config.expert_parallelism
        self.expert_start_idx = self.pmap.ep_ind * self.num_experts_per_rank
        self.expert_end_idx = (self.pmap.ep_ind + 1) * self.num_experts_per_rank

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.output_router_logits = config.output_router_logits

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        if not self.config.use_merged_mlp_in_fast_moe:
            self.experts = nn.ModuleList([OlmoeParallelMLP(config) for _ in range(self.num_experts_per_rank)])
        else:
            self.experts = MergedMLP(config)

        self.use_activation_checkpointing_for_moe = False
        if config.use_activation_checkpointing:
            self.use_activation_checkpointing_for_moe = config.activation_checkpointing_level & (1 << 0) # Bit 0 for MoE
        
        self.mask = None
        self.force_uniform_routing = config.force_uniform_routing

        # Reference path flags
        self.use_ref_path_for_k3 = False
        self.use_ref_path_for_k4_fwd = False
        self.use_ref_path_for_k4_bwd = False
        self.use_ref_path_for_k6_fwd = False
        self.use_ref_path_for_k6_bwd = False

        self.use_triton_path_for_gemm = config.use_triton_path_for_gemm_in_fast_moe
        self.use_triton_path_for_k1 = config.use_triton_path_for_nongemm_in_fast_moe
        self.use_triton_path_for_k3 = config.use_triton_path_for_nongemm_in_fast_moe
        self.use_triton_path_for_k4_fwd = config.use_triton_path_for_nongemm_in_fast_moe
        self.use_triton_path_for_k4_bwd = config.use_triton_path_for_nongemm_in_fast_moe
        self.use_triton_path_for_k6_fwd = config.use_triton_path_for_nongemm_in_fast_moe
        self.use_triton_path_for_k6_bwd = config.use_triton_path_for_nongemm_in_fast_moe

        self.use_allreduce_for_reducescatter = False

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        router_logits = self.gate(hidden_states.view(-1, hidden_dim)) # router_logits: (batch * sequence_length, n_experts)

        ########################################################################
        if self.force_uniform_routing:
            if self.mask is None:
                from optimus.utils import get_load_balanced_expert_mask
                self.mask = get_load_balanced_expert_mask(sequence_length, self.num_experts, self.num_experts_per_tok, self.pmap.ep_ind)
                self.mask = (self.mask == False).to(hidden_states.device)  # Invert the mask to use with `masked_fill`
            router_logits = router_logits.masked_fill(self.mask, float('-inf'))
        ########################################################################

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        def expert_computation_block(hidden_states, selected_experts, routing_weights):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)

            if self.pmap.expert_parallelism > 1:
                hidden_states = forward_allgather_backward_reducescatter.apply(hidden_states, self.pmap.ep_ind, self.pmap.expert_parallelism, self.pmap.ep_group, self.use_allreduce_for_reducescatter)
                routing_weights = forward_allgather_backward_reducescatter.apply(routing_weights, self.pmap.ep_ind, self.pmap.expert_parallelism, self.pmap.ep_group, self.use_allreduce_for_reducescatter)
                selected_experts = forward_allgather_backward_split.apply(selected_experts, self.pmap.ep_ind, self.pmap.expert_parallelism, self.pmap.ep_group)

            if self.pmap.expert_parallelism > 1:
                hidden_states = forward_identity_backward_synchronize.apply(hidden_states, self.pmap.ep_group)

            ##################################################################################################
            tokens_per_thread = 8
            num_tokens = (self.pmap.expert_parallelism * batch_size * sequence_length)
            num_threads = num_tokens // tokens_per_thread
            assert num_tokens % num_threads == 0, "Sequence length must be divisible by num_threads"
            
            # Allocation
            partial_token_counts = torch.zeros((self.num_experts_per_rank, num_threads), dtype=torch.int32, device=hidden_states.device)
            partial_cum_token_counts = torch.zeros(self.num_experts_per_rank * num_threads + 1, dtype=torch.int32, device=hidden_states.device)
            cum_token_counts = torch.empty(self.num_experts_per_rank+1, dtype=torch.int32)
            expert_counts_ep = torch.zeros(num_tokens, dtype=torch.int32, device=hidden_states.device)
            cum_expert_counts_ep = torch.zeros(num_tokens+1, dtype=torch.int32, device=hidden_states.device)
            input_increment_buffer = torch.zeros((self.num_experts_per_rank, num_threads), dtype=torch.int32, device=hidden_states.device)

            # K1
            with record_pcl_function("1moeblock--1partial_token_counting"):
                if self.use_triton_path_for_k1:
                    from optimus.kernels.moe_nongemm import compute_partial_token_counts_triton_kernel
                    compute_partial_token_counts_triton_kernel[num_threads,](
                            selected_experts, partial_token_counts, expert_counts_ep, 
                            self.expert_start_idx, self.expert_end_idx,
                            self.num_experts_per_tok, TB=tokens_per_thread)
                else:
                    import pcl_xpu_customops
                    partial_token_counts = pcl_xpu_customops.compute_partial_token_counts(
                                            selected_experts, partial_token_counts, expert_counts_ep, 
                                            self.expert_start_idx, self.expert_end_idx)
            
            # K2
            with record_pcl_function("1moeblock--2token_counting"):
                torch.cumsum(partial_token_counts.flatten(), dim=0, out=partial_cum_token_counts[1:])
                cum_token_counts.copy_(partial_cum_token_counts[0::num_threads])
                torch.cumsum(expert_counts_ep, dim=0, out=cum_expert_counts_ep[1:])
            
            # Allocation
            num_routed_tokens = cum_token_counts[-1].item()
            selected_experts_indices = torch.empty(num_routed_tokens, dtype=torch.int32, device=selected_experts.device)
            input_indices = torch.empty(num_routed_tokens, dtype=torch.int32, device=hidden_states.device)
            output_indices = torch.empty(num_routed_tokens, dtype=torch.int32, device=hidden_states.device)

            # K3
            with record_pcl_function("1moeblock--3index_calculation"):
                if self.use_triton_path_for_k3:
                    from optimus.kernels.moe_nongemm import compute_input_and_output_indices_triton_kernel
                    compute_input_and_output_indices_triton_kernel[num_threads,](
                        selected_experts,
                        partial_cum_token_counts,
                        input_increment_buffer,
                        cum_expert_counts_ep,
                        selected_experts_indices,
                        input_indices,
                        output_indices,
                        self.expert_start_idx,
                        self.expert_end_idx,
                        self.num_experts_per_tok, 
                        TB=tokens_per_thread
                    )
                else:
                    import pcl_xpu_customops    
                    pcl_xpu_customops.compute_input_and_output_indices(
                        selected_experts,
                        partial_cum_token_counts,
                        input_increment_buffer,
                        cum_expert_counts_ep,
                        selected_experts_indices,
                        input_indices,
                        output_indices,
                        self.expert_start_idx,
                        self.expert_end_idx
                    )

            # K4 (Gather)
            ggemm_input = PrepareInputForGroupGemmFunction.apply(
                hidden_states,
                input_indices,
                cum_expert_counts_ep,
                output_indices,
                self.use_ref_path_for_k4_fwd,
                self.use_ref_path_for_k4_bwd,
                self.use_triton_path_for_k4_fwd,
                self.use_triton_path_for_k4_bwd
            )

            # K5 (Expert blocks)
            if not self.config.use_merged_mlp_in_fast_moe:
                use_pt_function_for_expert_blocks = True
                if use_pt_function_for_expert_blocks:
                    mlp_params = []
                    for expert in self.experts:
                        mlp_params.append(expert.gate_proj.weight)
                        mlp_params.append(expert.up_proj.weight)
                        mlp_params.append(expert.down_proj.weight)
                    ggemm_output = GroupMLPFunction.apply(ggemm_input, cum_token_counts, self.use_triton_path_for_gemm, *mlp_params)
                else:
                    with record_pcl_function("1moeblock--7ggemm_expert_blocks"):
                        gemm_output_list = []
                        for n in range(self.num_experts_per_rank):
                            start = cum_token_counts[n].item()
                            end = cum_token_counts[n+1].item()
                            if start != end:
                                current_state = ggemm_input[start:end]
                                gemm_output = self.experts[n](current_state)
                                gemm_output_list.append(gemm_output)
                        ggemm_output = torch.cat(gemm_output_list, dim=0)
            else:
                ggemm_output = self.experts(ggemm_input, cum_token_counts)

            # K6 (Gather + Reduce)
            final_hidden_states = GroupGemmOutputReductionFunction.apply(
                                    ggemm_output, 
                                    routing_weights, 
                                    cum_expert_counts_ep, 
                                    selected_experts_indices, 
                                    input_indices,
                                    output_indices, 
                                    self.use_ref_path_for_k6_fwd, 
                                    self.use_ref_path_for_k6_bwd,
                                    self.use_triton_path_for_k6_fwd,
                                    self.use_triton_path_for_k6_bwd)
            
            if self.pmap.expert_parallelism > 1:
                forward_synchronize_backward_identity.apply(self.pmap.ep_group)
                final_hidden_states = forward_reducescatter_backward_allgather.apply(final_hidden_states, self.pmap.ep_ind, self.pmap.expert_parallelism, self.pmap.ep_group, self.use_allreduce_for_reducescatter, False)
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

            return final_hidden_states
        
        args = [hidden_states, selected_experts, routing_weights]
        if self.use_activation_checkpointing_for_moe:
            final_hidden_states = torch_activation_checkpoint(expert_computation_block, *args, use_reentrant=False)
        else:
            final_hidden_states = expert_computation_block(*args)

        if self.output_router_logits:
            return final_hidden_states, router_logits
        else:
            return final_hidden_states, None

    def set_parameters_from_full_module(self, module):
        self.gate.load_state_dict(module.gate.state_dict())
        if not self.config.use_merged_mlp_in_fast_moe:
            for local_expert_idx in range(self.num_experts_per_rank):
                self.experts[local_expert_idx].load_state_dict(module.experts[self.expert_start_idx + local_expert_idx].state_dict())
        else:
            for local_expert_idx in range(self.num_experts_per_rank):
                self.experts.gate_proj_merged_weight[local_expert_idx].copy_(module.experts[self.expert_start_idx + local_expert_idx].gate_proj.weight.data)
                self.experts.up_proj_merged_weight[local_expert_idx].copy_(module.experts[self.expert_start_idx + local_expert_idx].up_proj.weight.data)
                self.experts.down_proj_merged_weight[local_expert_idx].copy_(module.experts[self.expert_start_idx + local_expert_idx].down_proj.weight.data)