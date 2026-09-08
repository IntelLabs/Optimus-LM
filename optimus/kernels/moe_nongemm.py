import torch

import triton
import triton.language as tl

@triton.jit
def compute_partial_token_counts_triton_kernel(
    selected_experts, 
    partial_token_counts,
    expert_counts_ep,
    expert_start_idx, expert_end_idx,
    K,
    TB: tl.constexpr
    ):
    # selected_experts     : [T, K]
    # partial_token_counts : [NR, TH]
    # expert_counts_ep     : [T]

    # T  -> Number of tokens (EP * B * S))
    # K  -> Experts per token
    # NR -> Number of experts per rank (N / EP)
    # TH -> Number of threads

    tid = tl.program_id(axis=0)
    TH = tl.num_programs(axis=0)

    for lt in range(TB):
        expert_count = 0
        t = tid * TB + lt
        for k in range(K):
            n = tl.load(selected_experts + (t * K + k))
            if n >= expert_start_idx and n < expert_end_idx:
                offset = ((n - expert_start_idx) * TH + tid)
                cur_count = tl.load(partial_token_counts + offset)
                cur_count += 1
                expert_count += 1

                tl.store(partial_token_counts + offset, cur_count)
        tl.store(expert_counts_ep + t, expert_count)


@triton.jit
def compute_input_and_output_indices_triton_kernel(
    selected_experts,
    partial_cum_token_counts,
    input_increment_buffer,
    cum_expert_counts_ep,
    selected_experts_indices,
    input_indices,
    output_indices,
    expert_start_idx,
    expert_end_idx,
    K,
    TB: tl.constexpr
    ):

    # selected_experts         : [T, K]
    # partial_cum_token_counts : [NR*TH + 1]
    # input_increment_buffer   : [NR, TH]
    # cum_expert_counts_ep     : [T]
    # selected_experts_indices : TR
    # input_indices            : TR
    # output_indices           : TR

    tid = tl.program_id(axis=0)
    TH = tl.num_programs(axis=0)

    for t in range(tid*TB, (tid+1)*TB):
        output_index = tl.load(cum_expert_counts_ep + t)
        for k in range(K):
            n = tl.load(selected_experts + (t * K + k))
            if n >= expert_start_idx and n < expert_end_idx:
                ln = n - expert_start_idx
                base = tl.load(partial_cum_token_counts + (ln * TH + tid))
                offset = tl.load(input_increment_buffer + (ln * TH + tid))
                input_index = base + offset

                tl.store(input_indices + input_index, t)
                tl.store(input_increment_buffer + (ln * TH + tid), offset + 1)

                tl.store(output_indices + output_index, input_index)
                tl.store(selected_experts_indices + output_index, k)
                output_index += 1


@triton.jit
def index_gather_triton_kernel(
    hidden_states, 
    input_indices, 
    ggemm_input,
    E):

    o = tl.program_id(0)

    index = tl.load(input_indices + o)
    for e in range(E):
        val = tl.load(hidden_states + (index * E + e))
        tl.store(ggemm_input + (o * E + e), val)


@triton.jit
def shuffle_sub_matrix_reduce_triton_kernel(
    ggemm_input_grad, 
    cum_expert_counts_ep, 
    output_indices, 
    hidden_states_grad,
    E):

    t = tl.program_id(0)

    start_idx = tl.load(cum_expert_counts_ep + t)
    end_idx = tl.load(cum_expert_counts_ep + (t + 1))

    for e in range(E):
        acc = 0.0
        for i in range(start_idx, end_idx):
            index = tl.load(output_indices + i)
            val = tl.load(ggemm_input_grad + (index * E + e))
            acc += val
        tl.store(hidden_states_grad + (t * E + e), acc)


@triton.jit
def reduce_ggemm_output_triton_kernel( 
    ggemm_output, 
    routing_weights, 
    cum_expert_counts_ep, 
    selected_experts_indices, 
    output_indices, 
    final_hidden_states,
    K, E):

    t = tl.program_id(0)

    start_idx = tl.load(cum_expert_counts_ep + t)
    end_idx = tl.load(cum_expert_counts_ep + (t + 1))

    for e in range(E):
        acc = 0.0    
        for i in range(start_idx, end_idx):
            k = tl.load(selected_experts_indices + i)
            index = tl.load(output_indices + i)
            scaling = tl.load(routing_weights + (t * K + k))
            output = tl.load(ggemm_output + (index * E + e))
            acc += output * scaling
        tl.store(final_hidden_states + (t * E + e), acc)


@triton.jit
def reduce_ggemm_output_backward_triton_kernel( 
    final_hidden_states_grad,
    ggemm_output,
    routing_weights,
    cum_expert_counts_ep,
    selected_experts_indices,
    input_indices,
    output_indices,
    ggemm_output_grad,
    routing_weights_grad,
    K, E):

    rt = tl.program_id(0)
    index = tl.load(output_indices + rt)
    t = tl.load(input_indices + index)
    k = tl.load(selected_experts_indices + rt)

    weight_grad_acc = 0.0
    for e in range(E):
        scaling = tl.load(routing_weights + (t * K + k))
        output_grad = tl.load(final_hidden_states_grad + (t * E + e))
        input = tl.load(ggemm_output + (index * E + e))

        tl.store(ggemm_output_grad + (index * E + e), scaling * output_grad)
        weight_grad_acc += input * output_grad
    
    tl.store(routing_weights_grad + (t * K + k), weight_grad_acc)