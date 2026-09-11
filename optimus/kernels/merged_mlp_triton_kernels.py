import time
import torch

import triton
import triton.language as tl

####################################################################################################################
@triton.jit
def triton_grouped_linear_forward_kernel(
    ga_ptr, 
    b_ptr,
    gc_ptr,
    offsets_ptr,
    N: tl.constexpr, K: tl.constexpr, 
    G: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    ):
    
    group_idx = tl.program_id(axis=1)
    pid_mn = tl.program_id(axis=0)

    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    # Set GEMM pointers and descriptors
    gm_start = tl.load(offsets_ptr + group_idx)
    gm_end = tl.load(offsets_ptr + (group_idx + 1))
    M = gm_end - gm_start

    a_ptr = ga_ptr + (gm_start * K)
    b_base = b_ptr + group_idx * (N * K)
    c_ptr = gc_ptr + (gm_start * N)

    # local token start inside of the current expert
    m_start = pid_m * BLOCK_SIZE_M
    if m_start >= M:
        return

    a_desc = tl.make_tensor_descriptor(
        base = a_ptr,
        shape = (M, K),
        strides = (K, 1),
        block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_K)
    )

    # Memory layout of B is transposed
    b_desc = tl.make_tensor_descriptor(
        base = b_base,
        shape = (N, K),
        strides = (K, 1),
        block_shape = (BLOCK_SIZE_N, BLOCK_SIZE_K)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    off_k = 0
    for _ in range(0, K, BLOCK_SIZE_K):
        a_block = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        b_block = b_desc.load([pid_n * BLOCK_SIZE_N, off_k])
        accumulator = tl.dot(a_block, tl.trans(b_block), accumulator)
        off_k += BLOCK_SIZE_K
    
    c_desc = tl.make_tensor_descriptor(
        base = c_ptr,
        shape = (M, N),
        strides = (N, 1),
        block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_N)
    )
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], accumulator.to(tl.bfloat16))

def triton_grouped_linear_forward_wrapper(input, merged_weight, output, cum_token_counts, max_tokens_per_expert):
    G, N, K = merged_weight.shape
    
    # Calling triton kernel.
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 4
    
    num_pid_m = (max_tokens_per_expert + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (num_pid_m * num_pid_n, G)

    cum_token_counts_device = cum_token_counts.to(input.device)
    triton_grouped_linear_forward_kernel[grid](
        input, 
        merged_weight,
        output, 
        cum_token_counts_device,
        N, K,
        G,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=32, num_stages=4
    )

####################################################################################################################

####################################################################################################################
@triton.jit
def triton_grouped_linear_backward_input_kernel(
    ga_ptr, 
    b_ptr,
    gc_ptr,
    offsets_ptr,
    N: tl.constexpr, K: tl.constexpr, 
    G: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    ):
    
    group_idx = tl.program_id(axis=1)
    pid_mn = tl.program_id(axis=0)

    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    # Set GEMM pointers and descriptors
    gm_start = tl.load(offsets_ptr + group_idx)
    gm_end = tl.load(offsets_ptr + (group_idx + 1))
    M = gm_end - gm_start

    a_ptr = ga_ptr + (gm_start * K)
    b_base = b_ptr + group_idx * (K * N)
    c_ptr = gc_ptr + (gm_start * N)

    # local token start inside of the current expert
    m_start = pid_m * BLOCK_SIZE_M
    if m_start >= M:
        return

    a_desc = tl.make_tensor_descriptor(
        base = a_ptr,
        shape = (M, K),
        strides = (K, 1),
        block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_K)
    )

    b_desc = tl.make_tensor_descriptor(
        base = b_base,
        shape = (K, N),
        strides = (N, 1),
        block_shape = (BLOCK_SIZE_K, BLOCK_SIZE_N)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    off_k = 0
    for _ in range(0, K, BLOCK_SIZE_K):
        a_block = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        b_block = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        accumulator = tl.dot(a_block, b_block, accumulator)
        off_k += BLOCK_SIZE_K
    
    c_desc = tl.make_tensor_descriptor(
        base = c_ptr,
        shape = (M, N),
        strides = (N, 1),
        block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_N)
    )
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], accumulator.to(tl.bfloat16))

def triton_grouped_linear_backward_input_wrapper(output_grad, merged_weight, input_grad, cum_token_counts, max_tokens_per_expert):
    G, K, N = merged_weight.shape
    
    # Calling triton kernel.
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 4
    
    num_pid_m = (max_tokens_per_expert + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (num_pid_m * num_pid_n, G)

    cum_token_counts_device = cum_token_counts.to(output_grad.device)
    triton_grouped_linear_backward_input_kernel[grid](
        output_grad,
        merged_weight,
        input_grad,
        cum_token_counts_device,
        N, K,
        G,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=32, num_stages=4
    )
####################################################################################################################

####################################################################################################################
@triton.jit
def triton_grouped_linear_backward_weight_kernel(
    ga_ptr, 
    gb_ptr,
    c_ptr,
    offsets_ptr,
    M: tl.constexpr, N: tl.constexpr, 
    G: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    ):
    
    group_idx = tl.program_id(axis=1)
    pid_mn = tl.program_id(axis=0)

    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    # Set GEMM pointers and descriptors
    gm_start = tl.load(offsets_ptr + group_idx)
    gm_end = tl.load(offsets_ptr + (group_idx + 1))
    K = gm_end - gm_start

    a_ptr = ga_ptr + (gm_start * M)
    b_ptr = gb_ptr + (gm_start * N)
    c_base = c_ptr + group_idx * (M * N)

    # Memory layout of A is transposed
    a_desc = tl.make_tensor_descriptor(
        base = a_ptr,
        shape = (K, M),
        strides = (M, 1),
        block_shape = (BLOCK_SIZE_K, BLOCK_SIZE_M)
    )

    b_desc = tl.make_tensor_descriptor(
        base = b_ptr,
        shape = (K, N),
        strides = (N, 1),
        block_shape = (BLOCK_SIZE_K, BLOCK_SIZE_N)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    off_k = 0
    for _ in range(0, K, BLOCK_SIZE_K):
        a_block = a_desc.load([off_k, pid_m * BLOCK_SIZE_M])
        b_block = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        accumulator = tl.dot(tl.trans(a_block), b_block, accumulator)
        off_k += BLOCK_SIZE_K
    
    c_desc = tl.make_tensor_descriptor(
        base = c_base,
        shape = (M, N),
        strides = (N, 1),
        block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_N)
    )
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], accumulator.to(tl.bfloat16))

def triton_grouped_linear_backward_weight_wrapper(output_grad, input, merged_weight_grad, cum_token_counts, max_tokens_per_expert):
    G, M, N = merged_weight_grad.shape
    
    # Calling triton kernel.
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 4
    
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (num_pid_m * num_pid_n, G)

    cum_token_counts_device = cum_token_counts.to(output_grad.device)
    triton_grouped_linear_backward_weight_kernel[grid](
        output_grad,
        input,
        merged_weight_grad,
        cum_token_counts_device,
        M, N,
        G,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=32, num_stages=4
    )
####################################################################################################################