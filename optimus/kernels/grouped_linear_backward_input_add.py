import time
import torch

import triton
import triton.language as tl

@triton.jit
def triton_grouped_linear_backward_input_add_kernel(
    ga_ptr, 
    b_ptrs,
    gc_ptr,
    offsets_ptr,
    N, K, G, output_acc,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    ):

    pid_mn = tl.program_id(axis=0)
    group_idx = tl.program_id(axis=1)

    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    # Set GEMM pointers and descriptors
    gm_start = tl.load(offsets_ptr + group_idx)
    gm_end = tl.load(offsets_ptr + (group_idx + 1))
    M = gm_end - gm_start

    a_ptr = ga_ptr + (gm_start * K)
    b_ptr = tl.load(b_ptrs + group_idx).to(tl.pointer_type(tl.bfloat16))
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
        base = b_ptr,
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

    c_block = accumulator.to(tl.bfloat16)
    if output_acc:
        existing_c_block = c_desc.load([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N])
        c_block = tl.add(c_block, existing_c_block)
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c_block)

def triton_grouped_linear_backward_input_add(output_grad, weights, input_grad, cum_token_counts):
    K = output_grad.shape[1]
    N = weights[0].shape[1]
    num_experts = len(weights)
    max_tokens_per_expert = max([cum_token_counts[g + 1] - cum_token_counts[g] for g in range(num_experts)])
    
    # Preparing weight pointers.
    weight_addrs = [weight.data_ptr() for weight in weights]
    weight_ptrs = torch.tensor(weight_addrs, device=output_grad.device, dtype=torch.uint64)
    offsets = cum_token_counts.to(output_grad.device)

    # Calling triton kernel.
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 4

    num_pid_m = (max_tokens_per_expert + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    grid = (num_pid_m * num_pid_n, num_experts)
    triton_grouped_linear_backward_input_add_kernel[grid](
        output_grad, 
        weight_ptrs,
        input_grad, 
        offsets,
        N, K, num_experts, True,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=32, num_stages=4
    )

    return input_grad

def reference_grouped_linear_backward_input_add(output_grad, weights, input_grad, cum_token_counts):
    for g in range(len(weights)):
        start = cum_token_counts[g].item()
        end = cum_token_counts[g + 1].item()
        a_mat = output_grad[start:end, :]
        b_mat = weights[g]
        c_mat = input_grad[start:end, :]
        torch.addmm(c_mat, a_mat, b_mat, out=c_mat)
    return input_grad

if __name__ == "__main__":
    torch.manual_seed(0)

    # Model configuration
    seq_length = 4096
    ifm = 3072
    ofm = 2048
    num_experts = 16
    num_experts_per_tok = 4
    num_tokens = (seq_length * num_experts_per_tok)

    # Run configuration
    dtype = torch.bfloat16
    device = "xpu"
    warmup_iters = 10
    bench_iters = 10
    
    output_grad = torch.randn((num_tokens, ofm), dtype=dtype, device=device)
    weights = [torch.randn((ofm, ifm), dtype=dtype, device=device)  for _ in range(num_experts)]
    input_grad = torch.randn((num_tokens, ifm), dtype=dtype, device=device)
    input_grad_orig = input_grad.clone()

    sizes = torch.ones(num_experts, dtype=torch.int32) * (num_tokens // num_experts)
    cum_token_counts = torch.zeros(num_experts + 1, dtype=torch.int32)
    torch.cumsum(sizes, dim=0, out=cum_token_counts[1:])
    
    # Warmup steps
    for i in range(warmup_iters):
        output = triton_grouped_linear_backward_input_add(output_grad, weights, input_grad, cum_token_counts)
    torch.xpu.synchronize()

    # Benchmark steps
    st = time.time()
    for i in range(bench_iters):
        if i == bench_iters - 1:
            input_grad.copy_(input_grad_orig)
        output = triton_grouped_linear_backward_input_add(output_grad, weights, input_grad, cum_token_counts)
    torch.xpu.synchronize()
    et = time.time()
    avg_time = (et - st) / bench_iters
    tflops = 2 * num_tokens * ofm * ifm / avg_time / 1e12
    print(f"Triton path : {avg_time*1e3:5.2f} ms, {tflops:6.2f} TFLOPS")

    # Reference path
    input_grad_ref = input_grad.clone()
    for i in range(warmup_iters):
        input_grad_ref = reference_grouped_linear_backward_input_add(output_grad, weights, input_grad_ref, cum_token_counts)
    torch.xpu.synchronize()

    st = time.time()
    for i in range(bench_iters):
        if i == bench_iters - 1:
            input_grad_ref.copy_(input_grad_orig)
        input_grad_ref = reference_grouped_linear_backward_input_add(output_grad, weights, input_grad_ref, cum_token_counts)
    torch.xpu.synchronize()
    et = time.time()
    avg_time = (et - st) / bench_iters
    tflops = 2 * num_tokens * ofm * ifm / avg_time / 1e12
    print(f"OneDNN path : {avg_time*1e3:5.2f} ms, {tflops:6.2f} TFLOPS")

    # Verification
    torch.allclose(input_grad, input_grad_ref, atol=1e-2)
    print(input_grad.flatten()[:10])
    print(input_grad_ref.flatten()[:10])