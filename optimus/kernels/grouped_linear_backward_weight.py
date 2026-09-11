import time
import torch

import triton
import triton.language as tl
   

@triton.jit
def triton_grouped_linear_backward_weight_kernel(
    ga_ptr, 
    gb_ptr,
    c_ptrs,
    offsets_ptr,
    M, N, G,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    ):

    pid_mn = tl.program_id(axis=0)
    group_idx = tl.program_id(axis=1)

    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    # Set GEMM pointers and descriptors
    a_ptr = ga_ptr + (tl.load(offsets_ptr + group_idx) * M)
    b_ptr = gb_ptr + (tl.load(offsets_ptr + group_idx) * N)
    c_ptr = tl.load(c_ptrs + group_idx).to(tl.pointer_type(tl.bfloat16))
    K = tl.load(offsets_ptr + (group_idx + 1)) - tl.load(offsets_ptr + group_idx)

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
        base = c_ptr,
        shape = (M, N),
        strides = (N, 1),
        block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_N)
    )

    c_block = accumulator.to(tl.bfloat16)
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c_block)

def triton_grouped_linear_backward_weight(output_grad, input, weight_grads, cum_token_counts):
    num_experts = len(weight_grads)
    M, N = weight_grads[0].shape
    
    # Preparing weight pointers.
    weight_grad_addrs = [weight_grad.data_ptr() for weight_grad in weight_grads]
    weight_grad_ptrs = torch.tensor(weight_grad_addrs, device=output_grad.device, dtype=torch.uint64)
    offsets = cum_token_counts.to(output_grad.device)

    # Calling triton kernel.
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 4

    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (num_pid_m * num_pid_n, num_experts)

    triton_grouped_linear_backward_weight_kernel[grid](
        output_grad, 
        input,
        weight_grad_ptrs, 
        offsets,
        M, N, num_experts,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=32, num_stages=4
    )

def reference_grouped_linear_backward_weight(output_grad, input, weight_grads, cum_token_counts):
    for g in range(len(weight_grads)):
        start = cum_token_counts[g].item()
        end = cum_token_counts[g + 1].item()
        a_mat = output_grad[start:end, :]
        b_mat = input[start:end, :]
        c_mat = weight_grads[g]
        torch.matmul(a_mat.t(), b_mat, out=c_mat)

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
    input = torch.randn((num_tokens, ifm), dtype=dtype, device=device)
    weight_grads = [torch.zeros((ofm, ifm), dtype=dtype, device=device)  for _ in range(num_experts)]

    sizes = torch.ones(num_experts, dtype=torch.int32) * (num_tokens // num_experts)
    cum_token_counts = torch.zeros(num_experts + 1, dtype=torch.int32)
    torch.cumsum(sizes, dim=0, out=cum_token_counts[1:])
    
    # Warmup steps
    for i in range(warmup_iters):
        triton_grouped_linear_backward_weight(output_grad, input, weight_grads, cum_token_counts)
    torch.xpu.synchronize()

    # Benchmark steps
    st = time.time()
    for i in range(bench_iters):
        triton_grouped_linear_backward_weight(output_grad, input, weight_grads, cum_token_counts)
    torch.xpu.synchronize()
    et = time.time()
    avg_time = (et - st) / bench_iters
    tflops = 2 * num_tokens * ofm * ifm / avg_time / 1e12
    print(f"Triton path : {avg_time*1e3:5.2f} ms, {tflops:6.2f} TFLOPS")

    # Reference path
    weight_grads_ref = [torch.zeros((ofm, ifm), dtype=dtype, device=device)  for _ in range(num_experts)]
    for i in range(warmup_iters):
        reference_grouped_linear_backward_weight(output_grad, input, weight_grads_ref, cum_token_counts)
    torch.xpu.synchronize()

    st = time.time()
    for i in range(bench_iters):
        reference_grouped_linear_backward_weight(output_grad, input, weight_grads_ref, cum_token_counts)
    torch.xpu.synchronize()
    et = time.time()
    avg_time = (et - st) / bench_iters
    tflops = 2 * num_tokens * ofm * ifm / avg_time / 1e12
    print(f"OneDNN path : {avg_time*1e3:5.2f} ms, {tflops:6.2f} TFLOPS")

    # Verification
    torch.allclose(torch.cat(weight_grads), torch.cat(weight_grads_ref), atol=1e-2)
    print(weight_grads[0].flatten()[:10])
    print(weight_grads_ref[0].flatten()[:10])