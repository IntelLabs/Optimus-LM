import time
import torch

import triton
import triton.language as tl
   

@triton.jit
def triton_linear_backward_weight_kernel(
    a_ptr, 
    b_ptr,
    c_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    ):

    pid_mn = tl.program_id(axis=0)
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

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

def triton_linear_backward_weight(output_grad, input, weight_grad):
    K = output_grad.shape[0]
    M, N = weight_grad.shape
    
    # Calling triton kernel.
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 4

    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (num_pid_m * num_pid_n,)

    triton_linear_backward_weight_kernel[grid](
        output_grad, 
        input,
        weight_grad,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=32, num_stages=4
    )

def reference_linear_backward_weight(output_grad, input, weight_grad):
    torch.matmul(output_grad.t(), input, out=weight_grad)

if __name__ == "__main__":
    torch.manual_seed(0)

    # Model configuration
    seq_length = 4096
    ifm = 3072
    ofm = 2048

    # Run configuration
    dtype = torch.bfloat16
    device = "xpu"
    warmup_iters = 10
    bench_iters = 10
    
    output_grad = torch.randn((seq_length, ofm), dtype=dtype, device=device)
    input = torch.randn((seq_length, ifm), dtype=dtype, device=device)
    weight_grad = torch.zeros((ofm, ifm), dtype=dtype, device=device)

    # Warmup steps
    for i in range(warmup_iters):
        triton_linear_backward_weight(output_grad, input, weight_grad)
    torch.xpu.synchronize()

    # Benchmark steps
    st = time.time()
    for i in range(bench_iters):
        triton_linear_backward_weight(output_grad, input, weight_grad)
    torch.xpu.synchronize()
    et = time.time()
    avg_time = (et - st) / bench_iters
    tflops = 2 * seq_length * ofm * ifm / avg_time / 1e12
    print(f"Triton path : {avg_time*1e3:5.2f} ms, {tflops:6.2f} TFLOPS")

    # Reference path
    weight_grad_ref = torch.zeros((ofm, ifm), dtype=dtype, device=device)
    for i in range(warmup_iters):
        reference_linear_backward_weight(output_grad, input, weight_grad_ref)
    torch.xpu.synchronize()

    st = time.time()
    for i in range(bench_iters):
        reference_linear_backward_weight(output_grad, input, weight_grad_ref)
    torch.xpu.synchronize()
    et = time.time()
    avg_time = (et - st) / bench_iters
    tflops = 2 * seq_length * ofm * ifm / avg_time / 1e12
    print(f"OneDNN path : {avg_time*1e3:5.2f} ms, {tflops:6.2f} TFLOPS")

    # Verification
    torch.allclose(weight_grad, weight_grad_ref, atol=1e-2)
    print(weight_grad.flatten()[:10])
    print(weight_grad_ref.flatten()[:10])