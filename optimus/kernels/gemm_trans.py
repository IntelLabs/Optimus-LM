import time
import torch

import triton
import triton.language as tl

@triton.jit
def triton_gemm_kernel(
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

    a_desc = tl.make_tensor_descriptor(
        base = a_ptr,
        shape = (M, K),
        strides = (K, 1),
        block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_K)
    )

    b_desc = tl.make_tensor_descriptor(
        base = b_ptr,
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

    c_block = accumulator.to(tl.bfloat16)
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c_block)

def triton_gemm(a_mat, b_mat, c_mat):
    M, K = a_mat.shape
    N = b_mat.shape[0]

    # Calling triton kernel.
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 4
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M)*triton.cdiv(N, BLOCK_SIZE_N),)
    triton_gemm_kernel[grid](
        a_mat, 
        b_mat,
        c_mat, 
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=32, num_stages=4
    )

if __name__ == "__main__":
    torch.manual_seed(0)

    # Model configuration
    M = 4096
    K = 4096
    N = 4096

    # Run configuration
    dtype = torch.bfloat16
    device = "xpu"
    warmup_iters = 10
    bench_iters = 10
    
    a_mat = torch.randn((M, K), dtype=dtype, device=device)
    b_mat = torch.randn((N, K), dtype=dtype, device=device)
    c_mat = torch.empty((M, N), dtype=dtype, device=device)
    
    # Warmup steps
    for i in range(warmup_iters):
        triton_gemm(a_mat, b_mat, c_mat)
    torch.xpu.synchronize()

    # Benchmark steps
    st = time.time()
    for i in range(bench_iters):
        triton_gemm(a_mat, b_mat, c_mat)
    torch.xpu.synchronize()
    et = time.time()
    avg_time = (et - st) / bench_iters
    tflops = 2 * M * N * K / avg_time / 1e12
    print(f"Triton path : {avg_time*1e3:5.2f} ms, {tflops:6.2f} TFLOPS")

    # Reference path
    c_mat_ref = torch.empty((M, N), dtype=dtype, device=device)
    for i in range(warmup_iters):
        torch.matmul(a_mat, b_mat.t(), out=c_mat_ref)
    torch.xpu.synchronize()

    st = time.time()
    for i in range(bench_iters):
        torch.matmul(a_mat, b_mat.t(), out=c_mat_ref)
    torch.xpu.synchronize()
    et = time.time()
    avg_time = (et - st) / bench_iters
    tflops = 2 * M * N * K / avg_time / 1e12
    print(f"OneDNN path : {avg_time*1e3:5.2f} ms, {tflops:6.2f} TFLOPS")

    # Verification
    torch.allclose(c_mat, c_mat_ref, atol=1e-2)
    print(c_mat.flatten()[:10])
    print(c_mat_ref.flatten()[:10])
