import time
from typing import Callable, List, Optional
import torch

import triton
import triton.language as tl



from triton.testing import assert_close as triton_assert_close, Benchmark, do_bench as triton_do_bench


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.xpu.is_available():
        torch.xpu.synchronize()

def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if times.numel() > 2:
            # exclude max and min times
            times = torch.sort(times).values[1:-1]
        # add coefficient of the variance.
        std = torch.std(times)
        mean = torch.mean(times)
        cv = std / mean
        ret.extend([mean.tolist(), cv.tolist()])
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()


def do_bench_elapsed_time(fn, n_warmup=25, n_repeat=100, grad_to_none=None, quantiles=None, return_mode="mean",
                          device="xpu", time_warmup=False, benchmark_label=None,  # pylint: disable=W0613
                          ):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param n_warmup: Number of repetitions for warmup
    :type n_warmup: int
    :param n_repeat: Number of repetitions to collect measurements
    :type n_repeat: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    """
    assert return_mode in ["min", "max", "mean", "median"]

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    cache_size = 256 * 1024 * 1024
    cache = torch.empty(int(cache_size // 4), dtype=torch.int, device=device)

    # Estimate the runtime of the function
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # The cache is also maintained in `triton_do_bench` function,
    # there is no need to duplicate the amount of memory used.
    del cache

    # compute warmup and repeat times
    if time_warmup:
        warmup_ms = n_warmup
    else:
        warmup_ms = n_warmup * estimate_ms
    rep_time = n_repeat * estimate_ms

    times = triton_do_bench(fn, warmup=warmup_ms, rep=rep_time, grad_to_none=grad_to_none, return_mode="all")
    times = torch.tensor(times, dtype=torch.float)
    return _summarize_statistics(times, quantiles, return_mode)


def get_matmul_autotune_configs() -> List[triton.Config]:
    configs = [
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256'},
            num_stages=s, num_warps=32) for s in [1, 2, 3, 4]
    ] + [
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': m},
                      num_stages=s, num_warps=w) for s in [2, 3, 4] for (m, w) in ([('256', 32), ('128', 64)])
    ] + [
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256'},
            num_stages=s, num_warps=32) for s in [2]
    ] + [
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': m},
                      num_stages=s, num_warps=w) for s in [2, 3] for (m, w) in ([('256', 32), ('128', 64)])
    ]
    return configs


@triton.autotune(
    configs=get_matmul_autotune_configs(),
    key=['M', 'N', 'K'],
)
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
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c_block)

def triton_gemm(a_mat, b_mat, c_mat):
    M, K = a_mat.shape
    N = b_mat.shape[1]
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    triton_gemm_kernel[grid](
        a_mat, 
        b_mat,
        c_mat, 
        M, N, K
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
    b_mat = torch.randn((K, N), dtype=dtype, device=device)
    c_mat = torch.empty((M, N), dtype=dtype, device=device)
    
    # from optimus.profilers import PclProfiler
    # profiler = PclProfiler()
    # profiler.register_timer("kernel")

    """
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
    """

    perf_stats = do_bench_elapsed_time(
        lambda: triton_gemm(a_mat, b_mat, c_mat),
        n_warmup=warmup_iters,
        n_repeat=bench_iters,
        device=device
    )

    # print(perf_stats)
    avg_time = perf_stats
    tflops = 2 * M * N * K / avg_time / 1e12
    print(f"Triton path : {avg_time*1e3:5.2f} ms, {tflops:6.2f} TFLOPS")


    # Reference path
    c_mat_ref = torch.empty((M, N), dtype=dtype, device=device)
    for i in range(warmup_iters):
        torch.matmul(a_mat, b_mat, out=c_mat_ref)
    torch.xpu.synchronize()

    st = time.time()
    for i in range(bench_iters):
        torch.matmul(a_mat, b_mat, out=c_mat_ref)
    torch.xpu.synchronize()
    et = time.time()
    avg_time = (et - st) / bench_iters
    tflops = 2 * M * N * K / avg_time / 1e12
    print(f"OneDNN path : {avg_time*1e3:5.2f} ms, {tflops:6.2f} TFLOPS")

    # Verification
    torch.allclose(c_mat, c_mat_ref, atol=1e-2)
    print(c_mat.flatten()[:10])
    print(c_mat_ref.flatten()[:10])