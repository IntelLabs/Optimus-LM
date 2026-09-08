import torch
import intel_extension_for_pytorch
import time

import pcl_profiler

dtype = torch.bfloat16
device = "xpu"
warmup_iters = 10
bench_iters = 10

N = 4096
a = torch.randn((N, N), dtype=dtype, device=device)
b = torch.randn((N, N), dtype=dtype, device=device)
c = torch.randn((N, N), dtype=dtype, device=device)

# Warmup
start_event = pcl_profiler.mark_event()
for i in range(warmup_iters):
    torch.matmul(a, b, out=c)
end_event = pcl_profiler.mark_event()
torch.xpu.synchronize()

# Benchmark
start_time = time.time()
start_event = pcl_profiler.mark_event()
for i in range(bench_iters):
    torch.matmul(a, b, out=c)
end_event = pcl_profiler.mark_event()
torch.xpu.synchronize()
end_time = time.time()

print(f"CPU time (Using time.time) : {(end_time-start_time) * 1e3:.3f}")

# Get time
elapsed_cpu_time_ms = (end_event.get_submit_time() - start_event.get_submit_time()) * 1e-6
elapsed_gpu_time_ms = (end_event.get_start_time() - start_event.get_end_time()) * 1e-6
overhead_time_ms = (start_event.get_elapsed_time() + end_event.get_elapsed_time()) * 1e-6
print(f"CPU time  : {elapsed_cpu_time_ms:.3f}")
print(f"GPU time  : {elapsed_gpu_time_ms:.3f}")
print(f"MARK time : {overhead_time_ms:.3f}")

avg_gemm_time_ms = elapsed_gpu_time_ms / bench_iters
tops = (((2 * N * N * N) / avg_gemm_time_ms) * 1e-9) 

print("TOPS: {:.3f}".format(tops))
