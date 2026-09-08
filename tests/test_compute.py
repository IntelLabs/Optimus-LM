import torch
# import intel_extension_for_pytorch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    else:
        exit("No supported device found. Please ensure you have a compatible GPU or XPU available.")

dtype = torch.bfloat16
device = get_device()

N = 4
a = torch.randn((N,N), dtype=dtype, device=device)
b = torch.randn((N,N), dtype=dtype, device=device)
c =  torch.matmul(a, b)

print(c)
print(f"If it prints a BF16 4x4 tensor, it is a SUCCESS")