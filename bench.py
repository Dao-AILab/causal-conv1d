import torch
import torch.nn.functional as F
import time
from causal_conv1d import causal_conv1d_fn

# Set the device to GPU (assuming CUDA is available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# Define the benchmark function for the custom implementation
def benchmark_custom(x, weight, bias, activation):
    # Warmup run
    for _ in range(10):
        out = causal_conv1d_fn(x, weight, bias, activation)
    torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(100):  # Number of iterations for the benchmark
        out = causal_conv1d_fn(x, weight, bias, activation)
    torch.cuda.synchronize()
    end_time = time.time()
    return (end_time - start_time) / 100  # Average time per iteration

# Define the benchmark function for the PyTorch implementation
def benchmark_pytorch(x, weight, bias, width):
    # Warmup run
    dim = x.shape[1]
    for _ in range(10):
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)[..., :x.shape[-1]]
    torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(100):  # Number of iterations for the benchmark
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)[..., :x.shape[-1]]
    torch.cuda.synchronize()
    end_time = time.time()
    return (end_time - start_time) / 100  # Average time per iteration

# Parameters for the benchmark
batch_size = 128
dim = 512
seqlen = 16384
widths = [2, 3, 4]
dtypes = [torch.float32, torch.float16, torch.bfloat16]

for dtype in dtypes:
    print(f"Benchmarking with dtype: {dtype}")
    for width in widths:
        print(f"Kernel size: {width}")
        
        # Initialize inputs
        x = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
        weight = torch.randn(dim, width, device=device, dtype=dtype)
        bias = torch.randn(dim, device=device, dtype=dtype)
        
        # Benchmark custom implementation with activation=None
        custom_time = benchmark_custom(x, weight, bias, None)
        print(f"Custom implementation time (activation=None): {custom_time * 1e3:.3f} ms")
        
        # Benchmark PyTorch implementation
        pytorch_time = benchmark_pytorch(x, weight, bias, width)
        print(f"PyTorch implementation time: {pytorch_time * 1e3:.3f} ms")
        
        # Compare performance
        speedup = pytorch_time / custom_time
        print(f"Speedup: {speedup:.2f}x")
        print("-" * 50)
