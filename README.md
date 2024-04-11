## Prerequisites

Before installing the causal depthwise conv1d module, you need to patch your ROCm installation due to a known issue with some functions not being declared as `inline`. This step is required to avoid multiple definition errors during compilation.

### Patching ROCm

1. Locate your ROCm installation directory. This is typically found at `/opt/rocm/`, but the location may vary depending on your installation.

2. Backup the original ROCm header file:
   ```bash
    sudo cp /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h.backup
   ```

3. Copy the patched header file from this repository to your ROCm include directory:
   ```bash
    sudo cp rocm_update_files/amd_hip_bf16.h /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h
   ```

# Causal depthwise conv1d in CUDA with a PyTorch interface

Features:
- Support fp32, fp16, bf16.
- Kernel size 2, 3, 4.

## How to use

```
from causal_conv1d import causal_conv1d_fn
```

```
def causal_conv1d_fn(x, weight, bias=None, activation=None):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
```

Equivalent to:
```
import torch.nn.functional as F

F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)[..., :seqlen]
```
