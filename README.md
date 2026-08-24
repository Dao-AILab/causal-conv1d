# Causal depthwise conv1d in CUDA with a PyTorch interface

Features:
- Support fp32, fp16, bf16.
- Kernel size 2, 3, 4.

## How to use

```python
from causal_conv1d import causal_conv1d_fn
```

```python
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
```python
import torch.nn.functional as F

F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)[..., :seqlen]
```

## AMD ROCm/HIP support

causal-conv1d supports AMD GPUs via ROCm/HIP (ROCm 6.1+).

### Building from source

```bash
# Set your GPU architecture
export HIP_ARCHITECTURES=gfx1201  # e.g., gfx1100, gfx1101, gfx1201

# --no-build-isolation ensures the build uses your existing ROCm PyTorch
# instead of pulling the default CUDA PyTorch from PyPI
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-build-isolation -e .
```

`--no-build-isolation` is required because pip's default build isolation installs a CUDA-only PyTorch, which causes the HIP detection to fail.

### Patching ROCm 6.0

If you are on ROCm 6.0, apply the following patch before building. This is not required for ROCm 6.1 onwards.

1. Locate your ROCm installation directory. This is typically found at `/opt/rocm/`, but may vary depending on your installation.

2. Apply the Patch. Run with `sudo` in case you encounter permission issues.
   ```bash
    patch /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h < rocm_patch/rocm6_0.patch 
   ```
