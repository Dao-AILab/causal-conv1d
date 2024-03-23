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

## How to install

Make sure the CUDA version of the GPU is >= 11.6.

After clone the repository,

```bash
pip install .
```

Note that the default behavior of the installer is to download the release file from GitHub Release page. If this is not desired behavior, one might use environment variables to force build locally.

### How to build locally

Make sure the version of NVCC is 11.6 or above, but is below the CUDA runtime version of the GPU.

To check the NVCC version,

```bash
nvcc -V
```

To check CUDA version,

```bash
nvidia-smi
```

Once everything is set, we can start building the package.

```bash
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
```

The build may take several minutes to complete.

### Note

Ubuntu 22.04 LTS Jammy ships with NVCC version 11.5. To install a later NVCC version without committing the entire machine to Ubuntu 23.10 Mantic, one might look into *apt package pinning* to get the later NVCC from Ubuntu 23.10 Mantic repoistory.
