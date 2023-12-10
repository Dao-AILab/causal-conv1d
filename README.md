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
