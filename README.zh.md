# 基于 CUDA 实现并提供 PyTorch 接口的因果深度可分离一维卷积 (Causal Depthwise Conv1d)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

特性：
- 支持 fp32、fp16、bf16 数据类型。
- 支持卷积核尺寸（Kernel size）2、3、4。

## 使用方法 (How to use)

```python
from causal_conv1d import causal_conv1d_fn
```

```python
def causal_conv1d_fn(x, weight, bias=None, activation=None):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    activation: 可选 None、"silu" 或 "swish"

    out: (batch, dim, seqlen)
    """
```

等价于如下 PyTorch 实现：
```python
import torch.nn.functional as F

F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)[..., :seqlen]
```

## AMD 显卡额外先决条件 (Additional Prerequisites for AMD cards)

### 为 ROCm 打补丁 (Patching ROCm)

如果您使用的是 ROCm 6.0，请执行以下步骤以避免编译期间发生报错。ROCm 6.1 及更高版本无需执行此步骤。

1. 定位您的 ROCm 安装目录。通常位于 `/opt/rocm/`，但也可能因具体安装方式而异。

2. 应用补丁。如果遇到权限问题，请使用 `sudo` 运行：
   ```bash
   patch /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h < rocm_patch/rocm6_0.patch
   ```

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月30日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
