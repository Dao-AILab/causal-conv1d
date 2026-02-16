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

## Additional Prerequisites for AMD cards

### Patching ROCm

If you are on ROCm 6.0, run the following steps to avoid errors during compilation. This is not required for ROCm 6.1 onwards.

1. Locate your ROCm installation directory. This is typically found at `/opt/rocm/`, but may vary depending on your installation.

2. Apply the Patch. Run with `sudo` in case you encounter permission issues.
   ```bash
    patch /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h < rocm_patch/rocm6_0.patch 
   ```
# causal_conv1d 在windows 安装（MSVC上自行编译）
# 在windows （MSVC上自行编译）
## 1.创建编译环境：
    使用 visual studio  下载 windows 11/10 sdk 并下载MSVC编译器
    配置环境变量
    安装依赖ninja
## 2. 适配causal-conv1d源码

    clone 本项目 修改setup.py中 的
        
        ```python
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_89,code=sm_89")  #选择自己的显卡架构进行编译，减少编译时间和生成文件大小
        ```

    ### 如果是amd显卡需要把
     1. causal-conv1d\csrc\causal_conv1d.cpp 
     2. causal-conv1d\csrc\causal_conv1d_fwd.cu
     3. causal-conv1d\csrc\causal_conv1d_bwd.cu
     4. causal-conv1d\csrc\causal_conv1d_common.h 
    中
        #ifndef USE_ROCM
        else
        #endif
    改为保留else的代码
## 3. 编译代码
    使用x64 Native Tools Command Prompt for VS 2022
    转到项目下
    # 1. 告诉编译器我们要用 SDK
    set DISTUTILS_USE_SDK=1

    # 2. 限制只用 1 个线程编译，虽然慢点，但不会爆内存
    set MAX_JOBS=1


    # 3. 正式开始安装
    pip install . --no-build-isolation

推荐pr：[text](https://github.com/Dao-AILab/causal-conv1d/pull/93)