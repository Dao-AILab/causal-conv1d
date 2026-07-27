import torch
from torch import Tensor

import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_varlen_states(
    X,
    CU_SEQLENS,
    STATES,
    state_len,
    dim,
    stride_x_seqlen, stride_x_dim,
    stride_states_batch, stride_states_seqlen, stride_states_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Pointer offsets must be computed in int64. `rows * stride_x_seqlen` reaches
    # total_tokens * stride_x_seqlen, and X is typically a view into a wider fused
    # projection buffer (so stride_x_seqlen is that buffer's row width, not `dim`),
    # which overflows int32 long before the tensor itself is unreasonable. Triton
    # defaults program ids, tl.arange and int32 loads to int32, so widening must be
    # explicit; `rows`/`cols` then propagate int64 through the address arithmetic.
    batch_idx = tl.program_id(2).to(tl.int64)
    pid_m = tl.program_id(1).to(tl.int64)
    pid_n = tl.program_id(0).to(tl.int64)
    STATES += batch_idx * stride_states_batch
    end_idx = tl.load(CU_SEQLENS + batch_idx + 1).to(tl.int64)
    start_idx = tl.maximum(tl.load(CU_SEQLENS + batch_idx).to(tl.int64), end_idx - state_len)
    rows = end_idx - (pid_m + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(X + rows[:, None] * stride_x_seqlen + cols[None, :] * stride_x_dim,
                mask=(rows[:, None] >= start_idx) & (cols[None, :] < dim),
                other=0)
    rows_states = state_len - (pid_m + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(STATES + rows_states[:, None] * stride_states_seqlen + cols[None, :] * stride_states_dim,
             x,
             mask=(rows_states[:, None] >= 0) & (cols[None, :] < dim))


def causal_conv1d_varlen_states(x: Tensor, cu_seqlens: Tensor, state_len: int) -> Tensor:
    """
    Forward pass only, does not support backward pass.
    Parameters:
        x: (total_tokens, dim)
        cu_seqlens: (batch + 1), must already be sorted. The cumulative sum of the sequence lengths, starting from 0.
        state_len: int. For each cu_seqlens, how many elements from x should be copied to the state.
            If some of those elements belong to a different sequence, the value of the states will be zero.
    Return:
        states: (batch, dim, state_len)
    """
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    states = torch.empty(batch, state_len, dim, dtype=x.dtype, device=x.device).transpose(1, 2)
    BLOCK_M = min(triton.next_power_of_2(state_len), 16)
    BLOCK_N = min(triton.next_power_of_2(dim), 256)
    grid = (triton.cdiv(dim, BLOCK_N), triton.cdiv(state_len, BLOCK_M), batch)
    with torch.cuda.device(x.device.index):
        _causal_conv1d_varlen_states[grid](
            x,
            cu_seqlens,
            states,
            state_len,
            dim,
            x.stride(0), x.stride(1),
            states.stride(0), states.stride(2), states.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
    return states


def causal_conv1d_varlen_states_ref(x: Tensor, cu_seqlens: Tensor, state_len: int) -> Tensor:
    """
    Forward pass only, does not support backward pass.
    Parameters:
        x: (total_tokens, dim)
        cu_seqlens: (batch + 1), must already be sorted. The cumulative sum of the sequence lengths, starting from 0.
        state_len: int. For each cu_seqlens, how many elements from x should be copied to the state.
            If some of those elements belong to a different sequence, the value of the states will be zero.
    Return:
        states: (batch, dim, state_len)
    """
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    states = torch.zeros(batch, state_len, dim, dtype=x.dtype, device=x.device).transpose(1, 2)
    for i in range(batch):
        end_idx = cu_seqlens[i + 1]
        start_idx = torch.maximum(cu_seqlens[i], end_idx - state_len)
        states[i, :, -(end_idx - start_idx):] = x[start_idx:end_idx].T
    return states
