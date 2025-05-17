# Copyright (c) 2024, Tri Dao.

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

import causal_conv1d_cuda

@torch.library.custom_op(
    "custom_ops::causal_conv1d_fwd",
    device_types=["cuda"],
    mutates_args=(),
    schema="(Tensor x, Tensor weight, Tensor? bias, Tensor? seq_idx, Tensor? initial_states, Tensor? final_states_out, bool activation) -> (Tensor, Tensor?)",
)
def custom_causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    seq_idx: Optional[torch.Tensor],
    initial_states: Optional[torch.Tensor],
    final_states_out: Optional[torch.Tensor],
    activation: bool,
):
    pass

@torch.library.register_fake("custom_ops::causal_conv1d_fwd")
def custom_causal_conv1d_fwd_fake(
    x, weight, bias, seq_idx, initial_states, final_states_out, activation
):
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    
    out_fake = torch.empty_like(x)
    
    if final_states_out is not None:
        final_states_fake = final_states_out.clone()
    else:
        final_states_fake = x.new_empty(0)
    
    return out_fake, final_states_fake

@torch.library.register_kernel("custom_ops::causal_conv1d_fwd", "cuda")
def custom_causal_conv1d_fwd_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    seq_idx: Optional[torch.Tensor],
    initial_states: Optional[torch.Tensor],
    final_states_out: Optional[torch.Tensor],
    activation: bool,
):
    if x.stride(2) != 1 and x.stride(1) != 1:
        x = x.contiguous()
    
    bias = bias.contiguous() if bias is not None else None
    seq_idx = seq_idx.contiguous() if seq_idx is not None else None
    
    if initial_states is not None and (initial_states.stride(2) != 1 and initial_states.stride(1) != 1):
        initial_states = initial_states.contiguous()
    
    out = causal_conv1d_cuda.causal_conv1d_fwd(
        x, weight, bias, seq_idx, initial_states, final_states_out, activation
    )
    
    return out, final_states_out

@torch.library.custom_op(
    "custom_ops::causal_conv1d_bwd",
    device_types=["cuda"],
    mutates_args=(),
    schema="(Tensor x, Tensor weight, Tensor? bias, Tensor dout, Tensor? seq_idx, Tensor? initial_states, Tensor? dfinal_states, Tensor? dx_out, bool return_dinitial_states, bool activation) -> (Tensor, Tensor, Tensor?, Tensor?)",
)
def custom_causal_conv1d_bwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    dout: torch.Tensor,
    seq_idx: Optional[torch.Tensor],
    initial_states: Optional[torch.Tensor],
    dfinal_states: Optional[torch.Tensor],
    dx_out: Optional[torch.Tensor],
    return_dinitial_states: bool,
    activation: bool,
):
    pass

@torch.library.register_fake("custom_ops::causal_conv1d_bwd")
def custom_causal_conv1d_bwd_fake(
    x, weight, bias, dout, seq_idx, initial_states, dfinal_states, dx_out, return_dinitial_states, activation
):
    dx = torch.empty_like(x)
    dweight = torch.empty_like(weight)
    dbias = torch.empty_like(bias) if bias is not None else None
    dinitial_states = torch.empty_like(initial_states) if return_dinitial_states and initial_states is not None else None
    
    return dx, dweight, dbias, dinitial_states

@torch.library.register_kernel("custom_ops::causal_conv1d_bwd", "cuda")
def custom_causal_conv1d_bwd_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    dout: torch.Tensor,
    seq_idx: Optional[torch.Tensor],
    initial_states: Optional[torch.Tensor],
    dfinal_states: Optional[torch.Tensor],
    dx_out: Optional[torch.Tensor],
    return_dinitial_states: bool,
    activation: bool,
):
    if dout.stride(2) != 1 and dout.stride(1) != 1:
        dout = dout.contiguous()
    
    dx, dweight, dbias, dinitial_states = causal_conv1d_cuda.causal_conv1d_bwd(
        x, weight, bias, dout, seq_idx, initial_states, dfinal_states, dx_out, return_dinitial_states, activation
    )
    
    return dx, dweight, dbias, dinitial_states

def custom_bridge(ctx, *grads):
    dout = grads[0] if grads else ctx.saved_tensors[0].new_empty(0)
    dfinal_states = grads[1] if ctx.return_final_states and len(grads) > 1 else None
    
    x, weight, bias, seq_idx, initial_states = ctx.saved_tensors
    
    dx, dweight, dbias, dinitial_states = torch.ops.custom_ops.causal_conv1d_bwd(
        x,
        weight,
        bias,
        dout,
        seq_idx,
        initial_states,
        dfinal_states,
        None,
        ctx.return_dinitial_states,
        ctx.activation,
    )
    
    return (
        dx,
        dweight,
        dbias if bias is not None else None,
        None,
        dinitial_states if initial_states is not None else None,
        None,
        None,
        None,
    )

def custom_setup_context(ctx, inputs, output):
    (x, weight, bias, seq_idx, initial_states, return_final_states, final_states_out, activation) = inputs
    (out, final_states) = output
    
    ctx.activation = activation if isinstance(activation, bool) else activation in ["silu", "swish"]
    ctx.return_final_states = return_final_states
    ctx.return_dinitial_states = initial_states is not None and initial_states.requires_grad
    
    ctx.save_for_backward(x, weight, bias, seq_idx, initial_states)

torch.library.register_autograd(
    "custom_ops::causal_conv1d_fwd", custom_bridge, setup_context=custom_setup_context
)

def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    
    # Prepare final_states_out if needed but not provided
    if return_final_states and final_states_out is None:
        batch, dim, seqlen = x.shape
        width = weight.shape[1]
        final_states_out = torch.empty(
            batch, width - 1, dim, device=x.device, dtype=x.dtype
        ).transpose(1, 2)
    
    activation_bool = activation in ["silu", "swish"]
    
    out, final_states = torch.ops.custom_ops.causal_conv1d_fwd(
        x, weight, bias, seq_idx, initial_states, final_states_out, activation_bool
    )
    
    if return_final_states:
        return out, final_states
    else:
        return out

def causal_conv1d_ref(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)
    
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not return_final_states else (out, final_states_out)

def causal_conv1d_update(
    x, 
    conv_state, 
    weight, 
    bias=None, 
    activation=None, 
    cache_seqlens=None,
    conv_state_indices=None
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len.
        conv_state_indices: (batch,), dtype int32
        If None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    activation = activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    out = causal_conv1d_cuda.causal_conv1d_update(
        x, conv_state, weight, bias, activation, cache_seqlens, conv_state_indices
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out

def causal_conv1d_update_ref(
    x, 
    conv_state, 
    weight, 
    bias=None, 
    activation=None, 
    cache_seqlens=None
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.
    
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(-(width - 1), 0, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)
