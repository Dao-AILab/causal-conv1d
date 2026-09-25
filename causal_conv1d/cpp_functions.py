# Copyright (c) 2024, Tri Dao.

import os

import torch

# Import registers the stable-ABI ops into the `causal_conv1d` namespace.
from . import _C  # noqa: F401

causal_conv1d_cuda = torch.ops.causal_conv1d


LIBRARY_NAME = "DaoAILab"


def _use_deterministic_mode() -> bool:
    # CAUSAL_CONV1D_DETERMINISTIC overrides; otherwise follow torch's global flag.
    env = os.environ.get("CAUSAL_CONV1D_DETERMINISTIC")
    if env == "1":
        return True
    if env == "0":
        return False
    return torch.are_deterministic_algorithms_enabled()


@torch.library.custom_op(f"{LIBRARY_NAME}::_causal_conv1d_fwd_cpp", mutates_args={"out", "final_states_out"})
def _causal_conv1d_fwd_cpp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    out: torch.Tensor,
    final_states_out: torch.Tensor | None,
    silu_activation: bool,
) -> None:
    causal_conv1d_cuda.causal_conv1d_fwd(
        x,
        weight,
        bias,
        seq_idx,
        initial_states,
        out,
        final_states_out,
        silu_activation,
    )


@torch.library.custom_op(f"{LIBRARY_NAME}::_causal_conv1d_bwd_cpp", mutates_args={
    "dfinal_states",
    "dx",
    "dweight",
    "dbias",
    "dinitial_states",
})
def _causal_conv1d_bwd_cpp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    dfinal_states: torch.Tensor | None,
    dx: torch.Tensor,
    dweight: torch.Tensor,
    dbias: torch.Tensor | None,
    dinitial_states: torch.Tensor,
    silu_activation: bool,
    deterministic: bool,
) -> None:
    causal_conv1d_cuda.causal_conv1d_bwd(
        x,
        weight,
        bias,
        dout,
        seq_idx,
        initial_states,
        dfinal_states,
        dx,
        dweight,
        dbias,
        dinitial_states,
        silu_activation,
        deterministic,
    )


@torch.library.custom_op(f"{LIBRARY_NAME}::_causal_conv1d_update_cpp", mutates_args={"out", "conv_state"})
def _causal_conv1d_update_cpp(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    silu_activation: bool,
    cache_seqlens: torch.Tensor | None,
    conv_state_indices: torch.Tensor | None,
) -> None:
    causal_conv1d_cuda.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        out,
        silu_activation,
        cache_seqlens,
        conv_state_indices
    )


def causal_conv1d_fwd_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    final_states_out: torch.Tensor | None,
    silu_activation: bool,
) -> torch.Tensor:
    out = torch.empty_like(x)
    _causal_conv1d_fwd_cpp(
        x=x,
        weight=weight,
        bias=bias,
        seq_idx=seq_idx,
        initial_states=initial_states,
        out=out,
        final_states_out=final_states_out,
        silu_activation=silu_activation,
    )
    return out


def causal_conv1d_bwd_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    dfinal_states: torch.Tensor | None,
    dx: torch.Tensor | None,
    return_dinitial_states: torch.Tensor,
    silu_activation: bool,
) -> tuple[torch.Tensor | None]:
    batch_size, dim = x.size()[:2]
    width = weight.size(-1)

    if dx is None:
        dx = torch.empty_like(x)
    dweight = torch.zeros_like(weight, dtype=torch.float32)
    dbias = None
    if bias is not None:
        dbias = torch.zeros_like(bias, dtype=torch.float32)
    dinitial_states = None
    if return_dinitial_states:
        dinitial_states = torch.empty(batch_size, width - 1, dim, device=x.device, dtype=x.dtype).transpose(1, 2)

    _causal_conv1d_bwd_cpp(
        x=x,
        weight=weight,
        bias=bias,
        dout=dout,
        seq_idx=seq_idx,
        initial_states=initial_states,
        dfinal_states=dfinal_states,
        dx=dx,
        dweight=dweight,
        dbias=dbias,
        dinitial_states=dinitial_states,
        silu_activation=silu_activation,
        deterministic=_use_deterministic_mode(),
    )

    dweight = dweight.type_as(weight)
    if dbias is not None:
        dbias = dbias.type_as(bias)
    return dx, dweight, dbias, dinitial_states


def causal_conv1d_update_function(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    silu_activation: bool,
    cache_seqlens: torch.Tensor | None,
    conv_state_indices: torch.Tensor | None,
) -> torch.Tensor:
    out = torch.empty_like(x)
    _causal_conv1d_update_cpp(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=bias,
        out=out,
        silu_activation=silu_activation,
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
    )
    return out
