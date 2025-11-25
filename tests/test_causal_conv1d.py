# Copyright (C) 2024, Tri Dao.

import math

import torch
import torch.nn.functional as F

import pytest

from einops import rearrange

from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_ref
from causal_conv1d.causal_conv1d_interface import causal_conv1d_update, causal_conv1d_update_ref
from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states, causal_conv1d_varlen_states_ref


@pytest.mark.parametrize("return_final_states", [False, True])
# @pytest.mark.parametrize("return_final_states", [True])
@pytest.mark.parametrize("has_initial_states", [False, True])
# @pytest.mark.parametrize("has_initial_states", [False])
@pytest.mark.parametrize("channel_last", [False, True])
# @pytest.mark.parametrize('channel_last', [True])
@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize("width", [2, 3, 4, 5, 6, 7, 8])
# @pytest.mark.parametrize('width', [3])
@pytest.mark.parametrize(
    "seqlen", [1, 2, 8, 16, 32, 64, 128, 129, 130, 151, 256, 372, 512, 784, 1024, 1134, 2048, 4096]
)
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 512, 784, 1024, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [128])
@pytest.mark.parametrize('dim', [64, 4096 + 32])
# @pytest.mark.parametrize('dim', [64])
def test_causal_conv1d(dim, seqlen, width, has_bias, silu_activation, itype, channel_last, has_initial_states, return_final_states):
    if not channel_last and (has_initial_states or return_final_states):
        pytest.skip("Only channel_last support initial_states or return_final_states")
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch = 2
    # batch = 1
    if not channel_last:
        x = torch.randn(batch, 4096 + dim + 64, seqlen, device=device, dtype=itype)[:, 4096:4096 + dim, :].requires_grad_()
    else:
        x = rearrange(
            torch.randn(batch, seqlen, 4096 + dim + 64, device=device, dtype=itype)[:, :, 4096:4096 + dim], "b s d -> b d s"
        ).requires_grad_()
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    if has_initial_states:
        initial_states = torch.randn(batch, width - 1, dim, device=device, dtype=itype).transpose(1, 2).requires_grad_()
    else:
        initial_states = None
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    initial_states_ref = initial_states.detach().clone().requires_grad_() if initial_states is not None else None
    activation = None if not silu_activation else "silu"
    out = causal_conv1d_fn(x, weight, bias, initial_states=initial_states, return_final_states=return_final_states,
                           activation=activation)
    out_ref = causal_conv1d_ref(x_ref, weight_ref, bias_ref, initial_states=initial_states_ref, return_final_states=return_final_states, activation=activation)
    if return_final_states:
        out, final_states = out
        out_ref, final_states_ref = out_ref
        print(f"Final states max diff: {(final_states - final_states_ref).abs().max().item()}")
        print(f"Final states mean diff: {(final_states - final_states_ref).abs().mean().item()}")
        assert torch.allclose(final_states, final_states_ref, rtol=rtol, atol=atol)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    if return_final_states:
        out += F.sigmoid(final_states).sum(dim=-1, keepdim=True)
        out_ref += F.sigmoid(final_states_ref).sum(dim=-1, keepdim=True)

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)

    print(f"dx max diff: {(x.grad - x_ref.grad).abs().max().item()}")
    print(f"dweight max diff: {(weight.grad - weight_ref.grad).abs().max().item()}")
    if has_bias:
        print(f"dbias max diff: {(bias.grad - bias_ref.grad).abs().max().item()}")
    if has_initial_states:
        print(f"dinitial_states max diff: {(initial_states.grad - initial_states_ref.grad).abs().max().item()}")

    assert torch.allclose(x.grad, x_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)
    assert torch.allclose(weight.grad, weight_ref.grad, rtol=rtolw, atol=atolw)
    if has_bias:
        assert torch.allclose(bias.grad, bias_ref.grad, rtol=rtolw, atol=atolw)
    if has_initial_states:
        assert torch.allclose(initial_states.grad, initial_states_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize("has_cache_seqlens", [False, True])
# @pytest.mark.parametrize('has_cache_seqlens', [True])
@pytest.mark.parametrize("seqlen", [1, 4, 5])
# @pytest.mark.parametrize('seqlen', [4])
@pytest.mark.parametrize("width", [2, 3, 4, 5, 6, 7, 8])
# @pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_causal_conv1d_update(dim, width, seqlen, has_cache_seqlens, has_bias, silu_activation, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch = 64
    # batch = 1
    # dim = 64
    x = torch.randn(batch, seqlen, dim, device=device, dtype=itype).transpose(-1, -2)
    state_len = torch.randint(width - 1, width + 10, (1,)).item()
    conv_state = torch.randn(batch, state_len, dim, device=device, dtype=itype).transpose(-1, -2)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    conv_state_ref = conv_state.detach().clone()
    activation = None if not silu_activation else "silu"
    cache_seqlens = (torch.randint(0, 1024, (batch,), dtype=torch.int32, device=device)
                     if has_cache_seqlens else None)
    out = causal_conv1d_update(x, conv_state, weight, bias, activation=activation, cache_seqlens=cache_seqlens)
    out_ref = causal_conv1d_update_ref(x, conv_state_ref, weight, bias, activation=activation, cache_seqlens=cache_seqlens)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.equal(conv_state, conv_state_ref)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize("has_cache_seqlens", [False, True])
# @pytest.mark.parametrize('has_cache_seqlens', [True])
@pytest.mark.parametrize("seqlen", [1, 4, 5])
# @pytest.mark.parametrize('seqlen', [4])
@pytest.mark.parametrize("width", [2, 3, 4, 5, 6, 7, 8])
# @pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_causal_conv1d_update_with_batch_gather(dim, width, seqlen, has_cache_seqlens, has_bias, silu_activation, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch = 64
    # batch = 1
    # dim = 64
    x = torch.randn(batch, seqlen, dim, device=device, dtype=itype).transpose(-1, -2)
    state_len = torch.randint(width - 1, width + 10, (1,)).item()

    total_entries = 10 * batch
    conv_state = torch.randn(total_entries, state_len, dim, device=device, dtype=itype).transpose(-1, -2)
    conv_state_indices = torch.randperm(total_entries)[:batch].to(dtype=torch.int32, device=device)

    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    conv_state_ref = conv_state[conv_state_indices, :].detach().clone()
    activation = None if not silu_activation else "silu"
    cache_seqlens = (torch.randint(0, 1024, (batch,), dtype=torch.int32, device=device)
                     if has_cache_seqlens else None)
    out = causal_conv1d_update(x, conv_state, weight, bias, activation=activation,
                               cache_seqlens=cache_seqlens, conv_state_indices=conv_state_indices)
    out_ref = causal_conv1d_update_ref(x, conv_state_ref, weight, bias, activation=activation, cache_seqlens=cache_seqlens)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.equal(conv_state[conv_state_indices, :], conv_state_ref)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize("has_cache_seqlens", [False])#, True])
# @pytest.mark.parametrize('has_cache_seqlens', [True])
@pytest.mark.parametrize("seqlen", [1, 4, 5])
# @pytest.mark.parametrize('seqlen', [4])
@pytest.mark.parametrize("width", [2, 3, 4, 5, 6, 7, 8])
# @pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_causal_conv1d_update_with_padding(dim, width, seqlen, has_cache_seqlens, has_bias, silu_activation, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    torch.random.manual_seed(0)
    batch = 64
    x = torch.randn(batch, seqlen, dim, device=device, dtype=itype).transpose(-1, -2)
    state_len = torch.randint(width - 1, width + 10, (1,)).item()

    total_entries = 10 * batch
    conv_state = torch.randn(total_entries, state_len, dim, device=device, dtype=itype).transpose(-1, -2)

    # Introduce padding by setting some indices to -1
    num_valid_requests = batch // 2
    num_padded_requests = batch - num_valid_requests

    valid_indices = torch.randperm(total_entries)[:num_valid_requests].to(device=device)
    padded_indices = torch.full((num_padded_requests,), -1, device=device)

    # Shuffle the valid and padded indices together
    combined_indices = torch.cat([valid_indices, padded_indices])
    perm = torch.randperm(batch, device=device)
    conv_state_indices = combined_indices[perm].to(dtype=torch.int32)

    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32) if has_bias else None

    activation = "silu" if silu_activation else None

    has_cache_seqlens = False
    cache_seqlens = None
    #cache_seqlens = (torch.randint(0, 1024, (batch,), dtype=torch.int32, device=device)
    #                 if has_cache_seqlens else None)

    # Clone original state for later comparison
    conv_state_original = conv_state.clone()

    # Run the main function with padded indices
    out = causal_conv1d_update(x, conv_state, weight, bias, activation=activation,
                               cache_seqlens=cache_seqlens, conv_state_indices=conv_state_indices)

    # Manually compute the reference output and expected final state
    out_ref = torch.zeros_like(out)
    conv_state_expected = conv_state_original.clone()
    valid_mask = conv_state_indices != -1

    # Only compute the reference for the valid (non-padded) requests
    if num_valid_requests > 0:
        x_valid = x[valid_mask]
        conv_state_indices_valid = conv_state_indices[valid_mask]
        # This will be modified in-place by the ref function to get the expected updated state
        conv_state_valid_updated = conv_state_original[conv_state_indices_valid, :].detach().clone()
        cache_seqlens_valid = cache_seqlens[valid_mask] if has_cache_seqlens else None

        out_ref_valid = causal_conv1d_update_ref(x_valid, conv_state_valid_updated, weight, bias,
                                                 activation=activation, cache_seqlens=cache_seqlens_valid)
        out_ref[valid_mask] = out_ref_valid
        # Place the updated states into our expected full state tensor
        conv_state_expected[conv_state_indices_valid] = conv_state_valid_updated


    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")

    # The output for padded tokens should be exactly zero
    assert torch.all(out[~valid_mask] == 0)
    assert torch.allclose(out[valid_mask], out_ref[valid_mask], rtol=rtol, atol=atol)

    # Check that conv_state was updated correctly for valid tokens and untouched otherwise
    assert torch.equal(conv_state, conv_state_expected)


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_causal_conv1d_get_states(dim, itype):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    seqlens = torch.randint(1, 32, (100,), device=device)
    total_seqlen = seqlens.sum().item()
    x = torch.randn(total_seqlen, dim, device=device, dtype=itype)
    cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0))
    state_len = 20
    out = causal_conv1d_varlen_states(x, cu_seqlens, state_len)
    out_ref = causal_conv1d_varlen_states_ref(x, cu_seqlens, state_len)
    assert torch.equal(out, out_ref)


# @pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize('channel_last', [True])
# @pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('itype', [torch.bfloat16])
# @pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize('silu_activation', [True])
# @pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize('has_bias', [True])
# @pytest.mark.parametrize("width", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize(
    # "seqlen", [8, 16, 32, 64, 128, 151, 256, 372, 512, 784, 1024, 1134, 2048, 4096]
    "seqlen", [2048]
)
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 512, 784, 1024, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [128])
def test_causal_conv1d_race_condition(seqlen, width, has_bias, silu_activation, itype, channel_last):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch = 2
    # batch = 1
    dim = 4096 + 32  # Try dim not divisible by 64
    # dim = 64
    if not channel_last:
        x = torch.randn(batch, 4096 + dim + 64, seqlen, device=device, dtype=itype)[:, 4096:4096 + dim, :].requires_grad_()
    else:
        x = rearrange(
            torch.randn(batch, seqlen, 4096 + dim + 64, device=device, dtype=itype)[:, :, 4096:4096 + dim], "b s d -> b d s"
        ).requires_grad_()
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    activation = None if not silu_activation else "silu"
    out0 = causal_conv1d_fn(x, weight, bias, activation=activation)
    g = torch.randn_like(out0)
    dx0, dw0, db0 = torch.autograd.grad(out0, (x, weight, bias), g)
    dw_atol = 1e-4
    db_atol = 1e-4

    for i in range(10000):
        out = causal_conv1d_fn(x, weight, bias, activation=activation)
        dx, dw, db = torch.autograd.grad(out, (x, weight, bias), g)
        dw_equal = torch.allclose(dw, dw0, atol=dw_atol)
        # if not dw_equal:
        #     breakpoint()
        if has_bias:
            db_equal = torch.allclose(db, db0, atol=db_atol)
            # if not db_equal:
            #     breakpoint()
        assert torch.equal(out, out0)
        assert torch.equal(dx, dx0)
        assert dw_equal
        if has_bias:
            assert dw_equal


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [False])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [False])
@pytest.mark.parametrize("width", [2, 3, 4, 5, 6, 7, 8])
# @pytest.mark.parametrize('width', [2])
@pytest.mark.parametrize(
    "seqlen", [8, 16, 32, 64, 128, 151, 256, 372, 512, 784, 1024, 1134, 2048, 4096]
)
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 512, 784, 1024, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [2048])
@pytest.mark.parametrize('dim', [64, 4096 + 32])
# @pytest.mark.parametrize('dim', [64])
def test_causal_conv1d_varlen(dim, seqlen, width, has_bias, silu_activation, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(seqlen + dim + width)
    batch = 3
    seqlens = []
    for b in range(batch):
        nsplits = torch.randint(1, 5, (1,)).item()
        eos_pos = torch.randperm(seqlen - 1)[:nsplits].sort().values
        seqlens.append(torch.diff(torch.cat([torch.tensor([-1]), eos_pos, torch.tensor([seqlen - 1])])).tolist())
        assert sum(seqlens[-1]) == seqlen
        assert all(s > 0 for s in seqlens[-1])
    # Only support channel_last
    x = rearrange(
        torch.randn(batch, seqlen, 4096 + dim + 64, device=device, dtype=itype)[:, :, 4096:4096 + dim], "b s d -> b d s"
    ).requires_grad_()
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    seq_idx = torch.stack([torch.cat([torch.full((s,), i, dtype=torch.int32, device=device) for i, s in enumerate(sl)], dim=0)
                           for sl in seqlens], dim=0)
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    activation = None if not silu_activation else "silu"
    out = causal_conv1d_fn(x, weight, bias, seq_idx=seq_idx, activation=activation)
    out_ref = []
    for b in range(batch):
        out_ref_b = []
        for x_s in torch.split(x_ref[[b]], seqlens[b], dim=2):
            out_ref_b.append(causal_conv1d_ref(x_s, weight_ref, bias_ref, activation=activation))
        out_ref.append(torch.cat(out_ref_b, dim=2))
    out_ref = torch.cat(out_ref, dim=0)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    g = torch.randn_like(out)
    out_ref.backward(g)
    out.backward(g)

    print(f"dx max diff: {(x.grad - x_ref.grad).abs().max().item()}")
    print(f"dweight max diff: {(weight.grad - weight_ref.grad).abs().max().item()}")
    if has_bias:
        print(f"dbias max diff: {(bias.grad - bias_ref.grad).abs().max().item()}")

    assert torch.allclose(x.grad, x_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)
    assert torch.allclose(weight.grad, weight_ref.grad, rtol=rtolw, atol=atolw)
    if has_bias:
        assert torch.allclose(bias.grad, bias_ref.grad, rtol=rtolw, atol=atolw)

@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize("width", [2, 3, 4, 5, 6, 7, 8])
# @pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize(
    "seqlen", [128, 256, 512, 1024]
)
# @pytest.mark.parametrize('seqlen', [128])
@pytest.mark.parametrize('dim', [64, 4096 + 32])
# @pytest.mark.parametrize('dim', [64])
def test_causal_conv1d_varlen_padding(dim, seqlen, width, has_bias, silu_activation, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(seqlen + dim + width)
    batch = 3
    max_seqlen = seqlen

    # Generate sequences of varying lengths for each batch item, some shorter than max_seqlen
    true_seqlens = [torch.randint(max_seqlen // 2, max_seqlen + 1, (1,)).item() for _ in range(batch)]

    seqlens = []
    for b in range(batch):
        # Within each true sequence, we can have multiple sub-sequences
        nsplits = torch.randint(1, 4, (1,)).item()
        eos_pos = torch.randperm(true_seqlens[b] - 1)[:nsplits].sort().values
        sl = torch.diff(torch.cat([torch.tensor([-1]), eos_pos, torch.tensor([true_seqlens[b] - 1])])).tolist()
        seqlens.append(sl)
        assert sum(sl) == true_seqlens[b]
        assert all(s > 0 for s in sl)

    # Only support channel_last
    x = rearrange(
        torch.randn(batch, max_seqlen, 4096 + dim + 64, device=device, dtype=itype)[:, :, 4096:4096 + dim], "b s d -> b d s"
    ).requires_grad_()

    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True) if has_bias else None

    # Construct seq_idx with -1 for padding
    seq_idx_list = []
    conv_batch_idx_counter = 0
    for b in range(batch):
        seq_idx_b = []
        for s in seqlens[b]:
            seq_idx_b.append(torch.full((s,), conv_batch_idx_counter, dtype=torch.int32, device=device))
            conv_batch_idx_counter += 1
        seq_idx_b = torch.cat(seq_idx_b, dim=0)
        # Add padding indices
        padding_len = max_seqlen - true_seqlens[b]
        assert padding_len > 0
        seq_idx_b = torch.cat([seq_idx_b, torch.full((padding_len,), -1, dtype=torch.int32, device=device)], dim=0)
        seq_idx_list.append(seq_idx_b)
    seq_idx = torch.stack(seq_idx_list, dim=0)

    x_ref = x.detach().clone()
    weight_ref = weight.detach().clone()
    bias_ref = bias.detach().clone() if bias is not None else None
    activation = "silu" if silu_activation else None

    # Run forward pass with padding
    out = causal_conv1d_fn(x, weight, bias, seq_idx=seq_idx, activation=activation)

    # Manual reference calculation
    out_ref = torch.zeros_like(x_ref)
    for b in range(batch):
        out_ref_b = []
        # We only process the true sequence part
        x_b_unpadded = x_ref[[b], :, :true_seqlens[b]]
        # Split into sub-sequences
        for x_s in torch.split(x_b_unpadded, seqlens[b], dim=2):
            out_ref_b.append(causal_conv1d_ref(x_s, weight_ref, bias_ref, activation=activation))
        out_ref[b, :, :true_seqlens[b]] = torch.cat(out_ref_b, dim=2)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
