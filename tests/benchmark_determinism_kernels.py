#!/usr/bin/env python
# Copyright (c) 2024, Tri Dao.

import argparse
import gc
import statistics

import torch

from causal_conv1d.cpp_functions import causal_conv1d_bwd_function


def _peak_memory_mb(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _do_bench(fn, warmup, rep, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        for _ in range(rep):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / rep)
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--dim", type=int, default=4096 + 32)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--width", type=int, default=4, choices=[2, 3, 4])
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--channel-last", action="store_true")
    p.add_argument("--no-bias", action="store_true")
    p.add_argument("--activation", choices=["none", "silu"], default="silu")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--rep", type=int, default=300)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    has_bias = not args.no_bias
    silu = args.activation == "silu"

    device = "cuda"
    if args.channel_last:
        x = torch.randn(args.batch, args.seqlen, args.dim, device=device, dtype=dtype).transpose(1, 2)
    else:
        x = torch.randn(args.batch, args.dim, args.seqlen, device=device, dtype=dtype)
    weight = torch.randn(args.dim, args.width, device=device, dtype=torch.float32)
    bias = torch.randn(args.dim, device=device, dtype=torch.float32) if has_bias else None
    dout = torch.randn_like(x)
    dx = torch.empty_like(x)

    def bwd():
        causal_conv1d_bwd_function(x, weight, bias, dout, None, None, None, dx, False, silu)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"batch={args.batch} dim={args.dim} seqlen={args.seqlen} width={args.width} "
          f"dtype={args.dtype} channel_last={args.channel_last} bias={has_bias} silu={silu}")
    print(f"{'':10} {'ms':>9} {'det_ms':>9} {'overhead':>8} {'MB':>9} {'det_MB':>9} {'overhead':>8}")

    torch.use_deterministic_algorithms(False)
    ms = _do_bench(bwd, args.warmup, args.rep, args.iters)
    mb = _peak_memory_mb(bwd)

    torch.use_deterministic_algorithms(True)
    det_ms = _do_bench(bwd, args.warmup, args.rep, args.iters)
    det_mb = _peak_memory_mb(bwd)

    torch.use_deterministic_algorithms(False)

    ms_pct = (det_ms / ms - 1) * 100 if ms else 0
    mb_pct = (det_mb / mb - 1) * 100 if mb else 0
    print(f"{'bwd':<10} {ms:9.3f} {det_ms:9.3f} {ms_pct:>+7.1f}% {mb:9.1f} {det_mb:9.1f} {mb_pct:>+7.1f}%")


if __name__ == "__main__":
    main()
