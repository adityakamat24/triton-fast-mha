# scripts/bench_sweep.py
import argparse, csv, os, time, math
from statistics import median
import torch

# import your kernel
from fa_triton.fa_triton import triton_mha

def cuda_events_ms(fn, warmup=10, iters=50):
    """Accurate GPU timing with CUDA events."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms

def once(B,H,N,D,dtype,causal, torch_backend="math", iters=50, warmup=10):
    """Return (torch_ms, triton_ms, max_abs_err)."""
    q = torch.randn(B,H,N,D, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Torch (force MATH backend so it’s a fair baseline on all systems)
    def torch_op():
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        torch_ms = cuda_events_ms(torch_op, warmup, iters)
        ref = torch_op()

    # Triton
    def triton_op():
        return triton_mha(q, k, v, causal=causal)
    triton_ms = cuda_events_ms(triton_op, warmup, iters)
    out = triton_op()

    max_err = (out - ref).abs().max().item()
    return torch_ms, triton_ms, max_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns",  nargs="+", type=int, default=[512,1024,2048], help="sequence lengths")
    ap.add_argument("--Ds",  nargs="+", type=int, default=[64,128],        help="head dims")
    ap.add_argument("--B",   type=int, default=1)
    ap.add_argument("--H",   type=int, default=8)
    ap.add_argument("--dtype", choices=["fp16","bf16"], default="fp16")
    ap.add_argument("--causal", action="store_true", default=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dtype = torch.float16 if args.dtype=="fp16" else torch.bfloat16

    device = torch.cuda.get_device_name(0)
    torch_v = torch.__version__
    try:
        import triton as _triton
        triton_v = _triton.__version__
    except Exception:
        triton_v = "unknown"

    rows = []
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device","torch","triton","dtype","B","H","N","D","torch_ms","triton_ms","speedup","max_abs_err"])
        for N in args.Ns:
            for D in args.Ds:
                torch_ms, triton_ms, max_err = once(args.B, args.H, N, D, dtype, args.causal, iters=args.iters, warmup=args.warmup)
                speed = torch_ms / triton_ms if triton_ms>0 else float("nan")
                row = [device, torch_v, triton_v, args.dtype, args.B, args.H, N, D,
                       f"{torch_ms:.3f}", f"{triton_ms:.3f}", f"{speed:.2f}", f"{max_err:.3e}"]
                w.writerow(row); rows.append(row)
                print(f"B{args.B} H{args.H} N{N} D{D} -> torch {torch_ms:.3f} ms | triton {triton_ms:.3f} ms | x{speed:.2f} | err {max_err:.2e}")
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA GPU required"
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    main()
