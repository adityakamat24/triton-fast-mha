# fa_triton/bench.py
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import torch
import torch.nn.functional as F
from time import perf_counter
from .fa_triton import triton_mha


def time_ms(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = perf_counter(); fn(); torch.cuda.synchronize()
        times.append((perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times)//2]


def torch_sdpa_math(q, k, v, causal: bool):
    try:
        from torch.nn.attention import sdpa_kernel
        with sdpa_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    except Exception:
        from torch.backends.cuda import sdp_kernel
        with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


@torch.no_grad()
def run_case(B=1, H=8, N=1024, D=64, dtype=torch.float16, causal=True):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    ref = torch_sdpa_math(q, k, v, causal)
    out = triton_mha(q, k, v, causal)
    err = (ref - out).abs().max().item()
    print(f"Max abs error (Triton vs PyTorch MATH): {err:.3e}")

    t_math = time_ms(lambda: torch_sdpa_math(q, k, v, causal), iters=120, warmup=40)
    t_tri  = time_ms(lambda: triton_mha(q, k, v, causal), iters=200, warmup=40)
    print(f"PyTorch SDPA (MATH): {t_math:.2f} ms")
    print(f"Triton fused SDPA: {t_tri:.2f} ms")
    print(f"Speedup vs MATH: {t_math / t_tri:.2f}x")


if __name__ == "__main__":
    run_case(B=1, H=8, N=1024, D=64, dtype=torch.float16, causal=True)
