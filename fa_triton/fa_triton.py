# fa_triton/fa_triton.py
import math
import torch
import triton
from .kernels.mha_fwd import mha_fwd_kernel

@torch.no_grad()
def triton_mha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == torch.float16 and k.dtype == torch.float16 and v.dtype == torch.float16
    B, H, N, D = q.shape
    BH = B * H

    Q = q.contiguous().view(BH, N, D)
    K = k.contiguous().view(BH, N, D)
    V = v.contiguous().view(BH, N, D)
    O = torch.empty_like(Q)

    stride_qbh, stride_qm, stride_qd = Q.stride()
    stride_kbh, stride_kn, stride_kd = K.stride()
    stride_vbh, stride_vn, stride_vd = V.stride()
    stride_obh, stride_om, stride_od = O.stride()
    sm_scale = 1.0 / math.sqrt(D)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), BH)

    mha_fwd_kernel[grid](
        Q, K, V, O,
        stride_qbh, stride_qm, stride_qd,
        stride_kbh, stride_kn, stride_kd,
        stride_vbh, stride_vn, stride_vd,
        stride_obh, stride_om, stride_od,
        BH, N, D, sm_scale,
        CAUSAL=causal,
    )
    return O.view(B, H, N, D)
