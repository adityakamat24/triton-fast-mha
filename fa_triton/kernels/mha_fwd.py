# fa_triton/mha_fwd.py
import triton
import triton.language as tl

TUNING_CONFIGS = [
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=2),
]

@triton.autotune(configs=TUNING_CONFIGS, key=["N_CTX", "D_HEAD"])
@triton.jit
def mha_fwd_kernel(
    Q, K, V, O,                                    
    stride_qbh, stride_qm, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_om, stride_od,
    BH,                                            
    N_CTX,                                         
    D_HEAD: tl.constexpr,                          
    sm_scale,                                      
    CAUSAL: tl.constexpr,                          
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  
):

    pid_m  = tl.program_id(0)          
    pid_bh = tl.program_id(1)          

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n0 = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)

    row_mask = offs_m < N_CTX
    q_ptr = Q + pid_bh * stride_qbh
    k_ptr = K + pid_bh * stride_kbh
    v_ptr = V + pid_bh * stride_vbh
    o_ptr = O + pid_bh * stride_obh

    tl.multiple_of(offs_d, 8)
    tl.max_contiguous(offs_d, 8)


    q = tl.load(
        q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd),
        mask=row_mask[:, None], other=0.0
    ).to(tl.float16)
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D_HEAD), tl.float32)
    n0 = 0
    while n0 < N_CTX:
        offs_n = n0 + offs_n0
        col_mask = offs_n < N_CTX
        k = tl.load(
            k_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd),
            mask=col_mask[:, None], other=0.0
        ).to(tl.float16)
        v = tl.load(
            v_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd),
            mask=col_mask[:, None], other=0.0
        ).to(tl.float16)

        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale  # [BM, BN]
        if CAUSAL:
            causal = offs_n[None, :] <= offs_m[:, None]
            scores = tl.where(causal, scores, -float("inf"))
        scores = tl.where(col_mask[None, :], scores, -float("inf"))

        m_ij  = tl.max(scores, 1)
        m_new = tl.maximum(m_i, m_ij)
        p     = tl.exp(scores - m_new[:, None])             # fp32
        alpha = tl.exp(m_i - m_new)                         # fp32
        l_i   = l_i * alpha + tl.sum(p, 1)                  # fp32

        acc   = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)

        m_i = m_new
        n0 += BLOCK_N

    o = (acc / l_i[:, None]).to(tl.float16)
    tl.store(
        o_ptr + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_od),
        o, mask=row_mask[:, None]
    )
