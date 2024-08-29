import os
import numba
import math
import numpy as np
from numba import cuda
import numba.cuda
import torch
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"


d = 32
sram = d * 4 * 4
s = d * 4 + 2
B_r, B_c = math.ceil(sram / (d * 4)), math.ceil(sram / (d * 4))


# @numba.cuda.jit
@cuda.jit
def flash_attn_kernel(q, k, v, out, s, T_r, T_c, l, m):
    thread_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # k,v,q are shape (s,d)
    K_shared = cuda.shared.array(shape=(B_c, d), dtype=numba.float32)
    V_shared = cuda.shared.array(shape=(B_c, d), dtype=numba.float32)
    Q_shared = cuda.shared.array(shape=(B_r, d), dtype=numba.float32)

    for j in range(T_r):
        r_idx = j * B_r + thread_i
        if r_idx < s:
            for c in range(Q_shared.shape[1]):
                Q_shared[thread_i, c] = q[r_idx, c]
        cuda.syncthreads()
        for i in range(T_c):
            c_idx = i * B_c + thread_i
            if c_idx < s:
                for c in range(K_shared.shape[1]):
                    K_shared[thread_i, c] = k[c_idx, c]
                for c in range(V_shared.shape[1]):
                    V_shared[thread_i, c] = v[c_idx, c]
            cuda.syncthreads()
            if r_idx < s:
                # Compute attention scores Sij = Qi * Kj^T
                for b_c in range(i * B_c, min((i + 1) * B_c, s)):
                    b_c = b_c - i * B_c
                    curr_l = l[r_idx]
                    curr_m = m[r_idx]
                    Sij = 0.0
                    for k_dim in range(d):
                        Sij += Q_shared[thread_i, k_dim] * K_shared[b_c, k_dim]

                    new_m = max(curr_m, Sij)
                    exp_Sij = math.exp(Sij - new_m)
                    exp_max = math.exp(curr_m - new_m)
                    new_l = curr_l * exp_max + exp_Sij

                    # Update output Oi += softmax * Vj
                    for v_dim in range(d):
                        out[r_idx, v_dim] = (
                            out[r_idx, v_dim] * (curr_l * exp_max / new_l)
                            + (exp_Sij / new_l) * V_shared[b_c, v_dim]
                        )

                    l[r_idx] = new_l
                    m[r_idx] = new_m


def attention(q, k, v, mask=None, dropout=None):
    # q,k,v : (b, h, s, d)
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    # attn_logits = attn_logits / torch.sqrt(
    #     torch.tensor(q.size(-1), dtype=torch.float32)
    # )
    attn = torch.nn.functional.softmax(attn_logits, dim=-1)  # (b, h, s, s)
    return torch.matmul(attn, v), attn, attn_logits


def main():
    grid_dim = 1
    block_dim = B_r
    # set torch seed
    torch.manual_seed(0)
    q = torch.randn(s, d).cuda()
    k = torch.randn(s, d).cuda()
    v = torch.randn(s, d).cuda()
    out = torch.zeros(s, d).cuda()
    l = torch.zeros(s).cuda()
    m = -torch.ones(s).cuda() * math.inf
    T_r, T_c = math.ceil(s / B_r), math.ceil(s / B_c)
    print(B_c, B_r, s, d, T_r, T_c)  # 4 4 130 32 33 33

    q_numba = cuda.as_cuda_array(q)
    k_numba = cuda.as_cuda_array(k)
    v_numba = cuda.as_cuda_array(v)
    out_numba = cuda.as_cuda_array(out)
    l = cuda.as_cuda_array(l)
    m = cuda.as_cuda_array(m)

    flash_attn_kernel[grid_dim, block_dim](
        q_numba, k_numba, v_numba, out_numba, s, T_r, T_c, l, m
    )
    out_numba = out_numba.copy_to_host()
    print("out_numba", out_numba, out_numba.shape)

    out_attn = attention(
        q.unsqueeze(0).unsqueeze(0),
        k.unsqueeze(0).unsqueeze(0),
        v.unsqueeze(0).unsqueeze(0),
    )
    out_attn = out_attn[0].squeeze().cpu()
    print("out_attn", out_attn, out_attn.shape)
    print(np.allclose(out_attn.numpy(), out_numba, atol=1e-2))


if __name__ == "__main__":
    main()
