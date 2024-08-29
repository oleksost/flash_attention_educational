import os
import numba
import math
from numba import cuda
import numba.cuda
import torch
import pdb
from torch.autograd import Function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"


s = 1024  # seq length
n_heads = 8
d_model = 128  # hidden dim
d_head = d_model // n_heads  # this is in numbers, each number is float32, so 4 bytes
d_head_bytes = d_head * 4
assert d_model % n_heads == 0

b = 32  # batch size
sram = (
    cuda.get_current_device().MAX_SHARED_MEMORY_PER_BLOCK
)  # this is in bytes, e.g. 0xc000 = 49152 bytes
B_r = B_c = math.ceil(
    sram / (3 * d_head_bytes)
)  # because we need to store Q,K,V and O in SRAM <- the number of threads we can run in parallel given the SRAM size, this is also the number of threads per block
# the original algo also puts o in SRAM, but we can't do that here, we write directly in global memory (potentially this can lead to iefficiency)

B_r_bk = B_c_bk = math.ceil(
    sram / (10 * d_head_bytes)
)  # for backward pass, where we need to load much more stuff in SRAM.


@cuda.jit
def flash_attn_forward_kernel(q, k, v, out, T_r, T_c, tau, l_hbm, m_hbm):
    # we run one block per head
    block_b = cuda.blockIdx.x  # coordinate of the batch index
    block_h = cuda.blockIdx.y  # coordinate of the head index
    # then we run B_r threads per block taking care of the s and d dimention
    thread_i = cuda.threadIdx.x

    # k,v,q are shape (b,h,s,d)
    K_shared = cuda.shared.array(shape=(B_c, d_head), dtype=numba.float32)
    V_shared = cuda.shared.array(shape=(B_c, d_head), dtype=numba.float32)
    Q_shared = cuda.shared.array(shape=(B_r, d_head), dtype=numba.float32)

    # here we run in parallel over the s of K and O
    for j in range(T_r):
        r_idx = j * B_r + thread_i
        if r_idx < s:
            for c in range(Q_shared.shape[1]):
                Q_shared[thread_i, c] = q[block_b, block_h, r_idx, c]
        cuda.syncthreads()
        l, m = 0.0, 0.0
        for i in range(T_c):
            c_idx = i * B_c + thread_i
            if c_idx < s:
                for c in range(K_shared.shape[1]):
                    K_shared[thread_i, c] = k[block_b, block_h, c_idx, c]
                for c in range(V_shared.shape[1]):
                    V_shared[thread_i, c] = v[block_b, block_h, c_idx, c]
            cuda.syncthreads()
            if r_idx < s:
                # Compute attention scores Sij = Qi * Kj^T
                for b_c in range(i * B_c, min((i + 1) * B_c, s)):
                    b_c = b_c - i * B_c
                    curr_l = l
                    curr_m = m
                    Sij = 0.0
                    for k_dim in range(Q_shared.shape[1]):
                        Sij += tau * (Q_shared[thread_i, k_dim] * K_shared[b_c, k_dim])

                    new_m = max(curr_m, Sij)
                    exp_Sij = math.exp(Sij - new_m)
                    exp_max = math.exp(curr_m - new_m)
                    new_l = curr_l * exp_max + exp_Sij

                    # Update output Oi += softmax * Vj
                    for v_dim in range(d_head):
                        # this writes each element to the global memory
                        out[block_b, block_h, r_idx, v_dim] = (
                            out[block_b, block_h, r_idx, v_dim]
                            * (curr_l * exp_max / new_l)
                            + (exp_Sij / new_l) * V_shared[b_c, v_dim]
                        )

                    l = new_l
                    m = new_m
            cuda.syncthreads()  # this sync is needed as threads can go ovewrite the same l and m values, also K_shared and V_shared

        if r_idx < s:
            l_hbm[block_b, block_h, r_idx] = l
            m_hbm[block_b, block_h, r_idx] = m


@cuda.jit
def flash_attn_backward_kernel(
    grad_q, grad_k, grad_v, q, k, v, o, grad_out, T_r, T_c, tau, l_hbm, m_hbm
):
    # also grads are b, h, s, d
    # we run one block per head
    block_b = cuda.blockIdx.x  # coordinate of the batch index
    block_h = cuda.blockIdx.y  # coordinate of the head index
    # then we run B_r threads per block taking care of the s and d dimention
    thread_i = cuda.threadIdx.x

    dK_shared = cuda.shared.array(shape=(B_c_bk, d_head), dtype=numba.float32)
    dV_shared = cuda.shared.array(shape=(B_c_bk, d_head), dtype=numba.float32)
    dQ_shared = cuda.shared.array(shape=(B_r_bk, d_head), dtype=numba.float32)
    dO_shared = cuda.shared.array(shape=(B_r_bk, d_head), dtype=numba.float32)

    K_shared = cuda.shared.array(shape=(B_c_bk, d_head), dtype=numba.float32)
    V_shared = cuda.shared.array(shape=(B_c_bk, d_head), dtype=numba.float32)
    Q_shared = cuda.shared.array(shape=(B_r_bk, d_head), dtype=numba.float32)
    O_shared = cuda.shared.array(shape=(B_r_bk, d_head), dtype=numba.float32)

    l_shared = cuda.shared.array(shape=(B_r_bk), dtype=numba.float32)
    m_shared = cuda.shared.array(shape=(B_r_bk), dtype=numba.float32)

    for j in range(T_c):
        c_idx = j * B_c_bk + thread_i
        if c_idx < s:
            # load Kj and Vj, innitialize dK and dV
            for c in range(dK_shared.shape[1]):
                K_shared[thread_i, c] = k[block_b, block_h, c_idx, c]
                V_shared[thread_i, c] = v[block_b, block_h, c_idx, c]
                dK_shared[thread_i, c] = 0.0
                dV_shared[thread_i, c] = 0.0

        cuda.syncthreads()
        for i in range(T_r):
            r_idx = i * B_r_bk + thread_i
            if r_idx < s:
                for c in range(dQ_shared.shape[1]):
                    dQ_shared[thread_i, c] = 0.0 #grad_q[block_b, block_h, r_idx, c]
                    Q_shared[thread_i, c] = q[block_b, block_h, r_idx, c]
                    dO_shared[thread_i, c] = grad_out[block_b, block_h, r_idx, c]
                    O_shared[thread_i, c] = o[block_b, block_h, r_idx, c]

                l_shared[thread_i] = l_hbm[block_b, block_h, r_idx]
                m_shared[thread_i] = m_hbm[block_b, block_h, r_idx]
            cuda.syncthreads()
            if c_idx < s:
                # Re-computing attention scores Sij = Qi * Kj^T
                for rr in range(
                    i * B_r_bk, min((i + 1) * B_r_bk, s)
                ):  # this is from 0 to B_r basically
                    b_r = rr - (i * B_r_bk)
                    Sij = 0.0
                    for k_dim in range(Q_shared.shape[1]):
                        Sij += tau * (Q_shared[b_r, k_dim] * K_shared[thread_i, k_dim])
                    # now we can actually instantiate the attention scores
                    Pij = (math.exp(Sij - m_shared[b_r])) / l_shared[b_r]

                    # update dV
                    for v_dim in range(dV_shared.shape[1]):
                        # its just the sum of the block gradients
                        dV_shared[thread_i, v_dim] += Pij * dO_shared[b_r, v_dim]

                    # dP
                    dPij = 0.0
                    for o_dim in range(dO_shared.shape[1]):
                        dPij += dO_shared[b_r, o_dim] * V_shared[thread_i, o_dim]

                    # compute Di sum
                    Di = 0.0
                    for o_dim in range(dO_shared.shape[1]):
                        Di += dO_shared[b_r, o_dim] * O_shared[b_r, o_dim]

                    dSij = Pij * (dPij - Di)

                    for q_dim in range(dQ_shared.shape[1]):
                        # race condition? <- addressed with atomic add
                        # is this a problem? Cause all threads write to the same grad_q
                        # dQ_shared[i * B_r_bk + b_r, q_dim] += (tau * dSij * K_shared[thread_i, q_dim])  
                        # grad_q[block_b, block_h, i * B_r_bk + b_r, q_dim] += (tau * dSij * K_shared[thread_i, q_dim])
                        cuda.atomic.add(grad_q, (block_b, block_h, (i * B_r_bk) + b_r, q_dim), tau * dSij * K_shared[thread_i, q_dim]) # this is slow?

                    # update dK
                    for k_dim in range(dK_shared.shape[1]):
                        dK_shared[thread_i, k_dim] += (tau * dSij * Q_shared[b_r, k_dim])

            cuda.syncthreads()
            
        if c_idx < s:
            for v_dim in range(dV_shared.shape[1]):
                grad_v[block_b, block_h, c_idx, v_dim] = dV_shared[thread_i, v_dim]
                grad_k[block_b, block_h, c_idx, v_dim] = dK_shared[thread_i, v_dim]
        cuda.syncthreads()


class FlashAttn(Function):
    @staticmethod
    def forward(ctx, q, k, v):

        b, h, s, d_head_local = q.size()
        assert d_head_local == d_head
        # q is b, h, s, d
        # we parallelize by having a block of threads per head, and we have b x h heads
        grid_dim = (b, h)
        assert B_r == B_c
        block_dim = B_c
        # out should be b, h, s, d, where d is
        out = torch.zeros(b, h, s, d_head).to(device)
        l = torch.zeros(b, h, s).to(device)
        m = torch.zeros(b, h, s).to(device)

        q_numba = numba.cuda.as_cuda_array(q.detach())
        k_numba = numba.cuda.as_cuda_array(k.detach())
        v_numba = numba.cuda.as_cuda_array(v.detach())
        out = numba.cuda.as_cuda_array(out)
        l = numba.cuda.as_cuda_array(l)
        m = numba.cuda.as_cuda_array(m)

        T_c, T_r = math.ceil(s / B_c), math.ceil(s / B_r)
        tau = 1 / math.sqrt(q.size(-1))  # scaling constant
        flash_attn_forward_kernel[grid_dim, block_dim](
            q_numba, k_numba, v_numba, out, T_r, T_c, tau, l, m
        )
        cuda.synchronize()
        out_numba = torch.tensor(out.copy_to_host()).to(device)
        l = torch.tensor(l.copy_to_host()).to(device)
        m = torch.tensor(m.copy_to_host()).to(device)

        # we will need to save l and m, but they ae only s long, so we can store them in ctx
        ctx.save_for_backward(l, m, q.detach(), k.detach(), v.detach(), out_numba)

        return out_numba

    @staticmethod
    def backward(ctx, grad_output):
        # backard requires S and P -- logits and softmaxed attention
        dO = grad_output
        # dO is b, h, s, d
        b, h, s, d = dO.size()
        # no need to retrieve anything from storage here
        dO_numba = numba.cuda.as_cuda_array(dO.detach())

        dQ, dK, dV = torch.zeros_like(dO), torch.zeros_like(dO), torch.zeros_like(dO)
        dQ_numba = numba.cuda.as_cuda_array(dQ)
        dK_numba = numba.cuda.as_cuda_array(dK)
        dV_numba = numba.cuda.as_cuda_array(dV)
        T_c, T_r = math.ceil(s / B_c_bk), math.ceil(s / B_r_bk)

        grid_dim = (b, h)
        block_dim = B_c_bk
        l, m, q, k, v, o = ctx.saved_tensors
        tau = 1 / math.sqrt(q.size(-1))  # scaling constant
        l = numba.cuda.as_cuda_array(l.detach())
        m = numba.cuda.as_cuda_array(m.detach())
        q = numba.cuda.as_cuda_array(q.detach())
        k = numba.cuda.as_cuda_array(k.detach())
        v = numba.cuda.as_cuda_array(v.detach())
        o = numba.cuda.as_cuda_array(o.detach())

        # right now this obviously uses too much SRAM, we need to optimize this
        flash_attn_backward_kernel[grid_dim, block_dim](
            dQ_numba, dK_numba, dV_numba, q, k, v, o, dO_numba, T_r, T_c, tau, l, m
        )
        cuda.synchronize()
        dQ = dQ_numba.copy_to_host()
        dK = dK_numba.copy_to_host()
        dV = dV_numba.copy_to_host()

        dQ = torch.tensor(dQ).to(device)
        dK = torch.tensor(dK).to(device)
        dV = torch.tensor(dV).to(device)
        # K and Q gads are still too different from the ones in standard attention
        print("Attention FA grad sum:", dQ.sum(), dK.sum(), dV.sum())
        return dQ, dK, dV


class Attn(Function):
    @staticmethod
    def forward(ctx, q, k, v):
        out, attn, attn_logits = attention(q, k, v)
        # Save intermediate results needed for backward pass
        ctx.save_for_backward(
            q,
            k,
            v,
            attn,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # print("backward")
        # Retrieve saved tensors
        q, k, v, attn = ctx.saved_tensors

        # Compute the gradient w.r.t. v
        grad_v = torch.matmul(attn.transpose(-2, -1), grad_output)

        # Compute the gradient w.r.t. attn (intermediate softmax gradient)
        grad_attn = torch.matmul(grad_output, v.transpose(-2, -1))

        # Gradient of softmax
        d_attn_logits = attn * (
            grad_attn - torch.sum(grad_attn * attn, dim=-1, keepdim=True)
        )

        # Compute the gradient w.r.t. q and k
        grad_q = torch.matmul(d_attn_logits, k)
        grad_k = torch.matmul(d_attn_logits.transpose(-2, -1), q)
        # pdb.set_trace()
        print("Attention grad sum:", grad_q.sum(), grad_k.sum(), grad_v.sum())
        return grad_q, grad_k, grad_v


def attention(q, k, v, mask=None, dropout=None):
    # q,k,v : (b, h, s, d)
    # set seed
    torch.manual_seed(0)    
    
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / torch.sqrt(
        torch.tensor(q.size(-1), dtype=torch.float32)
    )
    attn = torch.nn.functional.softmax(
        attn_logits - torch.max(attn_logits, dim=-1)[0].unsqueeze(-1), dim=-1
    )
    return torch.matmul(attn, v), attn, attn_logits


class Attention(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head == d_head

        self.qkc_proj = torch.nn.Linear(d_model, 3 * d_model)
        self.o_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, use_flash=False):
        b, s, _ = x.size()
        qkv = self.qkc_proj(x).reshape(b, s, self.n_heads, 3 * self.d_head)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = torch.split(qkv, self.d_head, dim=-1)  # (b, h, s, d)
        if use_flash:
            out_attn = FlashAttn.apply(q, k, v)
        else:
            out_attn = Attn.apply(q, k, v)
        out = out_attn.permute(0, 2, 1, 3).reshape(b, s, self.d_model)
        o = self.o_proj(out)
        return o, out_attn


def main():
    x = torch.randn(b, s, d_model).to(device)
    attention = Attention(d_model, n_heads).to(device)
    o_fa, out_attn_fa = attention(x, use_flash=True)
    o_fa.sum().backward()
    attention.zero_grad()

    o_attn, out_attn = attention(x, use_flash=False)
    o_attn.sum().backward()

    print("Forward pass activations allighn:", torch.allclose(out_attn, out_attn_fa, atol=1e-2))
    print("Forward outputs align:", torch.allclose(o_fa.sum(), o_attn.sum(), atol=1e-1))


if __name__ == "__main__":
    main()
