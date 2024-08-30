from pathlib import Path
import torch
import math
import torch
from torch.utils.cpp_extension import load_inline
from torch.autograd import Function

assert torch.cuda.is_available()
device = torch.device("cuda")

s = 1024  # seq length
n_heads = 8
d_model = 128  # hidden dim
d_head = d_model // n_heads  # this is in numbers, each number is float32, so 4 bytes
d_head_bytes = d_head * 4
assert d_model % n_heads == 0
b = 32  # batch size




def compile_fa_forward():
    cuda_source = Path("kernel/fa.cu").read_text()
    cpp_source = "std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_attention_forwad(torch::Tensor q, torch::Tensor k, torch::Tensor v, float tau);"

    # Load the CUDA kernel as a PyTorch extension
    ext = load_inline(
        name="fa_forward",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["flash_attention_forwad"],
        with_cuda=True,
        extra_cuda_cflags=["-G","-g"], # "-g" is for debugging, "-G" is for device-side debugging
        extra_cflags=['-O0 -g'],  # Add '-g' for debug symbols in C++
        build_directory='cuda_build/',
    )
    return ext

fa_forward_kernel = compile_fa_forward()

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


class FlashAttn(Function):
    @staticmethod
    def forward(ctx, q, k, v):
        # q, k, v : (b, h, s, d)
        tau = 1 / math.sqrt(q.size(-1))
        # import pdb; pdb.set_trace()
        out, l, m = fa_forward_kernel.flash_attention_forwad(q, k, v, tau)

        # we will need to save l and m, but they ae only s long, so we can store them in ctx
        # ctx.save_for_backward(l, m, q.detach(), k.detach(), v.detach(), out)
        # import pdb; pdb.set_trace()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented yet")


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



class Attention(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

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
    # o_fa.sum().backward()
    # attention.zero_grad()

    o_attn, out_attn = attention(x, use_flash=False)
    # o_attn.sum().backward()
    import pdb; pdb.set_trace()
    print("Forward pass activations allighn:", torch.allclose(out_attn, out_attn_fa, atol=1e-2))
    print("Forward outputs align:", torch.allclose(o_fa.sum(), o_attn.sum(), atol=1e-1))


if __name__ == "__main__":
    main()