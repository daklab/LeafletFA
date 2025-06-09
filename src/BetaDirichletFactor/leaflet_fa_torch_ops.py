import torch
import triton
import triton.language as tl
from torch.autograd import Function

# 1. Triton kernel 
@triton.jit
def masked_matmul_kernel(
    A_ptr, B_ptr, M_ptr, Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_mn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    load_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    acc = acc * load_mask

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=load_mask)


# 2. Autograd wrapper
class MaskedMatmulFunction(Function):
    @staticmethod
    def forward(ctx, A, B, M):
        ctx.save_for_backward(A, B, M)
        Out = torch.zeros_like(M, dtype=A.dtype)

        M_, K = A.shape
        K_, N = B.shape
        assert K == K_, "Incompatible dimensions"
        assert M.shape == (M_, N), "Mask shape mismatch"

        grid = (triton.cdiv(M_, 32), triton.cdiv(N, 32))
        masked_matmul_kernel[grid](
            A, B, M, Out,
            M_, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            M.stride(0),
            Out.stride(0), Out.stride(1),
            BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
        )
        return Out

    @staticmethod
    def backward(ctx, grad_out):
        A, B, M = ctx.saved_tensors
        grad_A = grad_B = None

        if ctx.needs_input_grad[0]:
            grad_A = grad_out @ B.T  # (N, K)

        if ctx.needs_input_grad[1]:
            grad_B = A.T @ grad_out  # (K, P)

        return grad_A, grad_B, None  # No gradient for mask

# 3. Function to call in LeafletFA model
def masked_matmul(A, B, M):
    return MaskedMatmulFunction.apply(A, B, M)
