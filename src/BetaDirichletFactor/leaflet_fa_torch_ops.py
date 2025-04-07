import torch
import triton
import triton.language as tl
from torch.autograd import Function

def next_power_of_2(x):
    return 1 << (x - 1).bit_length()

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

# 4. CPU Fall Back 
def masked_matmul_sparse_cpu(A, B, mask_sparse):
    """
    Compute out[i, j] = A[i] @ B[:, j] only for (i, j) pairs in mask_sparse.indices().
    Returns a sparse COO tensor.
    """
        
    indices = mask_sparse._indices()
    i_idx = indices[0]
    j_idx = indices[1]

    A_rows = A[i_idx]  # shape: (nnz, K)
    B_cols = B[:, j_idx].T  # shape: (nnz, K)

    values = (A_rows * B_cols).sum(dim=1)  # shape: (nnz,)
    return values

def sparse_dot_cpu(assign, psi, cell_idx, junc_idx):
    """
    Compute output[k] = assign[cell_idx[k]] @ psi[:, junc_idx[k]]
    Returns a 1D tensor of shape (nnz,)
    """
    assign_rows = assign[cell_idx]             # shape: (nnz, K)
    psi_cols = psi[:, junc_idx].T              # shape: (nnz, K)
    return (assign_rows * psi_cols).sum(dim=1) # shape: (nnz,)

@triton.jit
def sparse_dot_kernel(
    assign_ptr, psi_ptr, cell_idx_ptr, junc_idx_ptr, out_ptr,
    C, J, K, nnz,
    stride_assign_c, stride_assign_k,
    stride_psi_k, stride_psi_j,
    BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= nnz:
        return

    i = tl.load(cell_idx_ptr + pid)
    j = tl.load(junc_idx_ptr + pid)

    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K

    a_ptr = assign_ptr + i * stride_assign_c + offsets
    b_ptr = psi_ptr + offsets * stride_psi_k + j * stride_psi_j

    a = tl.load(a_ptr, mask=mask, other=0.0)
    b = tl.load(b_ptr, mask=mask, other=0.0)

    acc = tl.sum(a * b)
    tl.store(out_ptr + pid, acc)

class SparseDotFunction(Function):
    @staticmethod
    def forward(ctx, assign, psi, cell_idx, junc_idx):
        assign = assign.contiguous()
        psi = psi.contiguous()

        nnz = cell_idx.shape[0]
        K = assign.shape[1]
        BLOCK_K = next_power_of_2(K)

        out = torch.empty(nnz, device=assign.device, dtype=assign.dtype)

        # Save for backward pass
        ctx.save_for_backward(assign, psi, cell_idx, junc_idx)
        ctx.K = K  # original K for masking

        grid = (triton.cdiv(nnz, 1),)
        sparse_dot_kernel[grid](
            assign, psi, cell_idx, junc_idx, out,
            assign.shape[0], psi.shape[1], K, nnz,
            assign.stride(0), assign.stride(1),
            psi.stride(0), psi.stride(1),
            BLOCK_K=BLOCK_K
        )

        return out

    @staticmethod
    def backward(ctx, grad_out):
        assign, psi, cell_idx, junc_idx = ctx.saved_tensors
        grad_assign = grad_psi = None

        K = ctx.K

        if ctx.needs_input_grad[0]:
            grad_assign = torch.zeros_like(assign)
            psi_cols = psi[:, junc_idx]  # (K, nnz)
            grad_vals = grad_out.view(-1, 1) * psi_cols.T  # (nnz, K)
            grad_assign.index_add_(0, cell_idx, grad_vals)

        if ctx.needs_input_grad[1]:
            grad_psi = torch.zeros_like(psi)
            assign_rows = assign[cell_idx]  # (nnz, K)
            grad_vals = grad_out.view(-1, 1) * assign_rows  # (nnz, K)
            grad_psi.index_add_(1, junc_idx, grad_vals.T)

        return grad_assign, grad_psi, None, None

def masked_sparse_matmul_triton(assign, psi, cell_idx, junc_idx):
    return SparseDotFunction.apply(assign, psi, cell_idx, junc_idx)
