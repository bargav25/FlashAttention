import torch
import triton
import triton.language as tl

class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):

        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)

        stage = 3 if causal else 1

        grid = lambda args: (triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), BATCH_SIZE * NUM_HEADS, 1)

        # M is the logsumexp for backward pass, one for each query
        M = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32)

        _attn_fwd[grid](
            Q = Q,
            K = K,
            V = V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch = Q.stride(0),
            stride_Q_head = Q.stride(1),
            stride_Q_seq = Q.stride(2),
            stride_Q_dim = Q.stride(3),
            stride_K_batch = K.stride(0),
            stride_K_head = K.stride(1),
            stride_K_seq = K.stride(2),
            stride_K_dim = K.stride(3),           
            stride_V_batch = V.stride(0),
            stride_V_head = V.stride(1),
            stride_V_seq = V.stride(2),
            stride_V_dim = V.stride(3),   
            stride_O_batch = O.stride(0),
            stride_O_head = O.stride(1),
            stride_O_seq = O.stride(2),
            stride_O_dim = O.stride(3),
            BATCH_SIZE = BATCH_SIZE,
            NUM_HEADS = NUM_HEADS,
            SEQ_LEN = SEQ_LEN,
            HEAD_DIM = HEAD_DIM,
            STAGE = stage         
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid 
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM
        ctx.causal = causal

        return 0



        


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal = True, dtype = torch.float16):

    Q = (
        torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std = 0.5)
        .requires_grad_())

    K = (
        torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std = 0.5)
        .requires_grad_())

    V = (
        torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std = 0.5)
        .requires_grad_())  

    softmax_scale = 1. / (HEAD_DIM ** 2)

    dO = torch.randn_like(Q) # Needs for backprop

    # Naive Implementation of Attention
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), dtype=torch.bool, device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale

    if causal:
        P[:, :, MASK == 0] = float("-inf")

    P = torch.softmax(P.float(), dim=-1).half()

    naive_out = torch.matmul(P, V)
    naive_out.backward(dO)

    naive_dV, V.grad = V.grad.clone(), None
    naive_dK, K.grad = K.grad.clone(), None
    naive_dQ, Q.grad = Q.grad.clone(), None

    # Triton Implementation of Attention
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)

    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None 

    # Comparison 
    rtol = 0.0
    atol = 1e-2

    assert torch.allclose(naive_out, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(naive_dQ, tri_dQ, atol=atol, rtol=rtol)
    assert torch.allclose(naive_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(naive_dV, tri_dV, atol=atol, rtol=rtol)

















