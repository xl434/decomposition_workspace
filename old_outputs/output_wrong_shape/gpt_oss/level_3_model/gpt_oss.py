"""
GPT-OSS Complete Transformer - Level 3 Model Component

Component: Transformer (gpt_oss)
Level: 3 (Model)
Parent: root
Children: Embedding (L0), TransformerBlock x NUM_HIDDEN_LAYERS (L2), FinalRMSNorm (L0), Unembedding (L0)

Operations:
    1. Token embedding lookup
    2. N transformer blocks (each: AttentionBlock + MLPBlock)
    3. Final RMSNorm
    4. Linear unembedding to vocabulary logits

Input Shape: x [SEQ_LEN=16] int32 (token indices)
Output Shape: [SEQ_LEN=16, VOCAB_SIZE=256] bfloat16 (logits)

Test Configuration:
    SEQ_LEN=16, HIDDEN_SIZE=128, NUM_ATTENTION_HEADS=8, NUM_KEY_VALUE_HEADS=2,
    HEAD_DIM=32, QKV_DIM=384, SLIDING_WINDOW=8, SM_SCALE=1/sqrt(32),
    ROPE_THETA=150000.0, ROPE_SCALING_FACTOR=32.0, ROPE_NTK_ALPHA=1.0,
    ROPE_NTK_BETA=32.0, INITIAL_CONTEXT_LENGTH=256,
    NUM_EXPERTS=4, EXPERTS_PER_TOKEN=2, INTERMEDIATE_SIZE=128, SWIGLU_LIMIT=7.0,
    VOCAB_SIZE=256, NUM_HIDDEN_LAYERS=2 (test; original=36)
"""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# AttentionBlock
# ============================================================
class AttentionBlock(nn.Module):
    """Full attention block: RMSNorm -> QKV -> RoPE -> SDPA -> OutProj + Residual."""

    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads,
                 head_dim, sliding_window, rope_theta, rope_scaling_factor,
                 rope_ntk_alpha, rope_ntk_beta, initial_context_length):
        super().__init__()
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.sliding_window = sliding_window
        self.sm_scale = 1 / math.sqrt(head_dim)

        # RMSNorm
        self.norm_scale = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = 1e-5

        # Attention sinks
        self.sinks = nn.Parameter(torch.empty(num_attention_heads, dtype=torch.bfloat16))
        nn.init.normal_(self.sinks)

        # QKV projection
        qkv_dim = head_dim * (num_attention_heads + 2 * num_key_value_heads)
        self.qkv = nn.Linear(hidden_size, qkv_dim, dtype=torch.bfloat16)

        # Output projection
        self.out = nn.Linear(head_dim * num_attention_heads, hidden_size, dtype=torch.bfloat16)

        # RoPE config
        self.rope_head_dim = head_dim
        self.rope_base = rope_theta
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_ntk_alpha = rope_ntk_alpha
        self.rope_ntk_beta = rope_ntk_beta
        self.initial_context_length = initial_context_length

    def _compute_rope(self, num_tokens):
        freq = self.rope_base ** (
            torch.arange(0, self.rope_head_dim, 2, dtype=torch.float) / self.rope_head_dim
        )
        if self.rope_scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.rope_scaling_factor) + 1.0
            d_half = self.rope_head_dim / 2
            low = (
                d_half
                * math.log(self.initial_context_length / (self.rope_ntk_beta * 2 * math.pi))
                / math.log(self.rope_base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.rope_ntk_alpha * 2 * math.pi))
                / math.log(self.rope_base)
            )
            interpolation = 1.0 / (self.rope_scaling_factor * freq)
            extrapolation = 1.0 / freq
            ramp = (torch.arange(d_half, dtype=torch.float32) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq
        t = torch.arange(num_tokens, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        return freqs.cos() * concentration, freqs.sin() * concentration

    def _apply_rotary(self, x, cos, sin):
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)

    def forward(self, x):
        # RMSNorm
        t = x.float()
        t = t * torch.rsqrt(torch.mean(t ** 2, dim=-1, keepdim=True) + self.eps)
        t = (t * self.norm_scale).to(x.dtype)

        # QKV projection
        qkv = self.qkv(t)
        q = qkv[:, :self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            self.num_attention_heads * self.head_dim:
            (self.num_attention_heads + self.num_key_value_heads) * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads) * self.head_dim:,
        ].contiguous()

        # Reshape for GQA
        q_mult = self.num_attention_heads // self.num_key_value_heads
        q = q.view(-1, self.num_key_value_heads, q_mult, self.head_dim)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)

        # RoPE
        num_tokens = q.shape[0]
        cos, sin = self._compute_rope(num_tokens)

        q_shape = q.shape
        q = q.view(num_tokens, -1, self.head_dim)
        q = self._apply_rotary(q, cos, sin)
        q = q.reshape(q_shape)

        k_shape = k.shape
        k = k.view(num_tokens, -1, self.head_dim)
        k = self._apply_rotary(k, cos, sin)
        k = k.reshape(k_shape)

        # SDPA
        n_tokens, n_heads, q_mult_dim, d_head = q.shape
        K = k[:, :, None, :].expand(-1, -1, q_mult_dim, -1)
        V = v[:, :, None, :].expand(-1, -1, q_mult_dim, -1)
        S = self.sinks.reshape(n_heads, q_mult_dim, 1, 1).expand(-1, -1, n_tokens, -1)

        mask_attn = torch.triu(
            q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1
        )
        if self.sliding_window > 0:
            mask_attn += torch.tril(
                mask_attn.new_full((n_tokens, n_tokens), -float("inf")),
                diagonal=-self.sliding_window,
            )

        QK = torch.einsum("qhmd,khmd->hmqk", q, K)
        QK *= self.sm_scale
        QK += mask_attn[None, None, :, :]
        QK = torch.cat([QK, S], dim=-1)
        W = torch.softmax(QK, dim=-1)
        W = W[..., :-1]
        attn = torch.einsum("hmqk,khmd->qhmd", W, V)
        t = attn.reshape(n_tokens, -1)

        # Output projection + residual
        t = self.out(t)
        return x + t


# ============================================================
# MLPBlock
# ============================================================
class MLPBlock(nn.Module):
    """MoE MLP block: RMSNorm -> Gate -> Expert Up + SwiGLU -> Expert Down + Combine + Residual."""

    def __init__(self, hidden_size, intermediate_size, num_experts, experts_per_token, swiglu_limit):
        super().__init__()
        self.experts_per_token = experts_per_token
        self.swiglu_limit = swiglu_limit

        # RMSNorm
        self.norm_scale = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = 1e-5

        # Gate
        self.gate = nn.Linear(hidden_size, num_experts, dtype=torch.bfloat16)

        # MLP1 expert weights
        self.mlp1_weight = nn.Parameter(
            torch.empty(num_experts, intermediate_size * 2, hidden_size, dtype=torch.bfloat16)
        )
        self.mlp1_bias = nn.Parameter(
            torch.empty(num_experts, intermediate_size * 2, dtype=torch.bfloat16)
        )

        # MLP2 expert weights
        self.mlp2_weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16)
        )
        self.mlp2_bias = nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.bfloat16)
        )

        # Init
        nn.init.normal_(self.mlp1_weight, std=0.02)
        nn.init.zeros_(self.mlp1_bias)
        nn.init.normal_(self.mlp2_weight, std=0.02)
        nn.init.zeros_(self.mlp2_bias)

    def forward(self, x):
        # RMSNorm
        t = x.float()
        t = t * torch.rsqrt(torch.mean(t ** 2, dim=-1, keepdim=True) + self.eps)
        t = (t * self.norm_scale).to(x.dtype)

        # Gating
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = F.softmax(experts.values, dim=1)
        expert_indices = experts.indices

        # MLP1 (up-projection)
        mlp1_w = self.mlp1_weight[expert_indices, ...]
        mlp1_b = self.mlp1_bias[expert_indices, ...]
        h = torch.einsum("beck,bk->bec", mlp1_w, t) + mlp1_b

        # SwiGLU activation
        x_glu = h[..., ::2]
        x_linear = h[..., 1::2]
        x_glu = x_glu.clamp(min=None, max=self.swiglu_limit)
        x_linear = x_linear.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        out_glu = x_glu * torch.sigmoid(1.702 * x_glu)
        h = out_glu * (x_linear + 1)

        # MLP2 (down-projection)
        mlp2_w = self.mlp2_weight[expert_indices, ...]
        mlp2_b = self.mlp2_bias[expert_indices, ...]
        h = torch.einsum("beck,bek->bec", mlp2_w, h)
        h += mlp2_b

        # Weighted combination + residual
        h = torch.einsum("bec,be->bc", h, expert_weights)
        return x + h


# ============================================================
# TransformerBlock
# ============================================================
class TransformerBlock(nn.Module):
    """Single transformer block: AttentionBlock + MLPBlock."""

    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads,
                 head_dim, sliding_window, rope_theta, rope_scaling_factor,
                 rope_ntk_alpha, rope_ntk_beta, initial_context_length,
                 intermediate_size, num_experts, experts_per_token, swiglu_limit,
                 layer_idx):
        super().__init__()
        effective_sliding_window = sliding_window if layer_idx % 2 == 0 else 0

        self.attn = AttentionBlock(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            sliding_window=effective_sliding_window,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            rope_ntk_alpha=rope_ntk_alpha,
            rope_ntk_beta=rope_ntk_beta,
            initial_context_length=initial_context_length,
        )

        self.mlp = MLPBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            swiglu_limit=swiglu_limit,
        )

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


# ============================================================
# RMSNorm (for final norm)
# ============================================================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features, dtype=torch.float32))

    def forward(self, x):
        t = x.float()
        t = t * torch.rsqrt(torch.mean(t ** 2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(x.dtype)


# ============================================================
# Complete Transformer Model (Level 3)
# ============================================================
class Model(nn.Module):
    """
    Complete GPT-OSS MoE Transformer model.
    Embedding -> N x TransformerBlock -> FinalRMSNorm -> Unembedding.
    """

    def __init__(self, num_hidden_layers=2, num_experts=4, experts_per_token=2,
                 vocab_size=256, hidden_size=128, intermediate_size=128,
                 swiglu_limit=7.0, head_dim=32, num_attention_heads=8,
                 num_key_value_heads=2, sliding_window=8,
                 initial_context_length=256, rope_theta=150000.0,
                 rope_scaling_factor=32.0, rope_ntk_alpha=1.0, rope_ntk_beta=32.0):
        super().__init__()

        # Level 0: Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_size, dtype=torch.bfloat16)

        # Level 2: Transformer blocks
        self.block = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                sliding_window=sliding_window,
                rope_theta=rope_theta,
                rope_scaling_factor=rope_scaling_factor,
                rope_ntk_alpha=rope_ntk_alpha,
                rope_ntk_beta=rope_ntk_beta,
                initial_context_length=initial_context_length,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                experts_per_token=experts_per_token,
                swiglu_limit=swiglu_limit,
                layer_idx=i,
            )
            for i in range(num_hidden_layers)
        ])

        # Level 0: Final RMSNorm
        self.norm = RMSNorm(hidden_size)

        # Level 0: Unembedding (no bias)
        self.unembedding = nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        """
        Forward pass for the complete transformer.

        Args:
            x: Token indices [seq_len] int32

        Returns:
            Logits [seq_len, vocab_size] bfloat16
        """
        # Level 0: Embedding
        x = self.embedding(x)

        # Level 2: Transformer blocks
        for block in self.block:
            x = block(x)

        # Level 0: Final RMSNorm
        x = self.norm(x)

        # Level 0: Unembedding
        x = self.unembedding(x)

        return x


def get_inputs():
    """Return sample inputs for testing."""
    return [torch.randint(0, 256, (16,), dtype=torch.int32)]


def get_init_inputs():
    """Return constructor arguments for the Model (all defaults)."""
    return []


def get_expected_output_shape():
    """Return the expected output shape."""
    return [(16, 256)]


def run_tests():
    """Run validation tests for the complete Transformer model."""
    print("=" * 60)
    print("Testing: GPT-OSS Transformer (Level 3 Model)")
    print("=" * 60)

    passed = True

    # Test 1: Model instantiation
    try:
        model = Model(*get_init_inputs())
        model.eval()
        print("[PASS] Model instantiation")
    except Exception as e:
        print(f"[FAIL] Model instantiation: {e}")
        return False

    # Test 2: Forward pass shape
    try:
        inputs = get_inputs()
        with torch.no_grad():
            output = model(*inputs)
        expected_shapes = get_expected_output_shape()
        assert output.shape == torch.Size(expected_shapes[0]), (
            f"Expected shape {expected_shapes[0]}, got {output.shape}"
        )
        print(f"[PASS] Output shape: {output.shape}")
    except Exception as e:
        print(f"[FAIL] Forward pass shape: {e}")
        passed = False

    # Test 3: Output dtype
    try:
        assert output.dtype == torch.bfloat16, (
            f"Expected bfloat16, got {output.dtype}"
        )
        print(f"[PASS] Output dtype: {output.dtype}")
    except Exception as e:
        print(f"[FAIL] Output dtype: {e}")
        passed = False

    # Test 4: Output contains finite values
    try:
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        print("[PASS] Output values are finite")
    except Exception as e:
        print(f"[FAIL] Finite check: {e}")
        passed = False

    # Test 5: Deterministic output
    try:
        with torch.no_grad():
            output2 = model(*inputs)
        assert torch.allclose(output, output2, atol=1e-6), "Non-deterministic output"
        print("[PASS] Deterministic output")
    except Exception as e:
        print(f"[FAIL] Deterministic: {e}")
        passed = False

    # Test 6: Verify model structure
    try:
        assert len(model.block) == 2, f"Expected 2 blocks, got {len(model.block)}"
        print(f"[PASS] Model has {len(model.block)} transformer blocks")

        # Block 0 should have sliding_window = 8 (even layer)
        assert model.block[0].attn.sliding_window == 8, (
            f"Block 0 sliding_window expected 8, got {model.block[0].attn.sliding_window}"
        )
        print("[PASS] Block 0 has sliding_window=8")

        # Block 1 should have sliding_window = 0 (odd layer)
        assert model.block[1].attn.sliding_window == 0, (
            f"Block 1 sliding_window expected 0, got {model.block[1].attn.sliding_window}"
        )
        print("[PASS] Block 1 has sliding_window=0")
    except Exception as e:
        print(f"[FAIL] Structure check: {e}")
        passed = False

    # Test 7: Embedding dimensions
    try:
        assert model.embedding.num_embeddings == 256, (
            f"Expected vocab_size=256, got {model.embedding.num_embeddings}"
        )
        assert model.embedding.embedding_dim == 128, (
            f"Expected hidden_size=128, got {model.embedding.embedding_dim}"
        )
        print("[PASS] Embedding dimensions correct (256, 128)")
    except Exception as e:
        print(f"[FAIL] Embedding check: {e}")
        passed = False

    # Test 8: Unembedding dimensions
    try:
        assert model.unembedding.in_features == 128, (
            f"Expected in_features=128, got {model.unembedding.in_features}"
        )
        assert model.unembedding.out_features == 256, (
            f"Expected out_features=256, got {model.unembedding.out_features}"
        )
        print("[PASS] Unembedding dimensions correct (128 -> 256)")
    except Exception as e:
        print(f"[FAIL] Unembedding check: {e}")
        passed = False

    # Test 9: Parameter count check
    try:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[INFO] Total parameters: {total_params:,}")
        assert total_params > 0, "Model has no parameters"
        print("[PASS] Parameter count > 0")
    except Exception as e:
        print(f"[FAIL] Parameter count: {e}")
        passed = False

    print("=" * 60)
    print(f"GPT-OSS Transformer: {'ALL TESTS PASSED' if passed else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return passed


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
