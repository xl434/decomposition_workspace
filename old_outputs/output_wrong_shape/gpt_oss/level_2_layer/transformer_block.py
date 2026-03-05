"""
GPT-OSS Transformer Block - Level 2 Layer Component

Component: TransformerBlock
Level: 2 (Layer)
Parent: Transformer / gpt_oss (Level 3)
Children: AttentionBlock (L2), MLPBlock (L2)

Operations:
    1. AttentionBlock: RMSNorm -> QKV -> RoPE -> SDPA -> OutProj + Residual
    2. MLPBlock: RMSNorm -> Gate -> Expert Up + SwiGLU -> Expert Down + Combine + Residual

Input Shape: x [SEQ_LEN=16, HIDDEN_SIZE=128] bfloat16
Output Shape: [SEQ_LEN=16, HIDDEN_SIZE=128] bfloat16

Test Configuration:
    SEQ_LEN=16, HIDDEN_SIZE=128, NUM_ATTENTION_HEADS=8, NUM_KEY_VALUE_HEADS=2,
    HEAD_DIM=32, QKV_DIM=384, SLIDING_WINDOW=8, SM_SCALE=1/sqrt(32),
    ROPE_THETA=150000.0, ROPE_SCALING_FACTOR=32.0, ROPE_NTK_ALPHA=1.0,
    ROPE_NTK_BETA=32.0, INITIAL_CONTEXT_LENGTH=256,
    NUM_EXPERTS=4, EXPERTS_PER_TOKEN=2, INTERMEDIATE_SIZE=128, SWIGLU_LIMIT=7.0
    layer_idx=0 (sliding_window IS applied for even layers)
"""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# AttentionBlock (Level 2 sub-component)
# ============================================================
class AttentionBlock(nn.Module):
    """
    Full attention block: RMSNorm -> QKV -> RoPE -> SDPA -> OutProj + Residual.
    """

    def __init__(self, hidden_size=128, num_attention_heads=8, num_key_value_heads=2,
                 head_dim=32, sliding_window=8, rope_theta=150000.0, rope_scaling_factor=32.0,
                 rope_ntk_alpha=1.0, rope_ntk_beta=32.0, initial_context_length=256):
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
# MLPBlock (Level 2 sub-component)
# ============================================================
class MLPBlock(nn.Module):
    """
    MoE MLP block: RMSNorm -> Gate -> Expert Up + SwiGLU -> Expert Down + Combine + Residual.
    """

    def __init__(self, hidden_size=128, intermediate_size=128, num_experts=4,
                 experts_per_token=2, swiglu_limit=7.0):
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

        # Weighted expert combination + residual
        h = torch.einsum("bec,be->bc", h, expert_weights)
        return x + h


# ============================================================
# TransformerBlock (Level 2 composition)
# ============================================================
class Model(nn.Module):
    """
    Single Transformer block: AttentionBlock followed by MLPBlock.
    Composes two Level 2 sub-components into a full transformer layer.
    """

    def __init__(self, hidden_size=128, num_attention_heads=8, num_key_value_heads=2,
                 head_dim=32, sliding_window=8, rope_theta=150000.0, rope_scaling_factor=32.0,
                 rope_ntk_alpha=1.0, rope_ntk_beta=32.0, initial_context_length=256,
                 intermediate_size=128, num_experts=4, experts_per_token=2, swiglu_limit=7.0,
                 layer_idx=0):
        super().__init__()
        # Sliding window only for even layers (layer_idx % 2 == 0)
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
        """
        Forward pass for the transformer block.

        Args:
            x: Input tensor [seq_len, hidden_size] bfloat16

        Returns:
            Output tensor [seq_len, hidden_size] bfloat16
        """
        x = self.attn(x)
        x = self.mlp(x)
        return x


def get_inputs():
    """Return sample inputs for testing."""
    return [torch.randn(16, 128, dtype=torch.bfloat16)]


def get_init_inputs():
    """Return constructor arguments for the Model (all defaults, layer_idx=0)."""
    return []


def get_expected_output_shape():
    """Return the expected output shape."""
    return [(16, 128)]


def run_tests():
    """Run validation tests for the TransformerBlock component."""
    print("=" * 60)
    print("Testing: TransformerBlock (Level 2 Layer)")
    print("=" * 60)

    passed = True

    # Test 1: Model instantiation with defaults (layer_idx=0)
    try:
        model = Model(*get_init_inputs())
        model.eval()
        print("[PASS] Model instantiation (layer_idx=0, defaults)")
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

    # Test 5: Residual connection (output should differ from input)
    try:
        input_tensor = inputs[0]
        diff = (output - input_tensor).abs().max().item()
        assert diff > 0, "Output is identical to input"
        print(f"[PASS] Residual connection active (max diff: {diff:.4e})")
    except Exception as e:
        print(f"[FAIL] Residual connection: {e}")
        passed = False

    # Test 6: Deterministic output
    try:
        with torch.no_grad():
            output2 = model(*inputs)
        assert torch.allclose(output, output2, atol=1e-6), "Non-deterministic output"
        print("[PASS] Deterministic output")
    except Exception as e:
        print(f"[FAIL] Deterministic: {e}")
        passed = False

    # Test 7: Layer with sliding_window disabled (layer_idx=1)
    try:
        model_odd = Model(layer_idx=1)
        model_odd.eval()
        assert model_odd.attn.sliding_window == 0, (
            f"Expected sliding_window=0 for odd layer, got {model_odd.attn.sliding_window}"
        )
        with torch.no_grad():
            output_odd = model_odd(*inputs)
        assert output_odd.shape == torch.Size(expected_shapes[0]), (
            f"Expected shape {expected_shapes[0]} for odd layer, got {output_odd.shape}"
        )
        print("[PASS] Odd layer (sliding_window=0) works correctly")
    except Exception as e:
        print(f"[FAIL] Odd layer test: {e}")
        passed = False

    # Test 8: Verify attention block has sliding window for layer_idx=0
    try:
        assert model.attn.sliding_window == 8, (
            f"Expected sliding_window=8 for even layer, got {model.attn.sliding_window}"
        )
        print("[PASS] Even layer has sliding_window=8")
    except Exception as e:
        print(f"[FAIL] Sliding window check: {e}")
        passed = False

    print("=" * 60)
    print(f"TransformerBlock: {'ALL TESTS PASSED' if passed else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return passed


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
