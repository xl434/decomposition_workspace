"""
GPT-OSS Attention Block - Level 2 Layer Component

Component: AttentionBlock
Level: 2 (Layer)
Parent: TransformerBlock (Level 2)
Children: AttnNormQKV (L1), RoPEApplication (L1), SDPACore (L1), AttnOutputResidual (L1)

Operations:
    1. RMSNorm on input
    2. QKV linear projection
    3. Split into Q, K, V and reshape for GQA
    4. RoPE (Rotary Position Embedding) on Q and K
    5. Scaled Dot-Product Attention with causal+sliding window mask and attention sinks
    6. Output linear projection + residual connection

Input Shape: x [SEQ_LEN=16, HIDDEN_SIZE=128] bfloat16
Output Shape: [SEQ_LEN=16, HIDDEN_SIZE=128] bfloat16

Test Configuration:
    SEQ_LEN=16, HIDDEN_SIZE=128, NUM_ATTENTION_HEADS=8, NUM_KEY_VALUE_HEADS=2,
    HEAD_DIM=32, QKV_DIM=384, SLIDING_WINDOW=8, SM_SCALE=1/sqrt(32),
    ROPE_THETA=150000.0, ROPE_SCALING_FACTOR=32.0, ROPE_NTK_ALPHA=1.0,
    ROPE_NTK_BETA=32.0, INITIAL_CONTEXT_LENGTH=256
"""

import math
import sys
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Full attention block: RMSNorm -> QKV projection -> RoPE -> SDPA -> Output projection + Residual.
    Combines all attention-related operations into a single Level 2 layer.
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

        # RMSNorm parameters
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
        """Compute RoPE cos/sin frequencies with NTK-aware scaling."""
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
        """Apply rotary position embedding to tensor x."""
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)

    def forward(self, x):
        """
        Forward pass for the attention block.

        Args:
            x: Input tensor [seq_len, hidden_size] bfloat16

        Returns:
            Output tensor [seq_len, hidden_size] bfloat16 (with residual connection)
        """
        # --- L1 Fusion: AttnNormQKV ---
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

        # --- L1 Fusion: RoPE Application ---
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

        # --- L1 Fusion: SDPA Core ---
        n_tokens, n_heads, q_mult_dim, d_head = q.shape
        K = k[:, :, None, :].expand(-1, -1, q_mult_dim, -1)
        V = v[:, :, None, :].expand(-1, -1, q_mult_dim, -1)
        S = self.sinks.reshape(n_heads, q_mult_dim, 1, 1).expand(-1, -1, n_tokens, -1)

        # Causal + sliding window mask
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

        # --- L1 Fusion: Output projection + residual ---
        t = self.out(t)
        return x + t


def get_inputs():
    """Return sample inputs for testing."""
    return [torch.randn(16, 128, dtype=torch.bfloat16)]


def get_init_inputs():
    """Return constructor arguments for the Model."""
    return [128, 8, 2, 32, 8, 150000.0, 32.0, 1.0, 32.0, 256]


def get_expected_output_shape():
    """Return the expected output shape."""
    return [(16, 128)]


def run_tests():
    """Run validation tests for the AttentionBlock component."""
    print("=" * 60)
    print("Testing: AttentionBlock (Level 2 Layer)")
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

    # Test 5: Residual connection check (output should differ from input)
    try:
        input_tensor = inputs[0]
        diff = (output - input_tensor).abs().max().item()
        assert diff > 0, "Output is identical to input - residual connection may be wrong"
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

    print("=" * 60)
    print(f"AttentionBlock: {'ALL TESTS PASSED' if passed else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return passed


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
