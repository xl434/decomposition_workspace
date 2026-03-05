"""
Component: rope_apply
Level: 1 (Fusion)
Parent: attention_block (Level 2)
Children: rope_frequency (L0), apply_rotary (L0)
Operations: RoPE frequency computation + apply rotary embeddings to Q and K

Input Shapes:
    Q: [SEQ_LEN=16, NUM_KEY_VALUE_HEADS=2, Q_MULT=4, HEAD_DIM=32] bfloat16
    K: [SEQ_LEN=16, NUM_KEY_VALUE_HEADS=2, HEAD_DIM=32] bfloat16

Output Shapes:
    Q_rotated: [SEQ_LEN=16, NUM_KEY_VALUE_HEADS=2, Q_MULT=4, HEAD_DIM=32] bfloat16
    K_rotated: [SEQ_LEN=16, NUM_KEY_VALUE_HEADS=2, HEAD_DIM=32] bfloat16

Test Configuration:
    HEAD_DIM=32, ROPE_THETA=150000.0, ROPE_SCALING_FACTOR=32.0,
    ROPE_NTK_ALPHA=1.0, ROPE_NTK_BETA=32.0, INITIAL_CONTEXT_LENGTH=256
"""

import sys
import math
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, head_dim=32, base=150000.0, scaling_factor=32.0,
                 ntk_alpha=1.0, ntk_beta=32.0, initial_context_length=256):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.initial_context_length = initial_context_length

    def _compute_cos_sin(self, num_tokens):
        freq = self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim)
        concentration = 0.1 * math.log(self.scaling_factor) + 1.0
        d_half = self.head_dim / 2
        low = d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base)
        high = d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base)
        interpolation = 1.0 / (self.scaling_factor * freq)
        extrapolation = 1.0 / freq
        ramp = (torch.arange(d_half, dtype=torch.float32) - low) / (high - low)
        mask = 1 - ramp.clamp(0, 1)
        inv_freq = interpolation * (1 - mask) + extrapolation * mask
        t = torch.arange(num_tokens, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def _apply_rotary(self, x, cos, sin):
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)

    def forward(self, q, k):
        num_tokens = q.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)
        # Apply to Q
        q_shape = q.shape
        q = q.view(num_tokens, -1, self.head_dim)
        q = self._apply_rotary(q, cos, sin)
        q = q.reshape(q_shape)
        # Apply to K
        k_shape = k.shape
        k = k.view(num_tokens, -1, self.head_dim)
        k = self._apply_rotary(k, cos, sin)
        k = k.reshape(k_shape)
        return q, k


def get_inputs():
    return [
        torch.randn(16, 2, 4, 32, dtype=torch.bfloat16),
        torch.randn(16, 2, 32, dtype=torch.bfloat16),
    ]


def get_init_inputs():
    return [32, 150000.0, 32.0, 1.0, 32.0, 256]


def get_expected_output_shape():
    return [(16, 2, 4, 32), (16, 2, 32)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        inputs = get_inputs()
        outputs = model(*inputs)

        # Verify output is a tuple of 2 elements
        assert isinstance(outputs, tuple), f"Expected tuple output, got {type(outputs)}"
        assert len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}"

        expected_shapes = get_expected_output_shape()
        q_rot, k_rot = outputs

        # Check shapes
        assert q_rot.shape == torch.Size(expected_shapes[0]), \
            f"Q_rotated shape mismatch: expected {expected_shapes[0]}, got {tuple(q_rot.shape)}"
        assert k_rot.shape == torch.Size(expected_shapes[1]), \
            f"K_rotated shape mismatch: expected {expected_shapes[1]}, got {tuple(k_rot.shape)}"

        # Check dtypes
        assert q_rot.dtype == torch.bfloat16, f"Q_rotated dtype mismatch: expected bfloat16, got {q_rot.dtype}"
        assert k_rot.dtype == torch.bfloat16, f"K_rotated dtype mismatch: expected bfloat16, got {k_rot.dtype}"

        # Check no NaN/Inf
        for name, tensor in [("Q_rotated", q_rot), ("K_rotated", k_rot)]:
            assert not torch.isnan(tensor).any(), f"{name} contains NaN"
            assert not torch.isinf(tensor).any(), f"{name} contains Inf"

        print("rope_apply: ALL TESTS PASSED")
        return True
    except Exception as e:
        print(f"rope_apply: TEST FAILED - {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
