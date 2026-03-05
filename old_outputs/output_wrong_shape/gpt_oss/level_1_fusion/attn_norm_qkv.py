"""
Component: attn_norm_qkv
Level: 1 (Fusion)
Parent: attention_block (Level 2)
Children: rms_norm (L0), linear_proj (L0), split_qkv (L0), gqa_reshape (L0)
Operations: RMSNorm -> QKV Linear Projection -> Split -> Reshape for GQA

Input Shapes:
    x: [SEQ_LEN=16, HIDDEN_SIZE=128] bfloat16

Output Shapes:
    Q: [SEQ_LEN=16, NUM_KEY_VALUE_HEADS=2, Q_MULT=4, HEAD_DIM=32] bfloat16
    K: [SEQ_LEN=16, NUM_KEY_VALUE_HEADS=2, HEAD_DIM=32] bfloat16
    V: [SEQ_LEN=16, NUM_KEY_VALUE_HEADS=2, HEAD_DIM=32] bfloat16

Test Configuration:
    SEQ_LEN=16, HIDDEN_SIZE=128, NUM_ATTENTION_HEADS=8, NUM_KEY_VALUE_HEADS=2,
    HEAD_DIM=32, Q_MULT=4, QKV_DIM=384
"""

import sys
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, hidden_size=128, num_attention_heads=8, num_key_value_heads=2, head_dim=32):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.q_mult = num_attention_heads // num_key_value_heads
        # RMSNorm
        self.scale = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = 1e-5
        # QKV Linear
        qkv_dim = head_dim * (num_attention_heads + 2 * num_key_value_heads)
        self.qkv = nn.Linear(hidden_size, qkv_dim, dtype=torch.bfloat16)

    def forward(self, x):
        # RMSNorm
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t ** 2, dim=-1, keepdim=True) + self.eps)
        t = (t * self.scale).to(dtype)
        # QKV projection
        qkv = self.qkv(t)
        # Split
        q_dim = self.num_attention_heads * self.head_dim
        k_dim = self.num_key_value_heads * self.head_dim
        q = qkv[:, :q_dim].contiguous()
        k = qkv[:, q_dim:q_dim + k_dim].contiguous()
        v = qkv[:, q_dim + k_dim:].contiguous()
        # Reshape for GQA
        q = q.view(-1, self.num_key_value_heads, self.q_mult, self.head_dim)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        return q, k, v


def get_inputs():
    return [torch.randn(16, 128, dtype=torch.bfloat16)]


def get_init_inputs():
    return [128, 8, 2, 32]


def get_expected_output_shape():
    return [(16, 2, 4, 32), (16, 2, 32), (16, 2, 32)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        inputs = get_inputs()
        outputs = model(*inputs)

        # Verify output is a tuple of 3 elements
        assert isinstance(outputs, tuple), f"Expected tuple output, got {type(outputs)}"
        assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"

        expected_shapes = get_expected_output_shape()
        q, k, v = outputs

        # Check shapes
        assert q.shape == torch.Size(expected_shapes[0]), \
            f"Q shape mismatch: expected {expected_shapes[0]}, got {tuple(q.shape)}"
        assert k.shape == torch.Size(expected_shapes[1]), \
            f"K shape mismatch: expected {expected_shapes[1]}, got {tuple(k.shape)}"
        assert v.shape == torch.Size(expected_shapes[2]), \
            f"V shape mismatch: expected {expected_shapes[2]}, got {tuple(v.shape)}"

        # Check dtypes
        assert q.dtype == torch.bfloat16, f"Q dtype mismatch: expected bfloat16, got {q.dtype}"
        assert k.dtype == torch.bfloat16, f"K dtype mismatch: expected bfloat16, got {k.dtype}"
        assert v.dtype == torch.bfloat16, f"V dtype mismatch: expected bfloat16, got {v.dtype}"

        # Check no NaN/Inf
        for name, tensor in [("Q", q), ("K", k), ("V", v)]:
            assert not torch.isnan(tensor).any(), f"{name} contains NaN"
            assert not torch.isinf(tensor).any(), f"{name} contains Inf"

        print("attn_norm_qkv: ALL TESTS PASSED")
        return True
    except Exception as e:
        print(f"attn_norm_qkv: TEST FAILED - {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
