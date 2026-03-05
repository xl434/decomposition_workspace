"""
Component: GQA Reshape
Abstraction Level: kernel (L0)
Parent: gpt_oss (L3)
Children: None (leaf)

Operations: Reshape Q, K, V tensors for Grouped Query Attention

Input Shapes:
  - Q: [16, 256] dtype=bfloat16
  - K: [16, 64] dtype=bfloat16
  - V: [16, 64] dtype=bfloat16

Output Shapes:
  - Q_reshaped: [16, 2, 4, 32] dtype=bfloat16 (seq, kv_heads, q_mult, head_dim)
  - K_reshaped: [16, 2, 32] dtype=bfloat16    (seq, kv_heads, head_dim)
  - V_reshaped: [16, 2, 32] dtype=bfloat16    (seq, kv_heads, head_dim)

Config:
  - NUM_ATTENTION_HEADS = 8
  - NUM_KEY_VALUE_HEADS = 2
  - HEAD_DIM = 32
  - Q_MULT = 4 (= 8 / 2)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Reshape Q, K, V for Grouped Query Attention. Extracted from: Transformer"""

    def __init__(self, num_attention_heads=8, num_key_value_heads=2, head_dim=32):
        super().__init__()
        self.num_key_value_heads = num_key_value_heads    # 2
        self.q_mult = num_attention_heads // num_key_value_heads  # 4
        self.head_dim = head_dim                          # 32

    def forward(self, q, k, v):
        q_reshaped = q.view(-1, self.num_key_value_heads, self.q_mult, self.head_dim)
        k_reshaped = k.view(-1, self.num_key_value_heads, self.head_dim)
        v_reshaped = v.view(-1, self.num_key_value_heads, self.head_dim)
        return q_reshaped, k_reshaped, v_reshaped


def get_inputs():
    return [
        torch.randn(16, 256, dtype=torch.bfloat16),
        torch.randn(16, 64, dtype=torch.bfloat16),
        torch.randn(16, 64, dtype=torch.bfloat16),
    ]


def get_init_inputs():
    return [8, 2, 32]


def get_expected_output_shape():
    return [(16, 2, 4, 32), (16, 2, 32), (16, 2, 32)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            assert isinstance(output, tuple), f"Expected tuple output, got {type(output)}"
            assert len(output) == 3, f"Expected 3 outputs, got {len(output)}"
            expected_shapes = get_expected_output_shape()
            actual_shapes = [o.shape for o in output]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), f"Output {i} shape mismatch: got {actual}, expected {expected}"
            for i, o in enumerate(output):
                assert not torch.isnan(o).any(), f"Output {i} contains NaN"
                assert not torch.isinf(o).any(), f"Output {i} contains Inf"
                assert o.dtype == torch.bfloat16, f"Output {i} dtype mismatch: {o.dtype} vs bfloat16"
            print(f"Input shape(s): {[x.shape for x in inputs]}")
            print(f"Output shape(s): {actual_shapes}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
