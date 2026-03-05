"""
Component: Split QKV
Abstraction Level: kernel (L0)
Parent: gpt_oss (L3)
Children: None (leaf)

Operations: Tensor slicing to split concatenated QKV into separate Q, K, V tensors

Input Shapes:
  - qkv: [16, 384] dtype=bfloat16 (concatenated Q, K, V)

Output Shapes:
  - Q: [16, 256] dtype=bfloat16 (NUM_ATTENTION_HEADS * HEAD_DIM = 8 * 32)
  - K: [16, 64] dtype=bfloat16  (NUM_KEY_VALUE_HEADS * HEAD_DIM = 2 * 32)
  - V: [16, 64] dtype=bfloat16  (NUM_KEY_VALUE_HEADS * HEAD_DIM = 2 * 32)

Config:
  - NUM_ATTENTION_HEADS = 8
  - NUM_KEY_VALUE_HEADS = 2
  - HEAD_DIM = 32
  - Q_DIM = 256, K_DIM = 64, V_DIM = 64
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Split concatenated QKV tensor into separate Q, K, V. Extracted from: Transformer"""

    def __init__(self, num_attention_heads=8, num_key_value_heads=2, head_dim=32):
        super().__init__()
        self.q_dim = num_attention_heads * head_dim      # 256
        self.k_dim = num_key_value_heads * head_dim       # 64
        self.v_dim = num_key_value_heads * head_dim       # 64

    def forward(self, qkv):
        q = qkv[:, :self.q_dim]
        k = qkv[:, self.q_dim:self.q_dim + self.k_dim]
        v = qkv[:, self.q_dim + self.k_dim:]
        return q, k, v


def get_inputs():
    return [torch.randn(16, 384, dtype=torch.bfloat16)]


def get_init_inputs():
    return [8, 2, 32]


def get_expected_output_shape():
    return [(16, 256), (16, 64), (16, 64)]


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
