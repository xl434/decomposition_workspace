"""
Component: Apply Rotary Embeddings
Abstraction Level: kernel (L0)
Parent: gpt_oss (L3)
Children: None (leaf)

Operations: Apply rotary position embeddings (RoPE) to a tensor using cos/sin frequencies

Input Shapes:
  - x: [16, 2, 32] dtype=bfloat16 (seq_len, kv_heads, head_dim) - testing with K shape
  - cos: [16, 16] dtype=float32 (seq_len, head_dim // 2)
  - sin: [16, 16] dtype=float32 (seq_len, head_dim // 2)

Output Shapes:
  - output: [16, 2, 32] dtype=bfloat16

Notes:
  - For Q tensor, input would be [16, 2, 4, 32] and output [16, 2, 4, 32]
  - cos/sin are broadcast with unsqueeze(-2) to match head dimensions
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Apply rotary embeddings to a tensor. Extracted from: Transformer"""

    def __init__(self):
        super().__init__()

    def forward(self, x, cos, sin):
        cos = cos.unsqueeze(-2).to(x.dtype)  # [16, 1, 16]
        sin = sin.unsqueeze(-2).to(x.dtype)  # [16, 1, 16]
        x1, x2 = torch.chunk(x, 2, dim=-1)  # each [16, 2, 16]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)   # [16, 2, 32]


def get_inputs():
    return [
        torch.randn(16, 2, 32, dtype=torch.bfloat16),
        torch.randn(16, 16, dtype=torch.float32),
        torch.randn(16, 16, dtype=torch.float32),
    ]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(16, 2, 32)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"
            expected_shapes = get_expected_output_shape()
            actual_shapes = [output.shape] if not isinstance(output, tuple) else [o.shape for o in output]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), f"Output {i} shape mismatch: got {actual}, expected {expected}"
            assert output.dtype == torch.bfloat16, f"Dtype mismatch: {output.dtype} vs bfloat16"
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
