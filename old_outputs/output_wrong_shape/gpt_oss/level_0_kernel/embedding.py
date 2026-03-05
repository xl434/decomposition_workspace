"""
Component: Token Embedding
Abstraction Level: kernel (L0)
Parent: gpt_oss (L3)
Children: None (leaf)

Operations: nn.Embedding lookup

Input Shapes:
  - x: [16] dtype=int32 (token indices)

Output Shapes:
  - output: [16, 128] dtype=bfloat16

Weight Shapes:
  - embedding.weight: [256, 128] dtype=bfloat16
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Token embedding lookup. Extracted from: Transformer"""

    def __init__(self, vocab_size=256, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, dtype=torch.bfloat16)

    def forward(self, x):
        return self.embedding(x)


def get_inputs():
    return [torch.randint(0, 256, (16,), dtype=torch.int32)]


def get_init_inputs():
    return [256, 128]


def get_expected_output_shape():
    return [(16, 128)]


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
