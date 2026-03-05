"""
Component: linear_mish_mish
Abstraction Level: fusion
Parent: none (top-level)
Children: linear, mish_1, mish_2

Operations: nn.Linear -> F.mish -> F.mish

Input Shapes:
  - x: [2, 16] dtype=float32

Output Shapes:
  - output: [2, 32] dtype=float32

Weight Shapes:
  - linear.weight: [32, 16]
  - linear.bias: [32]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Fused Linear -> Mish -> Mish pipeline.
    Applies a linear transformation followed by two successive Mish activations.

    Extracted from: 29_Matmul_Mish_Mish
    """
    def __init__(self, in_features=16, out_features=32):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = F.mish(x)
        x = F.mish(x)
        return x


def get_inputs():
    return [torch.randn(2, 16)]


def get_init_inputs():
    return [16, 32]


def get_expected_output_shape():
    return [(2, 32)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            expected_shapes = get_expected_output_shape()
            if isinstance(output, tuple):
                actual_shapes = [o.shape for o in output]
            else:
                actual_shapes = [output.shape]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), f"Shape mismatch: {actual} vs {expected}"
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
