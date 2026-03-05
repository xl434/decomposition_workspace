"""
Component: linear_min_subtract
Abstraction Level: fusion
Parent: none (top-level)
Children: linear, min_clamp, subtract

Operations: nn.Linear -> torch.min(x, constant) -> x - constant

Input Shapes:
  - x: [2, 16] dtype=float32

Output Shapes:
  - output: [2, 32] dtype=float32

Weight Shapes:
  - linear.weight: [32, 16]
  - linear.bias: [32]
  - constant: [] (scalar nn.Parameter, shared by min and subtract)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Full fusion of Linear -> Min -> Subtract operations.
    Performs a linear transformation, clamps to a maximum constant value,
    then subtracts the constant.
    Extracted from: 68_Matmul_Min_Subtract
    """
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, x):
        x = self.linear(x)
        x = torch.min(x, self.constant)
        x = x - self.constant
        return x


def get_inputs():
    return [torch.randn(2, 16)]


def get_init_inputs():
    return [16, 32, 2.0]


def get_expected_output_shape():
    return [(2, 32)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "NaN in output"
            assert not torch.isinf(output).any(), "Inf in output"
            expected_shapes = get_expected_output_shape()
            if isinstance(output, tuple):
                actual_shapes = [o.shape for o in output]
            else:
                actual_shapes = [output.shape]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), f"Shape mismatch: {actual} vs {expected}"
            # After min(x, c) - c, all values should be <= 0
            assert (output <= 1e-6).all(), "Output should be <= 0 after min(x,c) - c"
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
