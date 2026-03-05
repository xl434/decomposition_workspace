"""
Component: subtract
Abstraction Level: kernel
Parent: 68_Matmul_Min_Subtract (level_1_fusion/linear_min_subtract.py)
Children: none

Operations: x - constant (element-wise subtraction of a learned constant)

Input Shapes:
  - x: [2, 32] dtype=float32

Output Shapes:
  - output: [2, 32] dtype=float32

Weight Shapes:
  - constant: [] (scalar nn.Parameter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Element-wise subtraction of a learned constant parameter.
    Extracted from: 68_Matmul_Min_Subtract
    """
    def __init__(self, constant):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, x):
        return x - self.constant


def get_inputs():
    return [torch.randn(2, 32)]


def get_init_inputs():
    return [2.0]


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
            # Verify subtraction: output should be input - constant
            expected_output = inputs[0] - 2.0
            assert torch.allclose(output, expected_output, atol=1e-6), "Subtraction not correct"
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
