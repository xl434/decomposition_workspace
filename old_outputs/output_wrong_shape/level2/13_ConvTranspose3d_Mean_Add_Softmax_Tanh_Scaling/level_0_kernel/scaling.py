"""
Component: scaling
Abstraction Level: kernel
Parent: softmax_tanh_scale (level_1_fusion)
Children: none

Operations: x * scaling_factor  -- element-wise scaling by a constant factor

Input Shapes:
  - x: (2, 4, 1, 4, 4) dtype=float32

Output Shapes:
  - output: (2, 4, 1, 4, 4) dtype=float32

Weight Shapes:
  - none (scaling_factor is a constant, not a learned parameter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Element-wise scaling by a constant factor.
    Extracted from: softmax_tanh_scale fusion (level 1)
    """
    def __init__(self, scaling_factor=2.0):
        super().__init__()
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return x * self.scaling_factor


def get_inputs():
    return [torch.randn(2, 4, 1, 4, 4)]


def get_init_inputs():
    return [2.0]


def get_expected_output_shape():
    return [(2, 4, 1, 4, 4)]


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
            # Verify scaling: output should be exactly 2x input
            x = inputs[0]
            expected_output = x * 2.0
            assert torch.allclose(output, expected_output, atol=1e-6), \
                "Scaling output does not match expected (input * 2.0)"
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
