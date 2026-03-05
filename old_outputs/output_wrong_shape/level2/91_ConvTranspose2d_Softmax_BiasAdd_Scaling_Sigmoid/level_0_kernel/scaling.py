"""
Component: scaling
Abstraction Level: kernel
Parent: conv_softmax_bias_scale_sigmoid
Children: none

Operations: x * scaling_factor (elementwise multiplication by constant)

Input Shapes:
  - x: [2, 8, 9, 9] dtype=float32

Output Shapes:
  - output: [2, 8, 9, 9] dtype=float32

Weight Shapes:
  - none (scaling_factor is a constant, not a learnable parameter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Multiplies input tensor by a constant scaling factor.
    Extracted from: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid
    """
    def __init__(self, scaling_factor):
        super().__init__()
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return x * self.scaling_factor


def get_inputs():
    return [torch.randn(2, 8, 9, 9)]


def get_init_inputs():
    return [2.0]  # scaling_factor


def get_expected_output_shape():
    return [(2, 8, 9, 9)]


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
            # Verify scaling property
            expected_output = inputs[0] * 2.0
            assert torch.allclose(output, expected_output, atol=1e-6), \
                "Scaling output does not match expected x * scaling_factor"
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
