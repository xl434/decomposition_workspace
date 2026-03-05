"""
Component: sigmoid
Abstraction Level: kernel
Parent: conv_softmax_bias_scale_sigmoid
Children: none

Operations: torch.sigmoid(x)

Input Shapes:
  - x: [2, 8, 9, 9] dtype=float32

Output Shapes:
  - output: [2, 8, 9, 9] dtype=float32

Weight Shapes:
  - none
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Sigmoid activation function applied elementwise.
    Extracted from: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


def get_inputs():
    return [torch.randn(2, 8, 9, 9)]


def get_init_inputs():
    return []


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
            # Verify sigmoid property: all outputs in [0, 1]
            assert (output >= 0.0).all() and (output <= 1.0).all(), \
                "Sigmoid output not in [0, 1] range"
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
