"""
Component: conv_softmax_bias_scale_sigmoid
Abstraction Level: fusion
Parent: none (top-level)
Children: conv_transpose2d, softmax, bias_add, scaling, sigmoid

Operations: ConvTranspose2d -> Softmax(dim=1) -> BiasAdd -> Scaling(*2.0) -> Sigmoid

Input Shapes:
  - x: [2, 3, 4, 4] dtype=float32

Output Shapes:
  - output: [2, 8, 9, 9] dtype=float32

Weight Shapes:
  - conv_transpose.weight: [3, 8, 4, 4]
  - conv_transpose.bias: [8]
  - bias: [8, 1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Fused ConvTranspose2d + Softmax + BiasAdd + Scaling + Sigmoid pipeline.
    This is the complete model 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid.
    Extracted from: original model (top-level fusion)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.softmax(x, dim=1)
        x = x + self.bias
        x = x * self.scaling_factor
        x = torch.sigmoid(x)
        return x


def get_inputs():
    return [torch.randn(2, 3, 4, 4)]


def get_init_inputs():
    return [3, 8, 4, 2, 1, 1, (8, 1, 1), 2.0]
    # in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor


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
            # Verify sigmoid output property: all values in [0, 1]
            assert (output >= 0.0).all() and (output <= 1.0).all(), \
                "Final sigmoid output not in [0, 1] range"
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
