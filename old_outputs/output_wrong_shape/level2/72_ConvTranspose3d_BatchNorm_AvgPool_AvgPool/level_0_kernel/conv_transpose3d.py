"""
Component: conv_transpose3d
Abstraction Level: kernel
Parent: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool
Children: none

Operations: nn.ConvTranspose3d(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1)

Input Shapes:
  - x: [2, 2, 8, 8, 8] dtype=float32

Output Shapes:
  - output: [2, 4, 15, 15, 15] dtype=float32

Weight Shapes:
  - weight: [2, 4, 3, 3, 3]
  - bias: [4]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    3D transposed convolution layer.
    Extracted from: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        return self.conv_transpose(x)


def get_inputs():
    return [torch.randn(2, 2, 8, 8, 8)]


def get_init_inputs():
    return [2, 4, 3, 2, 1]


def get_expected_output_shape():
    return [(2, 4, 15, 15, 15)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "NaN"
            assert not torch.isinf(output).any(), "Inf"
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
