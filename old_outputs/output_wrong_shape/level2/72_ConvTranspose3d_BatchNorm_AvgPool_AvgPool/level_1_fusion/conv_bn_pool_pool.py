"""
Component: conv_bn_pool_pool
Abstraction Level: fusion
Parent: none (top-level)
Children: conv_transpose3d, batch_norm3d, avg_pool3d_1, avg_pool3d_2

Operations: nn.ConvTranspose3d -> nn.BatchNorm3d -> nn.AvgPool3d -> nn.AvgPool3d

Input Shapes:
  - x: [2, 2, 8, 8, 8] dtype=float32

Output Shapes:
  - output: [2, 4, 3, 3, 3] dtype=float32

Weight Shapes:
  - conv_transpose.weight: [2, 4, 3, 3, 3]
  - conv_transpose.bias: [4]
  - batch_norm.weight: [4]
  - batch_norm.bias: [4]
  - batch_norm.running_mean: [4]
  - batch_norm.running_var: [4]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Full fusion of ConvTranspose3d -> BatchNorm3d -> AvgPool3d -> AvgPool3d.
    Extracted from: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool (top-level)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.avg_pool1(x)
        x = self.avg_pool2(x)
        return x


def get_inputs():
    return [torch.randn(2, 2, 8, 8, 8)]


def get_init_inputs():
    return [2, 4, 3, 2, 1, (4, 1, 1, 1)]


def get_expected_output_shape():
    return [(2, 4, 3, 3, 3)]


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
