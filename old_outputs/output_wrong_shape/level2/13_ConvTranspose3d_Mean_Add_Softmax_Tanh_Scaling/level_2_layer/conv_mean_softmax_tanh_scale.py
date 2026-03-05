"""
Component: conv_mean_softmax_tanh_scale
Abstraction Level: layer
Parent: none (top-level)
Children: conv_mean_bias, softmax_tanh_scale (level_1_fusion)

Operations: ConvTranspose3d -> Mean(dim=2) -> BiasAdd -> Softmax(dim=1) -> Tanh -> Scale(*2.0)

Input Shapes:
  - x: (2, 2, 4, 4, 4) dtype=float32

Output Shapes:
  - output: (2, 4, 1, 4, 4) dtype=float32

Weight Shapes:
  - conv_transpose.weight: (2, 4, 3, 3, 3)
  - conv_transpose.bias: (4,)
  - bias: (1, 4, 1, 1, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Full layer: ConvTranspose3d -> Mean -> BiasAdd -> Softmax -> Tanh -> Scaling.
    This is the top-level (level 2) component combining both level 1 fusions.
    Extracted from: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling
    """
    def __init__(self, in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1, scaling_factor=2.0):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.mean(dim=2, keepdim=True)
        x = x + self.bias
        x = torch.softmax(x, dim=1)
        x = torch.tanh(x)
        x = x * self.scaling_factor
        return x


def get_inputs():
    return [torch.randn(2, 2, 4, 4, 4)]


def get_init_inputs():
    return [2, 4, 3, 1, 1, 2.0]


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
