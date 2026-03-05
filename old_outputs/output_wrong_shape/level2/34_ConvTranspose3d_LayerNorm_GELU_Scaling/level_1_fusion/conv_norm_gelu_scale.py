"""
Component: conv_norm_gelu_scale
Abstraction Level: fusion
Parent: none (top-level)
Children: conv_transpose3d, layer_norm, gelu, scaling

Operations: ConvTranspose3d -> LayerNorm -> GELU -> Scaling

Input Shapes:
  - x: [2, 2, 2, 4, 4] dtype=float32

Output Shapes:
  - output: [2, 4, 2, 4, 4] dtype=float32

Weight Shapes:
  - conv_transpose.weight: [2, 4, 1, 1, 1]
  - conv_transpose.bias: [4]
  - layer_norm.weight: [4]
  - layer_norm.bias: [4]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Full fusion of ConvTranspose3d -> LayerNorm -> GELU -> Scaling.
    Performs 3D transposed convolution, layer normalization over the last
    spatial dimension, GELU activation, and constant scaling.
    Extracted from: 34_ConvTranspose3d_LayerNorm_GELU_Scaling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias=True, eps=1e-5, scaling_factor=1.0):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.layer_norm(x)
        x = F.gelu(x)
        x = x * self.scaling_factor
        return x


def get_inputs():
    return [torch.randn(2, 2, 2, 4, 4)]


def get_init_inputs():
    return [2, 4, 1, 1, 0, True, 1e-5, 2.0]


def get_expected_output_shape():
    return [(2, 4, 2, 4, 4)]


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
