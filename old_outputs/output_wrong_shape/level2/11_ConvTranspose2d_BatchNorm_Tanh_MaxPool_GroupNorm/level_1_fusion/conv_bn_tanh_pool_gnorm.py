"""
Component: ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm (Full Fusion)
Abstraction Level: fusion
Parent: none (top-level)
Children: conv_transpose2d, batch_norm2d, tanh, max_pool2d, group_norm

Operations: nn.ConvTranspose2d -> nn.BatchNorm2d -> nn.Tanh -> nn.MaxPool2d -> nn.GroupNorm

Input Shapes:
  - x: [2, 3, 8, 8] dtype=float32

Output Shapes:
  - output: [2, 8, 4, 4] dtype=float32

Weight Shapes:
  - conv_transpose.weight: [3, 8, 3, 3]
  - conv_transpose.bias: [8]
  - batch_norm.weight: [8]
  - batch_norm.bias: [8]
  - group_norm.weight: [8]
  - group_norm.bias: [8]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Fused ConvTranspose2d + BatchNorm2d + Tanh + MaxPool2d + GroupNorm pipeline.
    Extracted from: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm (top-level fusion)
    """
    def __init__(self, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, groups=4, num_groups=4):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.tanh(x)
        x = self.max_pool(x)
        x = self.group_norm(x)
        return x


def get_inputs():
    return [torch.randn(2, 3, 8, 8)]


def get_init_inputs():
    return [3, 8, 3, 1, 1, 4, 4]


def get_expected_output_shape():
    return [(2, 8, 4, 4)]


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
