"""
Level 1 Fusion: Conv2d(3,96,7,stride=2) + ReLU + MaxPool2d(3,2,ceil=True)
Initial feature extraction block of SqueezeNet.
Fuses: conv2d_3_96 + relu + max_pool2d

Input: [2, 3, 32, 32] -> Output: [2, 96, 7, 7]

Shape trace:
  Conv2d(3,96,7,s=2): [2,3,32,32] -> [2,96,13,13]
  ReLU:               [2,96,13,13] -> [2,96,13,13]
  MaxPool2d(3,2,ceil): [2,96,13,13] -> [2,96,7,7]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Conv2d + ReLU + MaxPool2d fused into a single sequential block."""

    def __init__(self, in_channels=3, out_channels=96, kernel_size=7, stride=2,
                 pool_size=3, pool_stride=2, ceil_mode=True):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride, ceil_mode=ceil_mode)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 3, 32, 32)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use defaults


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 96, 7, 7)


def run_tests():
    """Verify fused conv+relu+pool produces correct output shape."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    assert (output >= 0).all(), "Output after ReLU should be non-negative"
    print(f"PASSED: conv_relu_pool output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
