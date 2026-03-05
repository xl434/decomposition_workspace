"""
Level 0 Kernel: Conv2d(3, 96, kernel_size=7, stride=2)
Initial convolution in SqueezeNet.
Input: [2, 3, 32, 32] -> Output: [2, 96, 13, 13]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Conv2d with 3 input channels, 96 output channels, 7x7 kernel, stride 2."""

    def __init__(self, in_channels=3, out_channels=96, kernel_size=7, stride=2):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 3, 32, 32)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use defaults: in_channels=3, out_channels=96, kernel_size=7, stride=2


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 96, 13, 13)


def run_tests():
    """Verify the kernel produces correct output shape and runs without error."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    print(f"PASSED: conv2d_3_96 output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
