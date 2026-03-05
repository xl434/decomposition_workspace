"""
Level 0 Kernel: Conv2d(96, 16, kernel_size=1)
Squeeze convolution in Fire Module 1 of SqueezeNet.
Input: [2, 96, 7, 7] -> Output: [2, 16, 7, 7]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """1x1 squeeze convolution reducing channels from in_channels to squeeze_channels."""

    def __init__(self, in_channels=96, squeeze_channels=16):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 96, 7, 7)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use defaults: in_channels=96, squeeze_channels=16


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 16, 7, 7)


def run_tests():
    """Verify squeeze conv produces correct output shape."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    print(f"PASSED: conv2d_squeeze output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
