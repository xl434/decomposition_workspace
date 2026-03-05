"""
Level 0 Kernel: Conv2d(16, 64, kernel_size=3, padding=1)
3x3 expand convolution in Fire Module 1 of SqueezeNet.
Input: [2, 16, 7, 7] -> Output: [2, 64, 7, 7]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """3x3 expand convolution from squeeze_channels to expand3x3_channels."""

    def __init__(self, squeeze_channels=16, expand3x3_channels=64):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 16, 7, 7)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use defaults: squeeze_channels=16, expand3x3_channels=64


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 64, 7, 7)


def run_tests():
    """Verify 3x3 expand conv produces correct output shape."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    print(f"PASSED: conv2d_expand3x3 output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
