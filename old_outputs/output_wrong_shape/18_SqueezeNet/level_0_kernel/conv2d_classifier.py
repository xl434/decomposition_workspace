"""
Level 0 Kernel: Conv2d(512, 10, kernel_size=1)
Classifier convolution in SqueezeNet (replaces fully-connected layer).
Input: [2, 512, 2, 2] -> Output: [2, 10, 2, 2]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """1x1 classifier convolution mapping features to num_classes."""

    def __init__(self, in_channels=512, num_classes=10):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 512, 2, 2)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use defaults: in_channels=512, num_classes=10


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 10, 2, 2)


def run_tests():
    """Verify classifier conv produces correct output shape."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    print(f"PASSED: conv2d_classifier output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
