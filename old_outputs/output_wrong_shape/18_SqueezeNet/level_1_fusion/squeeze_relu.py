"""
Level 1 Fusion: Conv2d(in_channels, squeeze_channels, 1) + ReLU
Squeeze path of Fire Module: 1x1 convolution followed by ReLU activation.
Fuses: conv2d_squeeze + relu

Test configuration (Fire Module 1):
Input: [2, 96, 7, 7] -> Output: [2, 16, 7, 7]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """1x1 squeeze convolution followed by ReLU."""

    def __init__(self, in_channels=96, squeeze_channels=16):
        super(Model, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.squeeze_activation(x)
        return x


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
    """Verify squeeze+relu produces correct shape and non-negative values."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    assert (output >= 0).all(), "Output after ReLU should be non-negative"
    print(f"PASSED: squeeze_relu output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
