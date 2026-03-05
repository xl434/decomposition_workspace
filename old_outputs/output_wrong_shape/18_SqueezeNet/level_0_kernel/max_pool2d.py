"""
Level 0 Kernel: MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
Used in SqueezeNet feature extractor between fire module groups.
Test input: [2, 96, 13, 13] -> Output: [2, 96, 7, 7]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """MaxPool2d with 3x3 kernel, stride 2, ceil_mode=True."""

    def __init__(self, kernel_size=3, stride=2, ceil_mode=True):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode)

    def forward(self, x):
        return self.pool(x)


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 96, 13, 13)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use defaults: kernel_size=3, stride=2, ceil_mode=True


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 96, 7, 7)


def run_tests():
    """Verify MaxPool2d produces correct output shape."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    print(f"PASSED: max_pool2d output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
