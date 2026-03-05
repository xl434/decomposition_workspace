"""
Level 0 Kernel: AdaptiveAvgPool2d((1, 1))
Global average pooling in SqueezeNet classifier.
Input: [2, 10, 2, 2] -> Output: [2, 10, 1, 1]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Adaptive average pooling to 1x1 spatial output."""

    def __init__(self, output_size=(1, 1)):
        super(Model, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return self.pool(x)


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 10, 2, 2)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use default: output_size=(1, 1)


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 10, 1, 1)


def run_tests():
    """Verify adaptive avg pool produces correct output shape."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    print(f"PASSED: adaptive_avg_pool2d output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
