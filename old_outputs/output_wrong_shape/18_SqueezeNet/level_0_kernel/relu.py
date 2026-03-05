"""
Level 0 Kernel: ReLU activation
Used throughout SqueezeNet after convolutions.
Input: [2, C, H, W] -> Output: [2, C, H, W] (same shape, values clamped to >= 0)
Test with: [2, 96, 13, 13]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """ReLU activation (inplace=True for memory efficiency)."""

    def __init__(self, inplace=True):
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x)


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 96, 13, 13)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use default: inplace=True


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 96, 13, 13)


def run_tests():
    """Verify ReLU produces correct output shape and non-negative values."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    assert (output >= 0).all(), "ReLU output contains negative values"
    print(f"PASSED: relu output shape {output.shape}, all values >= 0")
    return True


if __name__ == "__main__":
    run_tests()
