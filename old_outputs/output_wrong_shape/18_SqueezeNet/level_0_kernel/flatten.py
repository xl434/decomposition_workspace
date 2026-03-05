"""
Level 0 Kernel: torch.flatten(x, 1)
Flatten spatial dimensions in SqueezeNet classifier output.
Input: [2, 10, 1, 1] -> Output: [2, 10]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Flatten from start_dim=1 onward (keep batch dimension)."""

    def __init__(self, start_dim=1):
        super(Model, self).__init__()
        self.start_dim = start_dim

    def forward(self, x):
        return torch.flatten(x, self.start_dim)


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 10, 1, 1)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use default: start_dim=1


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 10)


def run_tests():
    """Verify flatten produces correct output shape."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    print(f"PASSED: flatten output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
