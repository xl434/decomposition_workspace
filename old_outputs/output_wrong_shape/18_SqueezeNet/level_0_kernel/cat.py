"""
Level 0 Kernel: torch.cat along dim=1 (channel dimension)
Concatenates expand1x1 and expand3x3 outputs in Fire Module.
Input: [2, 64, 7, 7] and [2, 64, 7, 7] -> Output: [2, 128, 7, 7]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Concatenation along channel dimension (dim=1)."""

    def __init__(self, dim=1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=self.dim)


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 64, 7, 7), torch.randn(2, 64, 7, 7)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use default: dim=1


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 128, 7, 7)


def run_tests():
    """Verify cat produces correct output shape."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    print(f"PASSED: cat output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
