"""
Level 0 Kernel: Dropout(p=0.0)
Dropout in SqueezeNet classifier (p=0.0 means no dropout, identity in eval).
Input: [2, 512, 2, 2] -> Output: [2, 512, 2, 2]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Dropout layer (p=0.0 effectively identity)."""

    def __init__(self, p=0.0):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        return self.dropout(x)


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 512, 2, 2)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use default: p=0.0


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 512, 2, 2)


def run_tests():
    """Verify dropout produces correct output shape (identity at p=0.0 in eval)."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    # At p=0.0, dropout is identity
    assert torch.allclose(output, inputs[0]), "Dropout(p=0.0) should be identity in eval mode"
    print(f"PASSED: dropout output shape {output.shape}, identity confirmed")
    return True


if __name__ == "__main__":
    run_tests()
