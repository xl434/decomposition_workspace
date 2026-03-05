"""
Level 0 Kernel: Dropout with p=0.0.

Applies dropout with probability 0.0 (identity operation in practice).

Input:  [2, 4, 8, 8]
Output: [2, 4, 8, 8]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Dropout with p=0.0 (effectively identity)."""

    def __init__(self, p: float = 0.0):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


def get_inputs():
    """Return a list of input tensors for the model."""
    batch_size = 2
    channels = 4
    height = 8
    width = 8
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    """Return a list of arguments to initialize the model."""
    p = 0.0
    return [p]


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 4, 8, 8)


def run_tests():
    """Validate the model produces correct output shapes and values."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    output = model(*inputs)

    # Check output shape
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Expected shape {expected_shape}, got {output.shape}"

    # Check output is finite
    assert torch.isfinite(output).all(), "Output contains non-finite values"

    # With p=0.0, dropout should be identity in eval mode
    assert torch.allclose(output, inputs[0]), \
        "Dropout(0.0) in eval mode should be identity"

    # Also check in train mode with p=0.0
    model.train()
    train_output = model(inputs[0])
    assert torch.allclose(train_output, inputs[0]), \
        "Dropout(0.0) in train mode should also be identity"

    print("dropout: All tests passed.")


if __name__ == "__main__":
    run_tests()
