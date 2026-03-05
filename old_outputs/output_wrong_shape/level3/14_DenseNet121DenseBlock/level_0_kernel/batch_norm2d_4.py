"""
Level 0 Kernel: BatchNorm2d with 4 features.

Applies batch normalization over a 4D input (B, 4, H, W).

Input:  [2, 4, 8, 8]
Output: [2, 4, 8, 8]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """BatchNorm2d with num_features=4."""

    def __init__(self, num_features: int = 4):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


def get_inputs():
    """Return a list of input tensors for the model."""
    batch_size = 2
    num_features = 4
    height = 8
    width = 8
    return [torch.randn(batch_size, num_features, height, width)]


def get_init_inputs():
    """Return a list of arguments to initialize the model."""
    num_features = 4
    return [num_features]


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

    # Check that BN preserves spatial dimensions
    assert output.shape[2:] == inputs[0].shape[2:], \
        "Spatial dimensions should be preserved"

    # Check that BN preserves channel count
    assert output.shape[1] == inputs[0].shape[1], \
        "Channel dimension should be preserved"

    print("batch_norm2d_4: All tests passed.")


if __name__ == "__main__":
    run_tests()
