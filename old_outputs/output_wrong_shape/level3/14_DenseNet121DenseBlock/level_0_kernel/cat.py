"""
Level 0 Kernel: torch.cat along dim=1.

Concatenates a list of tensors along the channel dimension (dim=1).
For test: cat([2,4,8,8], [2,4,8,8]) -> [2,8,8,8].

Input:  list of tensors, each [2, C_i, 8, 8]
Output: [2, sum(C_i), 8, 8]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Concatenation along channel dimension (dim=1)."""

    def __init__(self, dim: int = 1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        return torch.cat(tensors, dim=self.dim)


def get_inputs():
    """Return a list of input tensors for the model."""
    batch_size = 2
    channels = 4
    height = 8
    width = 8
    # Two tensors to concatenate
    t1 = torch.randn(batch_size, channels, height, width)
    t2 = torch.randn(batch_size, channels, height, width)
    return [t1, t2]


def get_init_inputs():
    """Return a list of arguments to initialize the model."""
    dim = 1
    return [dim]


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 8, 8, 8)


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

    # Check channel dimension is sum of inputs
    total_channels = sum(t.shape[1] for t in inputs)
    assert output.shape[1] == total_channels, \
        f"Expected {total_channels} channels, got {output.shape[1]}"

    # Check spatial dimensions preserved
    assert output.shape[2] == inputs[0].shape[2], "Height should be preserved"
    assert output.shape[3] == inputs[0].shape[3], "Width should be preserved"

    # Check that content matches: first part should equal first tensor
    assert torch.allclose(output[:, :4, :, :], inputs[0]), \
        "First 4 channels should match first input tensor"
    assert torch.allclose(output[:, 4:, :, :], inputs[1]), \
        "Last 4 channels should match second input tensor"

    # Test with 3 tensors (as in full DenseBlock after layer 2)
    t3 = torch.randn(2, 4, 8, 8)
    output3 = model(inputs[0], inputs[1], t3)
    assert output3.shape == torch.Size((2, 12, 8, 8)), \
        f"Expected (2,12,8,8) for 3-tensor cat, got {output3.shape}"

    print("cat: All tests passed.")


if __name__ == "__main__":
    run_tests()
