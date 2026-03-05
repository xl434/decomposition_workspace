"""
Level 0 Kernel: Conv2d(8, 4, kernel_size=3, padding=1, bias=False).

Applies 2D convolution: 8 input channels to 4 output channels.

Input:  [2, 8, 8, 8]
Output: [2, 4, 8, 8]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Conv2d with in_channels=8, out_channels=4, kernel_size=3, padding=1, no bias."""

    def __init__(self, in_channels: int = 8, out_channels: int = 4):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def get_inputs():
    """Return a list of input tensors for the model."""
    batch_size = 2
    in_channels = 8
    height = 8
    width = 8
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    """Return a list of arguments to initialize the model."""
    in_channels = 8
    out_channels = 4
    return [in_channels, out_channels]


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

    # Check spatial dimensions preserved (padding=1, kernel=3)
    assert output.shape[2] == inputs[0].shape[2], "Height should be preserved"
    assert output.shape[3] == inputs[0].shape[3], "Width should be preserved"

    # Check channel reduction: 8 -> 4
    assert output.shape[1] == 4, "Output channels should be 4"

    # Check no bias
    assert model.conv.bias is None, "Conv should have no bias"

    print("conv2d_8_4: All tests passed.")


if __name__ == "__main__":
    run_tests()
