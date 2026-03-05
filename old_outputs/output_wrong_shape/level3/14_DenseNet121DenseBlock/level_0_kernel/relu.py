"""
Level 0 Kernel: ReLU activation (generic).

Applies ReLU element-wise. Works with any channel count.
For testing, uses C=4.

Input:  [2, 4, 8, 8]
Output: [2, 4, 8, 8]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """ReLU activation (inplace)."""

    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x)


def get_inputs():
    """Return a list of input tensors for the model."""
    batch_size = 2
    channels = 4
    height = 8
    width = 8
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    """Return a list of arguments to initialize the model."""
    return []


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 4, 8, 8)


def run_tests():
    """Validate the model produces correct output shapes and values."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    # Clone input since ReLU is inplace
    x = inputs[0].clone()
    output = model(x)

    # Check output shape
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Expected shape {expected_shape}, got {output.shape}"

    # Check output is finite
    assert torch.isfinite(output).all(), "Output contains non-finite values"

    # Check all values are non-negative (ReLU property)
    assert (output >= 0).all(), "ReLU output should be non-negative"

    # Check ReLU correctness: positive values preserved, negatives zeroed
    test_input = torch.tensor([[-1.0, 0.0, 1.0, 2.0]]).unsqueeze(-1).unsqueeze(-1)
    test_output = Model()(test_input.clone())
    expected = torch.tensor([[0.0, 0.0, 1.0, 2.0]]).unsqueeze(-1).unsqueeze(-1)
    assert torch.allclose(test_output, expected), \
        "ReLU should zero negatives and preserve positives"

    print("relu: All tests passed.")


if __name__ == "__main__":
    run_tests()
