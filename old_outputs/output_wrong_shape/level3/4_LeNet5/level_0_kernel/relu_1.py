"""
Level 0 Kernel: ReLU_1
Operation: ReLU activation
Input: [2, 6, 28, 28] -> Output: [2, 6, 28, 28]

Part of LeNet-5 (4_LeNet5) hierarchical decomposition.
First ReLU activation applied after conv2d_1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """ReLU activation kernel."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, 6, 28, 28]
        Returns:
            Output tensor of shape [batch_size, 6, 28, 28]
        """
        return F.relu(x)


def get_inputs():
    """Return list of input tensors for this kernel."""
    batch_size = 2
    return [torch.randn(batch_size, 6, 28, 28)]


def get_init_inputs():
    """Return list of constructor arguments."""
    return []


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 6, 28, 28)


def run_tests():
    """Validate the kernel produces correct output shapes and behavior."""
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)

    expected_shape = get_expected_output_shape()
    assert output.shape == expected_shape, (
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    )

    # Test ReLU behavior: no negative values in output
    assert (output >= 0).all(), "ReLU output should have no negative values"

    # Test with known input
    test_input = torch.tensor([[-1.0, 0.0, 1.0, 2.0]]).unsqueeze(-1).unsqueeze(-1)
    test_output = model(test_input)
    expected = torch.tensor([[0.0, 0.0, 1.0, 2.0]]).unsqueeze(-1).unsqueeze(-1)
    assert torch.allclose(test_output, expected), "ReLU should zero out negatives"

    print("relu_1: All tests passed!")
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
