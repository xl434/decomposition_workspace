"""
Level 0 Kernel: ReLU_3
Operation: ReLU activation
Input: [2, 120] -> Output: [2, 120]

Part of LeNet-5 (4_LeNet5) hierarchical decomposition.
Third ReLU activation applied after linear_1 (fc1).
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
            x: Input tensor of shape [batch_size, 120]
        Returns:
            Output tensor of shape [batch_size, 120]
        """
        return F.relu(x)


def get_inputs():
    """Return list of input tensors for this kernel."""
    batch_size = 2
    return [torch.randn(batch_size, 120)]


def get_init_inputs():
    """Return list of constructor arguments."""
    return []


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 120)


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

    print("relu_3: All tests passed!")
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
