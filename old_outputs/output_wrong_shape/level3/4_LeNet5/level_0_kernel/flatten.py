"""
Level 0 Kernel: Flatten
Operation: View/Reshape to flatten spatial dimensions
Input: [2, 16, 5, 5] -> Output: [2, 400]

Part of LeNet-5 (4_LeNet5) hierarchical decomposition.
Flattens the 2D feature maps into a 1D vector for the fully connected layers.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Flatten kernel: reshapes [B, C, H, W] -> [B, C*H*W]."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, 16, 5, 5]
        Returns:
            Output tensor of shape [batch_size, 400]
        """
        return x.view(x.size(0), -1)


def get_inputs():
    """Return list of input tensors for this kernel."""
    batch_size = 2
    return [torch.randn(batch_size, 16, 5, 5)]


def get_init_inputs():
    """Return list of constructor arguments."""
    return []


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 400)


def run_tests():
    """Validate the kernel produces correct output shapes and behavior."""
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)

    expected_shape = get_expected_output_shape()
    assert output.shape == expected_shape, (
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    )

    # Test that total number of elements is preserved
    assert inputs[0].numel() == output.numel(), (
        "Total number of elements should be preserved after flattening"
    )

    # Test that values are preserved (just reshaped)
    assert torch.allclose(inputs[0].view(2, -1), output), (
        "Values should be preserved after flattening"
    )

    # Test batch dimension is preserved
    assert output.shape[0] == inputs[0].shape[0], "Batch dimension should be preserved"

    print("flatten: All tests passed!")
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
