"""
Level 0 Kernel: Conv2d_1
Operation: Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
Input: [2, 1, 32, 32] -> Output: [2, 6, 28, 28]

Part of LeNet-5 (4_LeNet5) hierarchical decomposition.
First convolution layer extracting low-level features from input images.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Conv2d kernel: 1 input channel -> 6 output channels, 5x5 kernel."""

    def __init__(self, in_channels=1, out_channels=6, kernel_size=5, stride=1):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, 1, 32, 32]
        Returns:
            Output tensor of shape [batch_size, 6, 28, 28]
        """
        return self.conv(x)


def get_inputs():
    """Return list of input tensors for this kernel."""
    batch_size = 2
    return [torch.randn(batch_size, 1, 32, 32)]


def get_init_inputs():
    """Return list of constructor arguments."""
    return [1, 6, 5, 1]  # in_channels, out_channels, kernel_size, stride


def get_expected_output_shape():
    """Return the expected output shape (excluding batch dimension verification)."""
    return (2, 6, 28, 28)


def run_tests():
    """Validate the kernel produces correct output shapes and runs without error."""
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)

    expected_shape = get_expected_output_shape()
    assert output.shape == expected_shape, (
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    )

    # Test that gradients flow
    loss = output.sum()
    loss.backward()
    assert inputs[0].grad is None  # Input doesn't require grad by default

    # Test with gradient tracking
    x = inputs[0].clone().requires_grad_(True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should flow back to input"
    assert x.grad.shape == x.shape, "Gradient shape should match input shape"

    print("conv2d_1: All tests passed!")
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
