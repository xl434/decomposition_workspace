"""
Level 0 Kernel: Conv2d_2
Operation: Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
Input: [2, 6, 14, 14] -> Output: [2, 16, 10, 10]

Part of LeNet-5 (4_LeNet5) hierarchical decomposition.
Second convolution layer extracting higher-level features.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Conv2d kernel: 6 input channels -> 16 output channels, 5x5 kernel."""

    def __init__(self, in_channels=6, out_channels=16, kernel_size=5, stride=1):
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
            x: Input tensor of shape [batch_size, 6, 14, 14]
        Returns:
            Output tensor of shape [batch_size, 16, 10, 10]
        """
        return self.conv(x)


def get_inputs():
    """Return list of input tensors for this kernel."""
    batch_size = 2
    return [torch.randn(batch_size, 6, 14, 14)]


def get_init_inputs():
    """Return list of constructor arguments."""
    return [6, 16, 5, 1]  # in_channels, out_channels, kernel_size, stride


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 16, 10, 10)


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
    x = inputs[0].clone().requires_grad_(True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should flow back to input"
    assert x.grad.shape == x.shape, "Gradient shape should match input shape"

    print("conv2d_2: All tests passed!")
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
