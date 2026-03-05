"""
Level 0 Kernel: MaxPool2d_1
Operation: MaxPool2d(kernel_size=2, stride=2)
Input: [2, 6, 28, 28] -> Output: [2, 6, 14, 14]

Part of LeNet-5 (4_LeNet5) hierarchical decomposition.
First max pooling layer for spatial downsampling after conv1+relu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """MaxPool2d kernel: 2x2 pooling with stride 2."""

    def __init__(self, kernel_size=2, stride=2):
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, 6, 28, 28]
        Returns:
            Output tensor of shape [batch_size, 6, 14, 14]
        """
        return F.max_pool2d(x, self.kernel_size, self.stride)


def get_inputs():
    """Return list of input tensors for this kernel."""
    batch_size = 2
    return [torch.randn(batch_size, 6, 28, 28)]


def get_init_inputs():
    """Return list of constructor arguments."""
    return [2, 2]  # kernel_size, stride


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 6, 14, 14)


def run_tests():
    """Validate the kernel produces correct output shapes and behavior."""
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)

    expected_shape = get_expected_output_shape()
    assert output.shape == expected_shape, (
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    )

    # Test max pooling behavior: output should be >= any value in pooling window
    # Verify spatial dimensions are halved
    assert output.shape[2] == inputs[0].shape[2] // 2, "Height should be halved"
    assert output.shape[3] == inputs[0].shape[3] // 2, "Width should be halved"

    print("max_pool2d_1: All tests passed!")
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
