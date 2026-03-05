"""
Level 1 Fusion: conv_relu_pool_1
Fused operations: Conv2d(1,6,5) -> ReLU -> MaxPool2d(2,2)
Input: [2, 1, 32, 32] -> Output: [2, 6, 14, 14]

Part of LeNet-5 (4_LeNet5) hierarchical decomposition.
First convolutional block: convolution + activation + pooling.

Composed of Level 0 kernels:
  - conv2d_1: Conv2d(1, 6, 5)      [2,1,32,32] -> [2,6,28,28]
  - relu_1:   ReLU                  [2,6,28,28] -> [2,6,28,28]
  - max_pool2d_1: MaxPool2d(2,2)    [2,6,28,28] -> [2,6,14,14]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Fused Conv2d + ReLU + MaxPool2d block (first convolutional block)."""

    def __init__(self, in_channels=1, out_channels=6, kernel_size=5, stride=1,
                 pool_kernel_size=2, pool_stride=2):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, 1, 32, 32]
        Returns:
            Output tensor of shape [batch_size, 6, 14, 14]
        """
        x = self.conv(x)
        x = F.relu(x)
        x = F.max_pool2d(x, self.pool_kernel_size, self.pool_stride)
        return x


def get_inputs():
    """Return list of input tensors for this fusion."""
    batch_size = 2
    return [torch.randn(batch_size, 1, 32, 32)]


def get_init_inputs():
    """Return list of constructor arguments."""
    return [1, 6, 5, 1, 2, 2]  # in_ch, out_ch, k_size, stride, pool_k, pool_s


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 6, 14, 14)


def run_tests():
    """Validate the fusion produces correct output shapes and behavior."""
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)

    expected_shape = get_expected_output_shape()
    assert output.shape == expected_shape, (
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    )

    # Test ReLU + pool behavior: output should be non-negative (ReLU before pool)
    assert (output >= 0).all(), "Output should be non-negative after ReLU"

    # Test that gradients flow
    x = inputs[0].clone().requires_grad_(True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should flow back to input"

    # Test equivalence with sequential application of L0 kernels
    model.eval()
    with torch.no_grad():
        x_test = torch.randn(2, 1, 32, 32)
        fused_out = model(x_test)

        # Manual sequential
        h = model.conv(x_test)
        h = F.relu(h)
        h = F.max_pool2d(h, 2, 2)

        assert torch.allclose(fused_out, h), (
            "Fused output should match sequential kernel application"
        )

    print("conv_relu_pool_1: All tests passed!")
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
