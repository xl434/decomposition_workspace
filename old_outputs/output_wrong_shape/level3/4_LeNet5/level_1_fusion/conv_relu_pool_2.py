"""
Level 1 Fusion: conv_relu_pool_2
Fused operations: Conv2d(6,16,5) -> ReLU -> MaxPool2d(2,2)
Input: [2, 6, 14, 14] -> Output: [2, 16, 5, 5]

Part of LeNet-5 (4_LeNet5) hierarchical decomposition.
Second convolutional block: convolution + activation + pooling.

Composed of Level 0 kernels:
  - conv2d_2: Conv2d(6, 16, 5)     [2,6,14,14] -> [2,16,10,10]
  - relu_2:   ReLU                  [2,16,10,10] -> [2,16,10,10]
  - max_pool2d_2: MaxPool2d(2,2)    [2,16,10,10] -> [2,16,5,5]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Fused Conv2d + ReLU + MaxPool2d block (second convolutional block)."""

    def __init__(self, in_channels=6, out_channels=16, kernel_size=5, stride=1,
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
            x: Input tensor of shape [batch_size, 6, 14, 14]
        Returns:
            Output tensor of shape [batch_size, 16, 5, 5]
        """
        x = self.conv(x)
        x = F.relu(x)
        x = F.max_pool2d(x, self.pool_kernel_size, self.pool_stride)
        return x


def get_inputs():
    """Return list of input tensors for this fusion."""
    batch_size = 2
    return [torch.randn(batch_size, 6, 14, 14)]


def get_init_inputs():
    """Return list of constructor arguments."""
    return [6, 16, 5, 1, 2, 2]  # in_ch, out_ch, k_size, stride, pool_k, pool_s


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 16, 5, 5)


def run_tests():
    """Validate the fusion produces correct output shapes and behavior."""
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)

    expected_shape = get_expected_output_shape()
    assert output.shape == expected_shape, (
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    )

    # Test ReLU + pool behavior: output should be non-negative
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
        x_test = torch.randn(2, 6, 14, 14)
        fused_out = model(x_test)

        # Manual sequential
        h = model.conv(x_test)
        h = F.relu(h)
        h = F.max_pool2d(h, 2, 2)

        assert torch.allclose(fused_out, h), (
            "Fused output should match sequential kernel application"
        )

    print("conv_relu_pool_2: All tests passed!")
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
