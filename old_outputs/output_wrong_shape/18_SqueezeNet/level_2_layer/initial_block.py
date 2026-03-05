"""
Level 2 Layer: Initial Block of SqueezeNet
Conv2d(3,96,7,stride=2) + ReLU + MaxPool2d(3,2,ceil_mode=True)

Composed of L1 fusions:
  - conv_relu_pool (Conv2d + ReLU + MaxPool2d)

Input: [2, 3, 32, 32] -> Output: [2, 96, 7, 7]

Shape trace:
  Conv2d(3,96,7,s=2): [2,3,32,32] -> [2,96,13,13]
  ReLU:               [2,96,13,13] -> [2,96,13,13]
  MaxPool2d(3,2,ceil): [2,96,13,13] -> [2,96,7,7]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """SqueezeNet initial block: convolution + activation + pooling."""

    def __init__(self, in_channels=3, out_channels=96):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

    def forward(self, x):
        return self.features(x)


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 3, 32, 32)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use defaults: in_channels=3, out_channels=96


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 96, 7, 7)


def run_tests():
    """Verify initial block produces correct output shape."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    assert (output >= 0).all(), "Output after ReLU should be non-negative"
    print(f"PASSED: initial_block output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
