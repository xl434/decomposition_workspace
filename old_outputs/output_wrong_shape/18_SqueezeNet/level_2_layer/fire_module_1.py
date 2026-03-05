"""
Level 2 Layer: Fire Module 1 of SqueezeNet
FireModule(in_channels=96, squeeze_channels=16, expand1x1_channels=64, expand3x3_channels=64)

Composed of L1 fusions:
  - squeeze_relu: Conv2d(96,16,1) + ReLU
  - expand_cat: Conv2d(16,64,1)+ReLU || Conv2d(16,64,3,p=1)+ReLU -> cat

Input: [2, 96, 7, 7] -> Output: [2, 128, 7, 7]

Shape trace:
  squeeze Conv2d(96,16,1): [2,96,7,7] -> [2,16,7,7]
  ReLU:                    [2,16,7,7] -> [2,16,7,7]
  expand1x1 Conv2d(16,64,1)+ReLU: [2,16,7,7] -> [2,64,7,7]
  expand3x3 Conv2d(16,64,3,p=1)+ReLU: [2,16,7,7] -> [2,64,7,7]
  cat(dim=1): [2,64,7,7]+[2,64,7,7] -> [2,128,7,7]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Fire Module: squeeze-and-expand with parallel 1x1 and 3x3 paths."""

    def __init__(self, in_channels=96, squeeze_channels=16,
                 expand1x1_channels=64, expand3x3_channels=64):
        super(Model, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 96, 7, 7)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use defaults: in_channels=96, squeeze=16, expand1x1=64, expand3x3=64


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 128, 7, 7)


def run_tests():
    """Verify fire module 1 produces correct output shape."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    assert (output >= 0).all(), "Output after ReLU should be non-negative"
    print(f"PASSED: fire_module_1 output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
