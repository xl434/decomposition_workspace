"""
Level 1 Fusion: Parallel expand paths + cat
  - Path A: Conv2d(squeeze_channels, expand1x1_channels, 1) + ReLU
  - Path B: Conv2d(squeeze_channels, expand3x3_channels, 3, padding=1) + ReLU
  - Cat along dim=1
Fuses: conv2d_expand1x1 + relu + conv2d_expand3x3 + relu + cat

Test configuration (Fire Module 1):
Input: [2, 16, 7, 7] -> Output: [2, 128, 7, 7]

Shape trace:
  expand1x1 Conv2d(16,64,1)+ReLU: [2,16,7,7] -> [2,64,7,7]
  expand3x3 Conv2d(16,64,3,p=1)+ReLU: [2,16,7,7] -> [2,64,7,7]
  cat(dim=1): [2,64,7,7] + [2,64,7,7] -> [2,128,7,7]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Parallel 1x1 and 3x3 expand convolutions with ReLU, concatenated along channels."""

    def __init__(self, squeeze_channels=16, expand1x1_channels=64, expand3x3_channels=64):
        super(Model, self).__init__()
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out1x1 = self.expand1x1_activation(self.expand1x1(x))
        out3x3 = self.expand3x3_activation(self.expand3x3(x))
        return torch.cat([out1x1, out3x3], dim=1)


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 16, 7, 7)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use defaults: squeeze_channels=16, expand1x1_channels=64, expand3x3_channels=64


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 128, 7, 7)


def run_tests():
    """Verify expand+cat produces correct shape and non-negative values."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    assert (output >= 0).all(), "Output after ReLU should be non-negative"
    print(f"PASSED: expand_cat output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
