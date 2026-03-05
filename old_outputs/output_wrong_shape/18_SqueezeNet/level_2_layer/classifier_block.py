"""
Level 2 Layer: Classifier Block of SqueezeNet
Dropout(p=0.0) + Conv2d(512,num_classes,1) + ReLU + AdaptiveAvgPool2d(1,1) + Flatten

This includes the final Fire Module 8 output as input.

Composed of L0/L1 kernels:
  - dropout
  - conv2d_classifier + relu (fused)
  - adaptive_avg_pool2d
  - flatten

Input: [2, 512, 2, 2] -> Output: [2, 10]

Shape trace:
  Dropout(0.0):            [2,512,2,2] -> [2,512,2,2]
  Conv2d(512,10,1):        [2,512,2,2] -> [2,10,2,2]
  ReLU:                    [2,10,2,2] -> [2,10,2,2]
  AdaptiveAvgPool2d(1,1):  [2,10,2,2] -> [2,10,1,1]
  Flatten(1):              [2,10,1,1] -> [2,10]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """SqueezeNet classifier: dropout + 1x1 conv + relu + global avg pool + flatten."""

    def __init__(self, in_channels=512, num_classes=10, dropout_p=0.0):
        super(Model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.classifier(x)
        return torch.flatten(x, 1)


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 512, 2, 2)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use defaults: in_channels=512, num_classes=10, dropout_p=0.0


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 10)


def run_tests():
    """Verify classifier block produces correct output shape."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    print(f"PASSED: classifier_block output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
