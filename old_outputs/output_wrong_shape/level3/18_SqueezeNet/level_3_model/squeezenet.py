"""
Component: squeezenet
Abstraction Level: model
Parent: root

Operations: [features, classifier, flatten]

Input Shapes:
  - x: (batch_size, 3, height, width) dtype=float32

Output Shapes:
  - output: (batch_size, num_classes) dtype=float32

Weight Shapes:
  - Multiple conv layers from features and classifier
"""

import torch
import torch.nn as nn

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super().__init__()
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

class Model(nn.Module):
    """
    SqueezeNet: features -> classifier -> flatten
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(512, 64, 256, 256),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

def get_inputs():
    """Generate test inputs."""
    batch_size = 2
    return [torch.randn(batch_size, 3, 32, 32, dtype=torch.float32)]

def get_init_inputs():
    """Return initialization parameters."""
    return [10]  # num_classes

def get_expected_output_shape():
    """Return expected output shape(s) for verification."""
    batch_size = 2
    num_classes = 10
    return [(batch_size, num_classes)]

def run_tests():
    """Verify this component executes correctly."""
    try:
        model = Model(*get_init_inputs())
        model.eval()

        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)

            # 1. Basic validation
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"

            # 2. Shape validation
            expected_shapes = get_expected_output_shape()
            actual_shape = output.shape

            assert tuple(actual_shape) == tuple(expected_shapes[0]), \
                f"Shape mismatch: got {actual_shape}, expected {expected_shapes[0]}"

            # 3. Dtype validation
            expected_dtype = inputs[0].dtype
            assert output.dtype == expected_dtype, f"Dtype mismatch: {output.dtype} vs {expected_dtype}"

            print(f"Input shape: {inputs[0].shape}")
            print(f"Output shape: {actual_shape}")
            print("PASS")
            return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
