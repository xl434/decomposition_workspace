"""
Component: fire_module
Abstraction Level: fusion
Parent: features

Operations: [Conv2d (squeeze), ReLU, Conv2d (expand1x1), ReLU, Conv2d (expand3x3), ReLU, torch.cat]

Input Shapes:
  - x: (batch_size, in_channels, height, width) dtype=float32

Output Shapes:
  - output: (batch_size, expand1x1_channels+expand3x3_channels, height, width) dtype=float32

Weight Shapes:
  - squeeze.weight: (squeeze_channels, in_channels, 1, 1)
  - expand1x1.weight: (expand1x1_channels, squeeze_channels, 1, 1)
  - expand3x3.weight: (expand3x3_channels, squeeze_channels, 3, 3)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    FireModule: squeeze -> ReLU -> [expand1x1 -> ReLU || expand3x3 -> ReLU] -> cat
    """
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

def get_inputs():
    """Generate test inputs."""
    batch_size = 2
    return [torch.randn(batch_size, 96, 6, 6, dtype=torch.float32)]

def get_init_inputs():
    """Return initialization parameters."""
    return [96, 16, 64, 64]  # in_channels, squeeze_channels, expand1x1, expand3x3

def get_expected_output_shape():
    """Return expected output shape(s) for verification."""
    batch_size = 2
    return [(batch_size, 128, 6, 6)]  # 64 + 64 = 128 channels

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
