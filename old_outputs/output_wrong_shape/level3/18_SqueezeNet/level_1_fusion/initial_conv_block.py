"""
Component: initial_conv_block
Abstraction Level: fusion
Parent: features

Operations: [Conv2d, ReLU, MaxPool2d]

Input Shapes:
  - x: (batch_size, 3, height, width) dtype=float32

Output Shapes:
  - output: (batch_size, 96, out_height, out_width) dtype=float32

Weight Shapes:
  - conv.weight: (96, 3, 7, 7)
  - conv.bias: (96,)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Initial convolution block: Conv2d(3,96,7,stride=2) -> ReLU -> MaxPool2d(3,stride=2,ceil_mode=True)
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

def get_inputs():
    """Generate test inputs."""
    batch_size = 2
    return [torch.randn(batch_size, 3, 32, 32, dtype=torch.float32)]

def get_init_inputs():
    """Return initialization parameters."""
    return []

def get_expected_output_shape():
    """Return expected output shape(s) for verification."""
    batch_size = 2
    # Conv: 32 -> (32-7)//2 + 1 = 13
    # MaxPool: 13 -> ceil((13-3)/2) + 1 = 6
    return [(batch_size, 96, 6, 6)]

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
