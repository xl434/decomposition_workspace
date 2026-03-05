"""
Component: conv2d
Abstraction Level: kernel
Parent: initial_conv_block, fire_module

Operations: [Conv2d]

Input Shapes:
  - x: (batch_size, in_channels, height, width) dtype=float32

Output Shapes:
  - output: (batch_size, out_channels, out_height, out_width) dtype=float32

Weight Shapes:
  - weight: (out_channels, in_channels, kernel_size, kernel_size)
  - bias: (out_channels,)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Parametric Conv2d operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)

def get_inputs():
    """Generate test inputs."""
    batch_size = 2
    return [torch.randn(batch_size, 3, 32, 32, dtype=torch.float32)]

def get_init_inputs():
    """Return initialization parameters."""
    return [3, 96, 7, 2, 0]  # in_channels, out_channels, kernel_size, stride, padding

def get_expected_output_shape():
    """Return expected output shape(s) for verification."""
    batch_size = 2
    # Input: 32x32, kernel=7, stride=2, padding=0 -> (32-7)//2 + 1 = 13
    return [(batch_size, 96, 13, 13)]

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
