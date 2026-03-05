"""
Component: max_pool2d
Abstraction Level: kernel
Parent: initial_conv_block, features

Operations: [MaxPool2d]

Input Shapes:
  - x: (batch_size, channels, height, width) dtype=float32

Output Shapes:
  - output: (batch_size, channels, out_height, out_width) dtype=float32

Weight Shapes:
  - None (no learnable parameters)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    MaxPool2d with ceil_mode=True.
    """
    def __init__(self, kernel_size=3, stride=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, ceil_mode=True)

    def forward(self, x):
        return self.maxpool(x)

def get_inputs():
    """Generate test inputs."""
    batch_size = 2
    return [torch.randn(batch_size, 96, 13, 13, dtype=torch.float32)]

def get_init_inputs():
    """Return initialization parameters."""
    return [3, 2]  # kernel_size, stride

def get_expected_output_shape():
    """Return expected output shape(s) for verification."""
    batch_size = 2
    # Input: 13x13, kernel=3, stride=2, ceil_mode=True -> ceil((13-3)/2) + 1 = ceil(5) + 1 = 6
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
