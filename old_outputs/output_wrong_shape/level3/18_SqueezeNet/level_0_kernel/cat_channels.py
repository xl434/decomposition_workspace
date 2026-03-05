"""
Component: cat_channels
Abstraction Level: kernel
Parent: fire_module

Operations: [torch.cat on dim=1]

Input Shapes:
  - x1: (batch_size, channels1, height, width) dtype=float32
  - x2: (batch_size, channels2, height, width) dtype=float32

Output Shapes:
  - output: (batch_size, channels1+channels2, height, width) dtype=float32

Weight Shapes:
  - None (no learnable parameters)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Concatenate two tensors along the channel dimension (dim=1).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)

def get_inputs():
    """Generate test inputs."""
    batch_size = 2
    return [
        torch.randn(batch_size, 64, 6, 6, dtype=torch.float32),
        torch.randn(batch_size, 64, 6, 6, dtype=torch.float32)
    ]

def get_init_inputs():
    """Return initialization parameters."""
    return []

def get_expected_output_shape():
    """Return expected output shape(s) for verification."""
    batch_size = 2
    return [(batch_size, 128, 6, 6)]

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

            print(f"Input shapes: {[x.shape for x in inputs]}")
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
