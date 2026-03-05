"""
Component: bias_add
Abstraction Level: kernel
Parent: conv_mean_bias (level_1_fusion)
Children: none

Operations: x + self.bias  -- per-channel bias addition

Input Shapes:
  - x: (2, 4, 1, 4, 4) dtype=float32

Output Shapes:
  - output: (2, 4, 1, 4, 4) dtype=float32

Weight Shapes:
  - bias: (1, 4, 1, 1, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Per-channel bias addition with learnable bias parameter.
    Extracted from: conv_mean_bias fusion (level 1)
    """
    def __init__(self, out_channels=4):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))

    def forward(self, x):
        return x + self.bias


def get_inputs():
    return [torch.randn(2, 4, 1, 4, 4)]


def get_init_inputs():
    return [4]


def get_expected_output_shape():
    return [(2, 4, 1, 4, 4)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "NaN in output"
            assert not torch.isinf(output).any(), "Inf in output"
            expected_shapes = get_expected_output_shape()
            if isinstance(output, tuple):
                actual_shapes = [o.shape for o in output]
            else:
                actual_shapes = [output.shape]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), f"Shape mismatch: {actual} vs {expected}"
            print(f"Input shape(s): {[x.shape for x in inputs]}")
            print(f"Output shape(s): {actual_shapes}")
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
