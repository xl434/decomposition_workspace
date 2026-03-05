"""
Component: BiasAdd
Abstraction Level: kernel
Parent: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling
Children: none

Operations: Element-wise addition of learnable bias parameter with broadcasting.
    Bias shape (4,1,1,1) broadcasts over batch and spatial dims of 5D tensor [B,C,D,H,W].

Input Shapes:
  - x: [2, 4, 3, 3, 3] dtype=float32

Output Shapes:
  - output: [2, 4, 3, 3, 3] dtype=float32

Weight Shapes:
  - bias: [4, 1, 1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Learnable bias addition with broadcasting over spatial dimensions.
    Bias shape (out_channels, 1, 1, 1) broadcasts as (1, out_channels, 1, 1, 1) over 5D input.
    Extracted from: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling
    """
    def __init__(self, bias_shape=(4, 1, 1, 1)):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return x + self.bias


def get_inputs():
    return [torch.randn(2, 4, 3, 3, 3)]


def get_init_inputs():
    return [(4, 1, 1, 1)]


def get_expected_output_shape():
    return [(2, 4, 3, 3, 3)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "NaN"
            assert not torch.isinf(output).any(), "Inf"
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
