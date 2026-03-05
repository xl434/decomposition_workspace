"""
Component: Scale2
Abstraction Level: kernel
Parent: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling
Children: none

Operations: Element-wise multiplication by learnable scalar parameter (init=1.0)

Input Shapes:
  - x: [2, 4, 3, 3, 3] dtype=float32

Output Shapes:
  - output: [2, 4, 3, 3, 3] dtype=float32

Weight Shapes:
  - scale: scalar (torch.Size([]))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Element-wise scaling by a learnable scalar parameter.
    Extracted from: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling
    """
    def __init__(self, scale_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale_value))

    def forward(self, x):
        return x * self.scale


def get_inputs():
    return [torch.randn(2, 4, 3, 3, 3)]


def get_init_inputs():
    return [1.0]


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
