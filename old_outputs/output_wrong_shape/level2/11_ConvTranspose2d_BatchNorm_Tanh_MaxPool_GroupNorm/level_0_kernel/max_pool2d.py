"""
Component: MaxPool2d
Abstraction Level: kernel
Parent: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm
Children: none

Operations: nn.MaxPool2d(kernel_size=2, stride=2)

Input Shapes:
  - x: [2, 8, 8, 8] dtype=float32

Output Shapes:
  - output: [2, 8, 4, 4] dtype=float32

Weight Shapes:
  - none
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    2D max pooling with kernel_size=2 and stride=2.
    Extracted from: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm
    """
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.max_pool(x)


def get_inputs():
    return [torch.randn(2, 8, 8, 8)]


def get_init_inputs():
    return [2, 2]


def get_expected_output_shape():
    return [(2, 8, 4, 4)]


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
