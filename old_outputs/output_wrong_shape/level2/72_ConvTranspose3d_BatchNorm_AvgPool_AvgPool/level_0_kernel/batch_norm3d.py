"""
Component: batch_norm3d
Abstraction Level: kernel
Parent: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool
Children: none

Operations: nn.BatchNorm3d(num_features=4)

Input Shapes:
  - x: [2, 4, 15, 15, 15] dtype=float32

Output Shapes:
  - output: [2, 4, 15, 15, 15] dtype=float32

Weight Shapes:
  - weight: [4]
  - bias: [4]
  - running_mean: [4]
  - running_var: [4]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    3D batch normalization layer.
    Extracted from: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool
    """
    def __init__(self, num_features):
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(num_features)

    def forward(self, x):
        return self.batch_norm(x)


def get_inputs():
    return [torch.randn(2, 4, 15, 15, 15)]


def get_init_inputs():
    return [4]


def get_expected_output_shape():
    return [(2, 4, 15, 15, 15)]


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
