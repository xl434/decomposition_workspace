"""
Component: softmax_tanh_scale
Abstraction Level: fusion
Parent: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling (level_2_layer)
Children: softmax, tanh, scaling (level_0_kernel)

Operations: Softmax(dim=1) -> Tanh -> Scale(*scaling_factor)

Input Shapes:
  - x: (2, 4, 1, 4, 4) dtype=float32

Output Shapes:
  - output: (2, 4, 1, 4, 4) dtype=float32

Weight Shapes:
  - none (scaling_factor is a constant)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Fusion of softmax over channels, tanh activation, and constant scaling.
    Extracted from: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling (level 2 layer)
    """
    def __init__(self, scaling_factor=2.0):
        super().__init__()
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = torch.softmax(x, dim=1)
        x = torch.tanh(x)
        x = x * self.scaling_factor
        return x


def get_inputs():
    return [torch.randn(2, 4, 1, 4, 4)]


def get_init_inputs():
    return [2.0]


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
