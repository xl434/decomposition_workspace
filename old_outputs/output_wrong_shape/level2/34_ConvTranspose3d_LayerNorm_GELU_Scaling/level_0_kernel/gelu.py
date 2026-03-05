"""
Component: gelu
Abstraction Level: kernel
Parent: 34_ConvTranspose3d_LayerNorm_GELU_Scaling (level_1_fusion/conv_norm_gelu_scale.py)
Children: none

Operations: torch.nn.functional.gelu(x) - Gaussian Error Linear Unit activation

Input Shapes:
  - x: [2, 4, 2, 4, 4] dtype=float32

Output Shapes:
  - output: [2, 4, 2, 4, 4] dtype=float32

Weight Shapes:
  - none (no learnable parameters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    GELU activation function.
    Extracted from: 34_ConvTranspose3d_LayerNorm_GELU_Scaling
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)


def get_inputs():
    return [torch.randn(2, 4, 2, 4, 4)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(2, 4, 2, 4, 4)]


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
            # Verify GELU properties: GELU(0) = 0
            zero_input = torch.zeros(1)
            zero_output = F.gelu(zero_input)
            assert zero_output.abs().item() < 1e-6, "GELU(0) should be 0"
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
