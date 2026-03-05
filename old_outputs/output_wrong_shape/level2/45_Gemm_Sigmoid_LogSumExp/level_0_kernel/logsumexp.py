"""
Component: logsumexp
Abstraction Level: kernel
Parent: 45_Gemm_Sigmoid_LogSumExp
Children: none

Operations: torch.logsumexp(x, dim=1)

Input Shapes:
  - x: [2, 8] dtype=float32

Output Shapes:
  - output: [2] dtype=float32

Weight Shapes:
  - none
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    LogSumExp reduction over dim=1 (batch dimension reduction).
    Extracted from: 45_Gemm_Sigmoid_LogSumExp
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.logsumexp(x, dim=1)


def get_inputs():
    return [torch.randn(2, 8)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(2,)]


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
