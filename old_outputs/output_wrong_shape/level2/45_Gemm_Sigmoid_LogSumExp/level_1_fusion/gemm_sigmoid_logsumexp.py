"""
Component: gemm_sigmoid_logsumexp
Abstraction Level: fusion
Parent: none (top-level)
Children: linear_1, sigmoid, linear_2, logsumexp

Operations: nn.Linear -> torch.sigmoid -> nn.Linear -> torch.logsumexp

Input Shapes:
  - x: [2, 16] dtype=float32

Output Shapes:
  - output: [2] dtype=float32

Weight Shapes:
  - linear1.weight: [32, 16]
  - linear1.bias: [32]
  - linear2.weight: [8, 32]
  - linear2.bias: [8]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Full fusion of Linear1 -> Sigmoid -> Linear2 -> LogSumExp.
    Extracted from: 45_Gemm_Sigmoid_LogSumExp (top-level)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.logsumexp(x, dim=1)
        return x


def get_inputs():
    return [torch.randn(2, 16)]


def get_init_inputs():
    return [16, 32, 8]


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
