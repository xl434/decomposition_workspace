"""
Component: ResidualAdd
Level: 0 (Kernel)
Parent: TransformerBlock (Level 2)
Children: None (leaf kernel)
Operations: element-wise addition of two tensors (residual connection)
Input Shapes:
    x: [16, 128] bfloat16 - transformed tensor (e.g., attention output or MoE output)
    residual: [16, 128] bfloat16 - residual tensor from skip connection
Output Shapes:
    output: [16, 128] bfloat16 - sum of x and residual
Weight Shapes: None (no learnable parameters)
"""

import sys
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, residual):
        return x + residual


def get_inputs():
    x = torch.randn(16, 128, dtype=torch.bfloat16)
    residual = torch.randn(16, 128, dtype=torch.bfloat16)
    return [x, residual]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(16, 128)]


def run_tests():
    try:
        model = Model(*get_init_inputs())

        inputs = get_inputs()
        output = model(*inputs)

        expected_shapes = get_expected_output_shape()

        # Check output shape
        if tuple(output.shape) != expected_shapes[0]:
            print(
                f"FAIL: Expected shape {expected_shapes[0]}, got {tuple(output.shape)}"
            )
            return False

        # Check output dtype
        if output.dtype != torch.bfloat16:
            print(f"FAIL: Expected dtype bfloat16, got {output.dtype}")
            return False

        # Check for NaN
        if torch.isnan(output).any():
            print("FAIL: Output contains NaN")
            return False

        # Check for Inf
        if torch.isinf(output).any():
            print("FAIL: Output contains Inf")
            return False

        # Verify correctness: output should equal x + residual
        x, residual = inputs
        expected = x + residual
        if not torch.allclose(output, expected, atol=0, rtol=0):
            print("FAIL: Output does not match x + residual")
            return False

        print("PASS: All tests passed for ResidualAdd")
        return True
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
