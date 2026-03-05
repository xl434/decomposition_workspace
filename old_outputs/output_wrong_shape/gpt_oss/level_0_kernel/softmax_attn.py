"""
Component: SoftmaxAttn
Level: 0 (Kernel)
Parent: SlidingWindowSDPA (Level 1)
Children: None (leaf kernel)
Operations: softmax over last dimension, slice to remove sink column
Input Shapes:
    QK: [2, 4, 16, 17] bfloat16 - scaled masked attention scores with sink column
Output Shapes:
    W: [2, 4, 16, 16] bfloat16 - attention weights (sink column removed)
Weight Shapes: None (no learnable parameters)
"""

import sys
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, QK):
        # QK: [n_heads, q_mult, n_tokens, n_tokens + 1]
        W = torch.softmax(QK, dim=-1)
        W = W[..., :-1]  # Remove sink column
        return W  # [2, 4, 16, 16]


def get_inputs():
    QK = torch.randn(2, 4, 16, 17, dtype=torch.bfloat16)
    return [QK]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(2, 4, 16, 16)]


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

        # Softmax outputs should be non-negative
        if (output < 0).any():
            print("FAIL: Softmax output contains negative values")
            return False

        # Softmax outputs (before trimming) should sum to 1 along last dim
        # After trimming the sink column, the sum should be <= 1
        row_sums = output.sum(dim=-1)
        if (row_sums > 1.01).any():
            print("FAIL: Attention weights sum exceeds 1.0 (after sink removal)")
            return False

        print("PASS: All tests passed for SoftmaxAttn")
        return True
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
