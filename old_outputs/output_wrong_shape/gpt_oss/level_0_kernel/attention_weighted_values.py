"""
Component: AttentionWeightedValues
Level: 0 (Kernel)
Parent: SlidingWindowSDPA (Level 1)
Children: None (leaf kernel)
Operations: einsum for weighted combination of values, reshape to flatten head dimensions
Input Shapes:
    W: [2, 4, 16, 16] bfloat16 - attention weights (n_kv_heads, q_mult, n_tokens, n_tokens)
    V_expanded: [16, 2, 4, 32] bfloat16 - value tensor expanded (n_tokens, n_kv_heads, q_mult, d_head)
Output Shapes:
    attn: [16, 256] bfloat16 - attention output reshaped (n_tokens, n_kv_heads * q_mult * d_head)
Weight Shapes: None (no learnable parameters)
"""

import sys
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, W, V_expanded):
        # W: [n_kv_heads, q_mult, n_tokens, n_tokens]
        # V_expanded: [n_tokens, n_kv_heads, q_mult, d_head]
        attn = torch.einsum("hmqk,khmd->qhmd", W, V_expanded)
        # attn: [n_tokens, n_kv_heads, q_mult, d_head] = [16, 2, 4, 32]
        return attn.reshape(attn.shape[0], -1)  # [16, 256] = [16, 2*4*32]


def get_inputs():
    W = torch.randn(2, 4, 16, 16, dtype=torch.bfloat16)
    V_expanded = torch.randn(16, 2, 4, 32, dtype=torch.bfloat16)
    return [W, V_expanded]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(16, 256)]


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

        # Verify reshape is correct: 2 * 4 * 32 = 256
        if output.shape[1] != 2 * 4 * 32:
            print(
                f"FAIL: Expected flattened dim 256 (2*4*32), got {output.shape[1]}"
            )
            return False

        print("PASS: All tests passed for AttentionWeightedValues")
        return True
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
