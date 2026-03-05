"""
Component: ExpertWeightedCombine
Level: 0 (Kernel)
Parent: MoELayer (Level 1)
Children: None (leaf kernel)
Operations: einsum weighted combination of expert outputs using expert routing weights
Input Shapes:
    expert_outputs: [16, 2, 128] bfloat16 - outputs from selected experts (n_tokens, experts_per_token, hidden_size)
    expert_weights: [16, 2] bfloat16 - routing weights for selected experts (n_tokens, experts_per_token)
Output Shapes:
    output: [16, 128] bfloat16 - weighted sum of expert outputs (n_tokens, hidden_size)
Weight Shapes: None (no learnable parameters)
"""

import sys
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, expert_outputs, expert_weights):
        # expert_outputs: [b, e, c] = [16, 2, 128]
        # expert_weights: [b, e] = [16, 2]
        return torch.einsum("bec,be->bc", expert_outputs, expert_weights)


def get_inputs():
    expert_outputs = torch.randn(16, 2, 128, dtype=torch.bfloat16)
    expert_weights = torch.randn(16, 2, dtype=torch.bfloat16)
    return [expert_outputs, expert_weights]


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

        # Verify correctness: with equal weights of 0.5, output should be mean of experts
        expert_out_test = torch.ones(2, 2, 4, dtype=torch.bfloat16)
        expert_out_test[:, 1, :] = 3.0  # Second expert outputs 3
        weights_test = torch.full((2, 2), 0.5, dtype=torch.bfloat16)
        out_test = model(expert_out_test, weights_test)
        # Expected: 0.5 * 1.0 + 0.5 * 3.0 = 2.0
        expected_val = torch.full((2, 4), 2.0, dtype=torch.bfloat16)
        if not torch.allclose(out_test, expected_val, atol=1e-2):
            print(
                f"FAIL: Correctness check failed. Expected {expected_val}, got {out_test}"
            )
            return False

        print("PASS: All tests passed for ExpertWeightedCombine")
        return True
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
