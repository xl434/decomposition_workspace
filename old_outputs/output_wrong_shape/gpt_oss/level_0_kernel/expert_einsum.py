"""
Component: ExpertEinsum
Level: 0 (Kernel)
Parent: MoEExpertMLP (Level 1)
Children: None (leaf kernel)
Operations: batched einsum matrix multiply with bias addition for expert MLP layers
Input Shapes:
    weight: [16, 2, 256, 128] bfloat16 - expert weight matrices (batch, experts_per_token, out_features, in_features)
    x: [16, 128] bfloat16 - input tensor (batch/n_tokens, in_features)
    bias: [16, 2, 256] bfloat16 - expert bias vectors (batch, experts_per_token, out_features)
Output Shapes:
    output: [16, 2, 256] bfloat16 - result of einsum + bias (batch, experts_per_token, out_features)
Weight Shapes: None (weights and biases are passed as inputs from gathered expert parameters)
"""

import sys
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, weight, x, bias):
        # MLP1 pattern: "beck,bk->bec"
        # weight: [b, e, c, k] = [16, 2, 256, 128]
        # x: [b, k] = [16, 128]
        # bias: [b, e, c] = [16, 2, 256]
        t = torch.einsum("beck,bk->bec", weight, x) + bias
        return t  # [16, 2, 256]


def get_inputs():
    weight = torch.randn(16, 2, 256, 128, dtype=torch.bfloat16)
    x = torch.randn(16, 128, dtype=torch.bfloat16)
    bias = torch.randn(16, 2, 256, dtype=torch.bfloat16)
    return [weight, x, bias]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(16, 2, 256)]


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

        # Verify correctness with small known example
        weight_test = torch.ones(1, 1, 2, 3, dtype=torch.bfloat16)
        x_test = torch.ones(1, 3, dtype=torch.bfloat16)
        bias_test = torch.zeros(1, 1, 2, dtype=torch.bfloat16)
        out_test = model(weight_test, x_test, bias_test)
        # Each output element should be sum of 3 ones = 3.0
        expected_val = torch.full((1, 1, 2), 3.0, dtype=torch.bfloat16)
        if not torch.allclose(out_test, expected_val, atol=1e-2):
            print(
                f"FAIL: Correctness check failed. Expected {expected_val}, got {out_test}"
            )
            return False

        print("PASS: All tests passed for ExpertEinsum")
        return True
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
