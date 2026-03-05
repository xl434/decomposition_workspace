"""
Component: expert_down_combine
Level: 1 (Fusion)
Parent: mlp_block (Level 2)
Children: expert_einsum (L0), expert_weighted_combine (L0), residual_add (L0)
Operations: MLP2 einsum + bias + expert weighted sum + residual connection

Input Shapes:
    activated: [SEQ_LEN=16, EXPERTS_PER_TOKEN=2, INTERMEDIATE_SIZE=128] bfloat16
    expert_indices: [SEQ_LEN=16, EXPERTS_PER_TOKEN=2] int64
    expert_weights: [SEQ_LEN=16, EXPERTS_PER_TOKEN=2] bfloat16
    mlp2_weight: [NUM_EXPERTS=4, HIDDEN_SIZE=128, INTERMEDIATE_SIZE=128] bfloat16
    mlp2_bias: [NUM_EXPERTS=4, HIDDEN_SIZE=128] bfloat16
    residual_x: [SEQ_LEN=16, HIDDEN_SIZE=128] bfloat16

Output Shapes:
    output: [SEQ_LEN=16, HIDDEN_SIZE=128] bfloat16

Test Configuration:
    NUM_EXPERTS=4, INTERMEDIATE_SIZE=128, HIDDEN_SIZE=128, EXPERTS_PER_TOKEN=2
"""

import sys
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, activated, expert_indices, expert_weights, mlp2_weight, mlp2_bias, residual_x):
        # Gather
        w = mlp2_weight[expert_indices, ...]   # [16, 2, 128, 128]
        b = mlp2_bias[expert_indices, ...]     # [16, 2, 128]
        # MLP2 einsum
        t = torch.einsum("beck,bek->bec", w, activated)  # [16, 2, 128]
        t += b
        # Weighted combination
        t = torch.einsum("bec,be->bc", t, expert_weights)  # [16, 128]
        # Residual
        return residual_x + t


def get_inputs():
    return [
        torch.randn(16, 2, 128, dtype=torch.bfloat16),
        torch.randint(0, 4, (16, 2)),
        torch.randn(16, 2, dtype=torch.bfloat16),
        torch.randn(4, 128, 128, dtype=torch.bfloat16),
        torch.randn(4, 128, dtype=torch.bfloat16),
        torch.randn(16, 128, dtype=torch.bfloat16),
    ]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(16, 128)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        inputs = get_inputs()
        output = model(*inputs)

        # Handle single tensor output
        if isinstance(output, tuple):
            output = output[0]

        expected_shapes = get_expected_output_shape()

        # Check shape
        assert output.shape == torch.Size(expected_shapes[0]), \
            f"Output shape mismatch: expected {expected_shapes[0]}, got {tuple(output.shape)}"

        # Check dtype
        assert output.dtype == torch.bfloat16, \
            f"Output dtype mismatch: expected bfloat16, got {output.dtype}"

        # Check no NaN/Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        print("expert_down_combine: ALL TESTS PASSED")
        return True
    except Exception as e:
        print(f"expert_down_combine: TEST FAILED - {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
