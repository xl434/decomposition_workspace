"""
Component: attn_output_residual
Level: 1 (Fusion)
Parent: attention_block (Level 2)
Children: linear_proj (L0), residual_add (L0)
Operations: Output linear projection + residual connection

Input Shapes:
    attn_output: [SEQ_LEN=16, NUM_ATTENTION_HEADS*HEAD_DIM=256] bfloat16
    residual: [SEQ_LEN=16, HIDDEN_SIZE=128] bfloat16

Output Shapes:
    output: [SEQ_LEN=16, HIDDEN_SIZE=128] bfloat16

Test Configuration:
    in_features=256 (NUM_ATTENTION_HEADS * HEAD_DIM), out_features=128 (HIDDEN_SIZE)
"""

import sys
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features=256, out_features=128):
        super().__init__()
        self.out = nn.Linear(in_features, out_features, dtype=torch.bfloat16)

    def forward(self, t, residual):
        t = self.out(t)
        return residual + t


def get_inputs():
    return [
        torch.randn(16, 256, dtype=torch.bfloat16),
        torch.randn(16, 128, dtype=torch.bfloat16),
    ]


def get_init_inputs():
    return [256, 128]


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

        print("attn_output_residual: ALL TESTS PASSED")
        return True
    except Exception as e:
        print(f"attn_output_residual: TEST FAILED - {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
