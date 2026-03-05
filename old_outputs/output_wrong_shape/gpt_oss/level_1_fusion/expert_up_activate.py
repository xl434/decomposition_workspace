"""
Component: expert_up_activate
Level: 1 (Fusion)
Parent: mlp_block (Level 2)
Children: expert_einsum (L0), swiglu (L0)
Operations: Expert weight gather + MLP1 einsum + bias + SwiGLU activation

Input Shapes:
    normed_x: [SEQ_LEN=16, HIDDEN_SIZE=128] bfloat16
    expert_indices: [SEQ_LEN=16, EXPERTS_PER_TOKEN=2] int64
    mlp1_weight: [NUM_EXPERTS=4, INTERMEDIATE_SIZE*2=256, HIDDEN_SIZE=128] bfloat16
    mlp1_bias: [NUM_EXPERTS=4, INTERMEDIATE_SIZE*2=256] bfloat16

Output Shapes:
    output: [SEQ_LEN=16, EXPERTS_PER_TOKEN=2, INTERMEDIATE_SIZE=128] bfloat16

Test Configuration:
    NUM_EXPERTS=4, INTERMEDIATE_SIZE=128, HIDDEN_SIZE=128,
    EXPERTS_PER_TOKEN=2, SWIGLU_ALPHA=1.702, SWIGLU_LIMIT=7.0
"""

import sys
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, swiglu_alpha=1.702, swiglu_limit=7.0):
        super().__init__()
        self.alpha = swiglu_alpha
        self.limit = swiglu_limit

    def forward(self, normed_x, expert_indices, mlp1_weight, mlp1_bias):
        # Gather expert weights
        w = mlp1_weight[expert_indices, ...]   # [16, 2, 256, 128]
        b = mlp1_bias[expert_indices, ...]     # [16, 2, 256]
        # MLP1 einsum
        t = torch.einsum("beck,bk->bec", w, normed_x) + b  # [16, 2, 256]
        # SwiGLU
        x_glu, x_linear = t[..., ::2], t[..., 1::2]
        x_glu = x_glu.clamp(min=None, max=self.limit)
        x_linear = x_linear.clamp(min=-self.limit, max=self.limit)
        out_glu = x_glu * torch.sigmoid(self.alpha * x_glu)
        return out_glu * (x_linear + 1)  # [16, 2, 128]


def get_inputs():
    return [
        torch.randn(16, 128, dtype=torch.bfloat16),
        torch.randint(0, 4, (16, 2)),
        torch.randn(4, 256, 128, dtype=torch.bfloat16),
        torch.randn(4, 256, dtype=torch.bfloat16),
    ]


def get_init_inputs():
    return [1.702, 7.0]


def get_expected_output_shape():
    return [(16, 2, 128)]


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

        print("expert_up_activate: ALL TESTS PASSED")
        return True
    except Exception as e:
        print(f"expert_up_activate: TEST FAILED - {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
