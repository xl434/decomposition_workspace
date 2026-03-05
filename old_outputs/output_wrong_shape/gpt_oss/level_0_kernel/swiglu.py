"""
Component: SwiGLU
Level: 0 (Kernel)
Parent: MoEExpertMLP (Level 1)
Children: None (leaf kernel)
Operations: interleaved split into glu/linear paths, clamping, sigmoid, SwiGLU gating
Input Shapes:
    x: [16, 2, 256] bfloat16 - MLP1 output before activation (n_tokens, experts_per_token, intermediate_size*2)
Output Shapes:
    output: [16, 2, 128] bfloat16 - activated output (n_tokens, experts_per_token, intermediate_size)
Weight Shapes: None (no learnable weight parameters; alpha and limit are scalar hyperparameters)
"""

import sys
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, alpha: float, limit: float):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def forward(self, x):
        # Split interleaved: even indices for glu, odd indices for linear
        x_glu = x[..., ::2]      # [16, 2, 128]
        x_linear = x[..., 1::2]  # [16, 2, 128]
        # Clamp values
        x_glu = x_glu.clamp(min=None, max=self.limit)
        x_linear = x_linear.clamp(min=-self.limit, max=self.limit)
        # SwiGLU: swish(x_glu) * (x_linear + 1)
        out_glu = x_glu * torch.sigmoid(self.alpha * x_glu)
        return out_glu * (x_linear + 1)


def get_inputs():
    x = torch.randn(16, 2, 256, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [1.702, 7.0]  # alpha, limit


def get_expected_output_shape():
    return [(16, 2, 128)]


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

        # Verify output dimension is half of input last dim
        if output.shape[-1] != inputs[0].shape[-1] // 2:
            print(
                f"FAIL: Output last dim {output.shape[-1]} should be half of "
                f"input last dim {inputs[0].shape[-1]}"
            )
            return False

        # Verify with known values: when x_glu=0, swish(0)=0, so output should be 0
        x_zero = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
        out_zero = model(x_zero)
        if not torch.allclose(
            out_zero, torch.zeros(1, 1, 128, dtype=torch.bfloat16), atol=1e-3
        ):
            print("FAIL: SwiGLU(0) should be 0")
            return False

        print("PASS: All tests passed for SwiGLU")
        return True
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
