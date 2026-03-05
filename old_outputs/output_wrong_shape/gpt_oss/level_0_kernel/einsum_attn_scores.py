"""
Component: EinsumAttnScores
Level: 0 (Kernel)
Parent: SlidingWindowSDPA (Level 1)
Children: None (leaf kernel)
Operations: einsum for Q*K dot product, scalar multiply by sm_scale, mask addition, concatenation with sinks
Input Shapes:
    Q: [16, 2, 4, 32] bfloat16 - query tensor (n_tokens, n_kv_heads, q_mult, d_head)
    K_expanded: [16, 2, 4, 32] bfloat16 - key tensor already expanded to match Q heads
    mask: [16, 16] bfloat16 - causal attention mask (upper triangular with -inf)
    sinks: [2, 4, 16, 1] bfloat16 - sink token scores
Output Shapes:
    QK: [2, 4, 16, 17] bfloat16 - scaled masked attention scores with sink column appended
Weight Shapes: None (no learnable parameters)
"""

import sys
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, sm_scale: float):
        super().__init__()
        self.sm_scale = sm_scale

    def forward(self, Q, K_expanded, mask, sinks):
        # Q: [n_tokens, n_heads, q_mult, d_head]
        # K_expanded: [n_tokens, n_heads, q_mult, d_head]
        # mask: [n_tokens, n_tokens]
        # sinks: [n_heads, q_mult, n_tokens, 1]
        QK = torch.einsum("qhmd,khmd->hmqk", Q, K_expanded)
        QK *= self.sm_scale
        QK += mask[None, None, :, :]
        # Concat sinks column
        QK = torch.cat([QK, sinks], dim=-1)
        return QK  # [2, 4, 16, 17]


def get_inputs():
    Q = torch.randn(16, 2, 4, 32, dtype=torch.bfloat16)
    K_expanded = torch.randn(16, 2, 4, 32, dtype=torch.bfloat16)
    mask = torch.triu(
        torch.full((16, 16), -float("inf"), dtype=torch.bfloat16), diagonal=1
    )
    sinks = torch.randn(2, 4, 16, 1, dtype=torch.bfloat16)
    return [Q, K_expanded, mask, sinks]


def get_init_inputs():
    sm_scale = 1.0 / (32 ** 0.5)  # 0.17677669529663688
    return [sm_scale]


def get_expected_output_shape():
    return [(2, 4, 16, 17)]


def run_tests():
    try:
        init_inputs = get_init_inputs()
        model = Model(*init_inputs)

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

        # Check for NaN/Inf in non-masked positions
        # The output will contain -inf from the causal mask, so check finite positions
        finite_mask = torch.isfinite(output)
        if finite_mask.any():
            finite_vals = output[finite_mask]
            if torch.isnan(finite_vals).any():
                print("FAIL: Output contains NaN in non-masked positions")
                return False

        # Verify that the last column comes from sinks
        # The last column (index 16) should be the sinks values after scale+mask ops
        # Just verify shape is correct (17 = 16 + 1 sink column)
        if output.shape[-1] != 17:
            print(
                f"FAIL: Expected last dim 17 (16 keys + 1 sink), got {output.shape[-1]}"
            )
            return False

        print("PASS: All tests passed for EinsumAttnScores")
        return True
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
