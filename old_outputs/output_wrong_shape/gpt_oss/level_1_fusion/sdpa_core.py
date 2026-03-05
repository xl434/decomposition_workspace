"""
Component: sdpa_core
Level: 1 (Fusion)
Parent: attention_block (Level 2)
Children: attention_mask (L0), einsum_attn_scores (L0), softmax_attn (L0),
          attention_weighted_values (L0)
Operations: Full scaled dot-product attention with attention sinks and sliding window

Input Shapes:
    Q: [SEQ_LEN=16, NUM_KEY_VALUE_HEADS=2, Q_MULT=4, HEAD_DIM=32] bfloat16
    K: [SEQ_LEN=16, NUM_KEY_VALUE_HEADS=2, HEAD_DIM=32] bfloat16
    V: [SEQ_LEN=16, NUM_KEY_VALUE_HEADS=2, HEAD_DIM=32] bfloat16
    S (sinks): [NUM_ATTENTION_HEADS=8] bfloat16

Output Shapes:
    attn_output: [SEQ_LEN=16, NUM_ATTENTION_HEADS*HEAD_DIM=256] bfloat16

Test Configuration:
    SM_SCALE=1/sqrt(32)=0.17677669529663688, SLIDING_WINDOW=8,
    NUM_ATTENTION_HEADS=8, HEAD_DIM=32
"""

import sys
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, sm_scale=0.17677669529663688, sliding_window=8):
        super().__init__()
        self.sm_scale = sm_scale
        self.sliding_window = sliding_window

    def forward(self, Q, K, V, S):
        n_tokens, n_heads, q_mult, d_head = Q.shape
        # Expand K and V for GQA
        K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
        V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
        # Reshape sinks
        S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
        # Causal mask
        mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
        # Sliding window mask
        if self.sliding_window > 0:
            mask += torch.tril(
                mask.new_full((n_tokens, n_tokens), -float("inf")),
                diagonal=-self.sliding_window,
            )
        # Attention scores
        QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
        QK *= self.sm_scale
        QK += mask[None, None, :, :]
        # Append sinks
        QK = torch.cat([QK, S], dim=-1)
        # Softmax
        W = torch.softmax(QK, dim=-1)
        # Remove sink column
        W = W[..., :-1]
        # Weighted sum of values
        attn = torch.einsum("hmqk,khmd->qhmd", W, V)
        return attn.reshape(n_tokens, -1)


def get_inputs():
    return [
        torch.randn(16, 2, 4, 32, dtype=torch.bfloat16),
        torch.randn(16, 2, 32, dtype=torch.bfloat16),
        torch.randn(16, 2, 32, dtype=torch.bfloat16),
        torch.randn(8, dtype=torch.bfloat16),
    ]


def get_init_inputs():
    return [0.17677669529663688, 8]


def get_expected_output_shape():
    return [(16, 256)]


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

        print("sdpa_core: ALL TESTS PASSED")
        return True
    except Exception as e:
        print(f"sdpa_core: TEST FAILED - {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
