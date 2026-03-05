"""
Level 0 Kernel: Multi-Head Attention
Applies multi-head self-attention over the token sequence.
Input: [batch_size, seq_len, dim] = [2, 17, 32]
Output: [batch_size, seq_len, dim] = [2, 17, 32]
Weights: nn.MultiheadAttention(embed_dim=32, num_heads=4)

Note: nn.TransformerEncoderLayer uses batch_first=False by default, so internally
the attention expects [seq, batch, dim]. This standalone kernel uses batch_first=True
for cleaner interface, but weight sharing with the original model is still compatible
since MultiheadAttention weights are the same regardless of batch_first.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=32, heads=4, dropout=0.0):
        super(Model, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        return attn_output


def get_inputs():
    return [torch.randn(2, 17, 32)]


def get_init_inputs():
    return [32, 4, 0.0]  # dim, heads, dropout


def get_expected_output_shape():
    return (2, 17, 32)


def run_tests():
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"
    print(f"multihead_attention: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
