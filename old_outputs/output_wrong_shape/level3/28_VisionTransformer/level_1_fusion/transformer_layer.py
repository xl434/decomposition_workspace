"""
Level 1 Fusion: Single Transformer Encoder Layer
One complete TransformerEncoderLayer: self_attention + Add&Norm + FFN + Add&Norm.
Fuses: level_0_kernel/multihead_attention.py + level_0_kernel/layer_norm.py +
       level_0_kernel/linear_ffn_up.py + level_0_kernel/gelu.py +
       level_0_kernel/linear_ffn_down.py + level_0_kernel/layer_norm.py
Input: [batch_size, seq_len, dim] = [2, 17, 32]
Output: [batch_size, seq_len, dim] = [2, 17, 32]

Note: Uses nn.TransformerEncoderLayer which by default uses ReLU activation
and batch_first=False. The original ViT model passes [batch, seq, dim] tensors
to the TransformerEncoder without setting batch_first=True.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=32, heads=4, mlp_dim=64, dropout=0.0):
        super(Model, self).__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x):
        return self.layer(x)


def get_inputs():
    return [torch.randn(2, 17, 32)]


def get_init_inputs():
    return [32, 4, 64, 0.0]  # dim, heads, mlp_dim, dropout


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
    print(f"transformer_layer: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
