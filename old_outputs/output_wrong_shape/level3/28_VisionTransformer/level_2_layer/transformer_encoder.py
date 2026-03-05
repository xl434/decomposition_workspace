"""
Level 2 Layer: Transformer Encoder
Stack of TransformerEncoderLayers (depth=2).
Fuses: level_1_fusion/transformer_layer.py x 2
Input: [batch_size, seq_len, dim] = [2, 17, 32]
Output: [batch_size, seq_len, dim] = [2, 17, 32]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=32, depth=2, heads=4, mlp_dim=64, dropout=0.0):
        super(Model, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.transformer(x)


def get_inputs():
    return [torch.randn(2, 17, 32)]


def get_init_inputs():
    return [32, 2, 4, 64, 0.0]  # dim, depth, heads, mlp_dim, dropout


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

    # Verify decomposition: run each layer separately
    x = inputs[0]
    with torch.no_grad():
        step1 = model.transformer.layers[0](x)  # Layer 0
        step2 = model.transformer.layers[1](step1)  # Layer 1
    assert torch.allclose(output, step2, atol=1e-5), "Encoder vs per-layer mismatch"
    print(f"transformer_encoder: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
