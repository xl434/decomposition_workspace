"""
Level 0 Kernel: FFN Down-Projection (within Transformer)
Linear layer that projects from mlp_dim back to dim in the feed-forward network.
Input: [batch_size, seq_len, mlp_dim] = [2, 17, 64]
Output: [batch_size, seq_len, dim] = [2, 17, 32]
Weights: Linear(64, 32)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, mlp_dim=64, dim=32):
        super(Model, self).__init__()
        self.linear = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        return self.linear(x)


def get_inputs():
    return [torch.randn(2, 17, 64)]


def get_init_inputs():
    return [64, 32]  # mlp_dim, dim


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
    print(f"linear_ffn_down: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
