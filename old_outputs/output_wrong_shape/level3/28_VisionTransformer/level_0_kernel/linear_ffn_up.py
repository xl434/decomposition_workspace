"""
Level 0 Kernel: FFN Up-Projection (within Transformer)
Linear layer that projects from dim to mlp_dim in the feed-forward network.
Input: [batch_size, seq_len, dim] = [2, 17, 32]
Output: [batch_size, seq_len, mlp_dim] = [2, 17, 64]
Weights: Linear(32, 64)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=32, mlp_dim=64):
        super(Model, self).__init__()
        self.linear = nn.Linear(dim, mlp_dim)

    def forward(self, x):
        return self.linear(x)


def get_inputs():
    return [torch.randn(2, 17, 32)]


def get_init_inputs():
    return [32, 64]  # dim, mlp_dim


def get_expected_output_shape():
    return (2, 17, 64)


def run_tests():
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"
    print(f"linear_ffn_up: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
