"""
Level 0 Kernel: Linear Embedding
Projects flattened patches to the embedding dimension.
Input: [batch_size, num_patches, patch_dim] = [2, 16, 48]
Output: [batch_size, num_patches, dim] = [2, 16, 32]
Weights: Linear(48, 32)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, patch_dim=48, dim=32):
        super(Model, self).__init__()
        self.linear = nn.Linear(patch_dim, dim)

    def forward(self, x):
        return self.linear(x)


def get_inputs():
    return [torch.randn(2, 16, 48)]


def get_init_inputs():
    return [48, 32]  # patch_dim, dim


def get_expected_output_shape():
    return (2, 16, 32)


def run_tests():
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"
    print(f"linear_embed: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
