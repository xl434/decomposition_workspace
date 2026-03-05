"""
Level 0 Kernel: Positional Embedding Addition
Adds a learnable positional embedding to the token sequence.
Input: [batch_size, num_patches+1, dim] = [2, 17, 32]
Output: [batch_size, num_patches+1, dim] = [2, 17, 32]
Weights: nn.Parameter pos_embedding [1, 17, 32]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_positions=17, dim=32):
        super(Model, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_positions, dim))

    def forward(self, x):
        return x + self.pos_embedding


def get_inputs():
    return [torch.randn(2, 17, 32)]


def get_init_inputs():
    return [17, 32]  # num_positions, dim


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
    # Verify output is input + pos_embedding
    expected_out = inputs[0] + model.pos_embedding
    assert torch.allclose(output, expected_out, atol=1e-6), "Positional embedding addition mismatch"
    print(f"pos_embedding_add: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
