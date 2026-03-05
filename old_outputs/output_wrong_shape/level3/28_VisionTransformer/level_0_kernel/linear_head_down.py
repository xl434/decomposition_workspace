"""
Level 0 Kernel: MLP Head Down-Projection (Final Classification)
Linear layer that projects from mlp_dim to num_classes for final prediction.
Input: [batch_size, mlp_dim] = [2, 64]
Output: [batch_size, num_classes] = [2, 10]
Weights: Linear(64, 10)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, mlp_dim=64, num_classes=10):
        super(Model, self).__init__()
        self.linear = nn.Linear(mlp_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def get_inputs():
    return [torch.randn(2, 64)]


def get_init_inputs():
    return [64, 10]  # mlp_dim, num_classes


def get_expected_output_shape():
    return (2, 10)


def run_tests():
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"
    print(f"linear_head_down: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
