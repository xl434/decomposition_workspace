"""
Level 0 Kernel: Layer Normalization
Applies layer normalization over the last dimension.
Input: [batch_size, seq_len, dim] = [2, 17, 32]
Output: [batch_size, seq_len, dim] = [2, 17, 32]
Weights: LayerNorm(32) - weight and bias of shape [32]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=32):
        super(Model, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.layer_norm(x)


def get_inputs():
    return [torch.randn(2, 17, 32)]


def get_init_inputs():
    return [32]  # dim


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
    # Verify normalization: mean ~0, std ~1 along last dim
    mean = output.mean(dim=-1)
    std = output.std(dim=-1, correction=0)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), "Mean should be ~0"
    assert torch.allclose(std, torch.ones_like(std), atol=1e-3), "Std should be ~1"
    print(f"layer_norm: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
