"""
Level 0 Kernel: LayerNorm
LayerNorm(16) applied to the last dimension.

Input: [2, 64, 16] -> Output: [2, 64, 16]

Normalizes each token's embedding independently.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """LayerNorm on embedding dimension.

    Input: [B, 64, 16]
    Output: [B, 64, 16]
    """
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(16)

    def forward(self, x):
        return self.norm(x)


def get_inputs():
    """Return sample inputs."""
    return [torch.randn(2, 64, 16)]


def get_init_inputs():
    """Return inputs for model initialization."""
    return []


def get_expected_output_shape():
    """Return expected output shapes."""
    return [(2, 64, 16)]


def run_tests():
    """Run validation tests."""
    print("=" * 60)
    print("Testing Level 0: LayerNorm")
    print("=" * 60)

    torch.manual_seed(42)
    model = Model()
    model.eval()

    inputs = get_inputs()
    expected_shapes = get_expected_output_shape()

    with torch.no_grad():
        output = model(*inputs)

    assert output.shape == expected_shapes[0], \
        f"Shape mismatch: {output.shape} vs {expected_shapes[0]}"
    print(f"[PASS] Output shape: {output.shape}")

    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print("[PASS] No NaN/Inf in output")

    # Verify normalization: mean ~0, std ~1 along last dim
    mean = output.mean(dim=-1)
    std = output.std(dim=-1, correction=0)
    assert mean.abs().max() < 1e-5, f"Mean not near zero: {mean.abs().max()}"
    assert (std - 1.0).abs().max() < 1e-4, f"Std not near one: {std.max()}"
    print("[PASS] Normalization verified (mean~0, std~1)")

    print("\n[PASS] All LayerNorm tests passed!")
    return True


if __name__ == "__main__":
    run_tests()
