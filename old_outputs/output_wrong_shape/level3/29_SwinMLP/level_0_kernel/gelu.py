"""
Level 0 Kernel: GELU Activation
GELU activation function applied element-wise.

Input: [2, 64, 32] -> Output: [2, 64, 32]

GELU(x) = x * Phi(x) where Phi is the standard Gaussian CDF.
Used between the up and down projections in the FFN/MLP block.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """GELU activation.

    Input: [B, 64, 32]
    Output: [B, 64, 32]
    """
    def __init__(self):
        super().__init__()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x)


def get_inputs():
    """Return sample inputs."""
    return [torch.randn(2, 64, 32)]


def get_init_inputs():
    """Return inputs for model initialization."""
    return []


def get_expected_output_shape():
    """Return expected output shapes."""
    return [(2, 64, 32)]


def run_tests():
    """Run validation tests."""
    print("=" * 60)
    print("Testing Level 0: GELU Activation")
    print("=" * 60)

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

    # Verify GELU properties
    x = inputs[0]
    # GELU(0) should be 0
    zero_test = model(torch.zeros(1))
    assert zero_test.item() == 0.0, "GELU(0) should be 0"
    print("[PASS] GELU(0) = 0 verified")

    # Large positive input: GELU(x) ~ x
    large_pos = torch.tensor([10.0])
    assert abs(model(large_pos).item() - 10.0) < 0.01
    print("[PASS] GELU(large) ~ x verified")

    print("\n[PASS] All GELU tests passed!")
    return True


if __name__ == "__main__":
    run_tests()
