"""
Level 0 Kernel: Flatten + Transpose
flatten(2) followed by transpose(1, 2).

Input: [2, 16, 8, 8] -> Output: [2, 64, 16]

Converts Conv2d output (B, C, H, W) to sequence format (B, H*W, C).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Flatten spatial dims then transpose to sequence format.

    Input: [B, 16, 8, 8]
    Output: [B, 64, 16]
    """
    def __init__(self):
        super().__init__()
        # No learnable parameters

    def forward(self, x):
        # x: [B, 16, 8, 8]
        x = x.flatten(2)          # [B, 16, 64]
        x = x.transpose(1, 2)    # [B, 64, 16]
        return x


def get_inputs():
    """Return sample inputs."""
    return [torch.randn(2, 16, 8, 8)]


def get_init_inputs():
    """Return inputs for model initialization."""
    return []


def get_expected_output_shape():
    """Return expected output shapes."""
    return [(2, 64, 16)]


def run_tests():
    """Run validation tests."""
    print("=" * 60)
    print("Testing Level 0: Flatten + Transpose")
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

    # Verify values are preserved (just reordered)
    x = inputs[0]
    assert output[:, 0, :].equal(x[:, :, 0, 0]), \
        "First spatial position values should match first channel slice"
    print("[PASS] Value preservation verified")

    print("\n[PASS] All Flatten + Transpose tests passed!")
    return True


if __name__ == "__main__":
    run_tests()
