"""
Level 0 Kernel: Conv2d Patch Projection
Conv2d(3, 16, kernel_size=4, stride=4) for patch embedding.

Input: [2, 3, 32, 32] -> Output: [2, 16, 8, 8]

This is the initial convolution that converts image patches into embeddings.
Each 4x4 patch of 3 channels is projected to a 16-dim embedding.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Conv2d patch projection.

    Input: [B, 3, 32, 32]
    Output: [B, 16, 8, 8]
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=4, stride=4)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    """Return sample inputs."""
    return [torch.randn(2, 3, 32, 32)]


def get_init_inputs():
    """Return inputs for model initialization."""
    return []


def get_expected_output_shape():
    """Return expected output shapes."""
    return [(2, 16, 8, 8)]


def run_tests():
    """Run validation tests."""
    print("=" * 60)
    print("Testing Level 0: Conv2d Patch Projection")
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

    # Verify kernel properties
    print(f"[INFO] Conv weight shape: {model.conv.weight.shape}")  # [16, 3, 4, 4]
    print(f"[INFO] Conv bias shape: {model.conv.bias.shape}")      # [16]

    print("\n[PASS] All Conv2d Patch Projection tests passed!")
    return True


if __name__ == "__main__":
    run_tests()
