"""
Level 0 Kernel: Linear FFN Down-Projection
Linear(32, 16) - second layer of the MLP/FFN block.

Input: [2, 64, 32] -> Output: [2, 64, 16]

Down-projects from hidden_dim=32 back to dim=16.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Linear down-projection for FFN.

    Input: [B, 64, 32]
    Output: [B, 64, 16]
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 16)

    def forward(self, x):
        return self.fc(x)


def get_inputs():
    """Return sample inputs."""
    return [torch.randn(2, 64, 32)]


def get_init_inputs():
    """Return inputs for model initialization."""
    return []


def get_expected_output_shape():
    """Return expected output shapes."""
    return [(2, 64, 16)]


def run_tests():
    """Run validation tests."""
    print("=" * 60)
    print("Testing Level 0: Linear FFN Down-Projection")
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

    print(f"[INFO] Weight shape: {model.fc.weight.shape}")   # [16, 32]
    print(f"[INFO] Bias shape: {model.fc.bias.shape}")       # [16]

    print("\n[PASS] All Linear FFN Down-Projection tests passed!")
    return True


if __name__ == "__main__":
    run_tests()
