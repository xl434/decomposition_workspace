"""
Level 0 Kernel: Linear Classification Head
Linear(128, 10) for final classification.

Input: [2, 128] -> Output: [2, 10]

Maps from num_features=128 to num_classes=10.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Linear classification head.

    Input: [B, 128]
    Output: [B, 10]
    """
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        return self.head(x)


def get_inputs():
    """Return sample inputs."""
    return [torch.randn(2, 128)]


def get_init_inputs():
    """Return inputs for model initialization."""
    return []


def get_expected_output_shape():
    """Return expected output shapes."""
    return [(2, 10)]


def run_tests():
    """Run validation tests."""
    print("=" * 60)
    print("Testing Level 0: Linear Classification Head")
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

    print(f"[INFO] Weight shape: {model.head.weight.shape}")  # [10, 128]
    print(f"[INFO] Bias shape: {model.head.bias.shape}")      # [10]

    print("\n[PASS] All Linear Classification Head tests passed!")
    return True


if __name__ == "__main__":
    run_tests()
