"""
Level 0 Kernel: Linear Reduction (PatchMerging)
Linear(64, 32, bias=False) for the PatchMerging downsampling operation.

Input: [2, 16, 64] -> Output: [2, 16, 32]

PatchMerging concatenates 4 adjacent patches (2x2 grid) along channel dim,
then reduces: 4*dim -> 2*dim. For layer 0: 4*16=64 -> 2*16=32.

Note: In the full PatchMerging, a LayerNorm(4*dim) is applied before the linear.
This kernel only covers the linear reduction step.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Linear reduction for PatchMerging.

    Input: [B, 16, 64]   (seq_len=H/2*W/2=16, concat_dim=4*16=64)
    Output: [B, 16, 32]  (seq_len=16, reduced_dim=2*16=32)
    """
    def __init__(self):
        super().__init__()
        dim = 16
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        return self.reduction(x)


def get_inputs():
    """Return sample inputs."""
    return [torch.randn(2, 16, 64)]


def get_init_inputs():
    """Return inputs for model initialization."""
    return []


def get_expected_output_shape():
    """Return expected output shapes."""
    return [(2, 16, 32)]


def run_tests():
    """Run validation tests."""
    print("=" * 60)
    print("Testing Level 0: Linear Reduction (PatchMerging)")
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

    print(f"[INFO] Weight shape: {model.reduction.weight.shape}")  # [32, 64]
    assert model.reduction.bias is None, "Reduction should have no bias"
    print("[PASS] No bias confirmed")

    print("\n[PASS] All Linear Reduction tests passed!")
    return True


if __name__ == "__main__":
    run_tests()
