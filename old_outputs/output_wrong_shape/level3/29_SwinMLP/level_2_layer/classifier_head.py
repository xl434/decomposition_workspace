"""
Level 2 Layer: Classifier Head
LayerNorm + AdaptiveAvgPool1d + Flatten + Linear.

Input: [2, 1, 128] -> Output: [2, 10]

Params:
  - LayerNorm(128) (num_features = embed_dim * 2^(num_layers-1) = 16*8 = 128)
  - AdaptiveAvgPool1d(1)
  - Linear(128, 10)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Classifier Head: LayerNorm + AvgPool + Flatten + Linear.

    Input: [B, 1, 128]  (seq_len=1, dim=128 from last stage)
    Output: [B, 10]
    """
    def __init__(self):
        super().__init__()
        num_features = 128  # embed_dim * 2^(num_layers-1) = 16 * 8
        num_classes = 10

        self.norm = nn.LayerNorm(num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # x: [B, 1, 128]
        x = self.norm(x)               # [B, 1, 128]
        x = self.avgpool(x.transpose(1, 2))  # [B, 128, 1] -> [B, 128, 1]
        x = torch.flatten(x, 1)        # [B, 128]
        x = self.head(x)               # [B, 10]
        return x


def get_inputs():
    """Return sample inputs."""
    return [torch.randn(2, 1, 128)]


def get_init_inputs():
    """Return inputs for model initialization."""
    return []


def get_expected_output_shape():
    """Return expected output shapes."""
    return [(2, 10)]


def run_tests():
    """Run validation tests."""
    print("=" * 60)
    print("Testing Level 2: Classifier Head")
    print("=" * 60)

    torch.manual_seed(42)
    model = Model()
    model.eval()

    inputs = get_inputs()
    expected_shapes = get_expected_output_shape()

    with torch.no_grad():
        output = model(*inputs)

    # Shape test
    assert output.shape == expected_shapes[0], \
        f"Shape mismatch: {output.shape} vs {expected_shapes[0]}"
    print(f"[PASS] Output shape: {output.shape}")

    # No NaN/Inf
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print("[PASS] No NaN/Inf in output")

    # Verify intermediate shapes
    x = get_inputs()[0]
    normed = model.norm(x)
    print(f"[INFO] After LayerNorm: {normed.shape}")          # [2, 1, 128]
    pooled = model.avgpool(normed.transpose(1, 2))
    print(f"[INFO] After AvgPool: {pooled.shape}")             # [2, 128, 1]
    flat = torch.flatten(pooled, 1)
    print(f"[INFO] After Flatten: {flat.shape}")               # [2, 128]
    out = model.head(flat)
    print(f"[INFO] After Linear: {out.shape}")                 # [2, 10]

    print("\n[PASS] All Classifier Head tests passed!")
    return True


if __name__ == "__main__":
    run_tests()
