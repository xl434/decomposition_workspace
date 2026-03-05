"""
Level 2 Layer: PatchEmbed
Conv2d projection + flatten + transpose + LayerNorm.

Input: [2, 3, 32, 32] -> Output: [2, 64, 16]

Params: img_size=32, patch_size=4, in_chans=3, embed_dim=16
  - patches_resolution = (8, 8)
  - num_patches = 64
  - Conv2d(3, 16, kernel_size=4, stride=4)
  - LayerNorm(16)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """PatchEmbed: Conv2d patch projection + LayerNorm.

    Input: [B, 3, 32, 32]
    Output: [B, 64, 16]
    """
    def __init__(self):
        super().__init__()
        img_size = 32
        patch_size = 4
        in_chans = 3
        embed_dim = 16

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, 3, 32, 32]
        x = self.proj(x)           # [B, 16, 8, 8]
        x = x.flatten(2)           # [B, 16, 64]
        x = x.transpose(1, 2)     # [B, 64, 16]
        x = self.norm(x)           # [B, 64, 16]
        return x


def get_inputs():
    """Return sample inputs."""
    return [torch.randn(2, 3, 32, 32)]


def get_init_inputs():
    """Return inputs for model initialization."""
    return []


def get_expected_output_shape():
    """Return expected output shapes."""
    return [(2, 64, 16)]


def run_tests():
    """Run validation tests."""
    print("=" * 60)
    print("Testing Level 2: PatchEmbed")
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
    x = inputs[0]
    proj_out = model.proj(x)
    print(f"[INFO] After Conv2d: {proj_out.shape}")  # [2, 16, 8, 8]
    flat = proj_out.flatten(2)
    print(f"[INFO] After flatten: {flat.shape}")       # [2, 16, 64]
    trans = flat.transpose(1, 2)
    print(f"[INFO] After transpose: {trans.shape}")     # [2, 64, 16]

    print("\n[PASS] All PatchEmbed tests passed!")
    return True


if __name__ == "__main__":
    run_tests()
