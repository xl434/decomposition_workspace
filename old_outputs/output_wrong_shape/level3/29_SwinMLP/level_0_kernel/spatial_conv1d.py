"""
Level 0 Kernel: Spatial Conv1d
Conv1d for spatial mixing within SwinMLP windows.

Input: [8, 32, 8] -> Output: [8, 32, 8]

This is the core spatial mixing operation:
  - channels = num_heads * window_size^2 = 2 * 16 = 32
  - head_dim = dim / num_heads = 16 / 2 = 8
  - groups = num_heads = 2
  - Conv1d(32, 32, kernel_size=1, groups=2)

The Conv1d mixes spatial positions (window_size^2) within each attention head group.
Input shape: (nW*B, num_heads * window_size^2, head_dim) = (8, 32, 8)
  where nW = (8/4)*(8/4) = 4 windows, B=2, so nW*B = 8

This replaces the attention matrix multiplication in standard transformers.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Conv1d for spatial mixing in SwinMLP.

    Input: [nW*B, num_heads*window_size^2, head_dim] = [8, 32, 8]
    Output: [8, 32, 8]
    """
    def __init__(self):
        super().__init__()
        num_heads = 2
        window_size = 4
        channels = num_heads * window_size ** 2  # 2 * 16 = 32

        self.spatial_mlp = nn.Conv1d(
            channels, channels,
            kernel_size=1,
            groups=num_heads
        )

    def forward(self, x):
        # x: [8, 32, 8]
        return self.spatial_mlp(x)


def get_inputs():
    """Return sample inputs.
    nW*B = 4*2 = 8 (4 windows per image, batch of 2)
    channels = num_heads * window_size^2 = 2*16 = 32
    head_dim = dim / num_heads = 16/2 = 8
    """
    return [torch.randn(8, 32, 8)]


def get_init_inputs():
    """Return inputs for model initialization."""
    return []


def get_expected_output_shape():
    """Return expected output shapes."""
    return [(8, 32, 8)]


def run_tests():
    """Run validation tests."""
    print("=" * 60)
    print("Testing Level 0: Spatial Conv1d")
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

    # Verify Conv1d properties
    print(f"[INFO] Weight shape: {model.spatial_mlp.weight.shape}")  # [32, 16, 1]
    print(f"[INFO] Groups: {model.spatial_mlp.groups}")               # 2
    assert model.spatial_mlp.groups == 2
    print("[PASS] Groups = num_heads = 2 verified")

    # Verify grouped structure: head 0 uses channels 0-15, head 1 uses 16-31
    # Each group has 16 input channels and 16 output channels
    w = model.spatial_mlp.weight
    print(f"[INFO] Weight per group: in_ch={w.shape[1]}, out_ch_per_group={w.shape[0]//2}")

    print("\n[PASS] All Spatial Conv1d tests passed!")
    return True


if __name__ == "__main__":
    run_tests()
