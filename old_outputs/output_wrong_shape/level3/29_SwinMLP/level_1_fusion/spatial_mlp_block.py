"""
Level 1 Fusion: Spatial MLP Block
The KEY novel component of SwinMLP - spatial mixing via Conv1d within windows.

window_partition -> reshape (group by heads) -> Conv1d (spatial_mlp) -> reshape -> window_reverse

Input: [2, 64, 16] -> Output: [2, 64, 16]
  (resolution 8x8, dim=16, num_heads=2, window_size=4, shift_size=0)

This is the spatial mixing operation that REPLACES self-attention in SwinMLP.
Instead of Q*K^T*V attention, it uses a learned spatial mixing matrix via Conv1d.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    """Partition into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Model(nn.Module):
    """Spatial MLP block: window partition + grouped Conv1d spatial mixing + window reverse.

    This is the core novel operation in SwinMLP that replaces self-attention.
    Uses Conv1d with groups=num_heads to mix spatial positions within each window.

    Input: [B, 64, 16]  (H=8, W=8, C=16)
    Output: [B, 64, 16]
    """
    def __init__(self):
        super().__init__()
        self.dim = 16
        self.num_heads = 2
        self.window_size = 4
        self.input_resolution = (8, 8)

        # Spatial MLP: Conv1d mixing spatial positions within each window
        # channels = num_heads * window_size^2, groups = num_heads
        # This means each head independently learns a spatial mixing for its window_size^2 positions
        self.spatial_mlp = nn.Conv1d(
            self.num_heads * self.window_size ** 2,
            self.num_heads * self.window_size ** 2,
            kernel_size=1,
            groups=self.num_heads
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)

        # Window partition
        x_windows = window_partition(x, self.window_size)
        # x_windows: (nW*B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        # x_windows: (nW*B, wh*ww, C)

        nW_B = x_windows.shape[0]
        wh_ww = self.window_size * self.window_size
        head_dim = C // self.num_heads

        # Reshape for grouped Conv1d:
        # (nW*B, wh*ww, C) -> (nW*B, wh*ww, num_heads, head_dim)
        # -> permute -> (nW*B, num_heads, wh*ww, head_dim)
        # -> reshape -> (nW*B, num_heads*wh*ww, head_dim)
        x_windows_heads = x_windows.view(nW_B, wh_ww, self.num_heads, head_dim)
        x_windows_heads = x_windows_heads.permute(0, 2, 1, 3)
        x_windows_heads = x_windows_heads.reshape(nW_B, self.num_heads * wh_ww, head_dim)

        # Apply spatial mixing via Conv1d
        x_windows_heads = self.spatial_mlp(x_windows_heads)

        # Reverse reshape
        x_windows_heads = x_windows_heads.view(nW_B, self.num_heads, wh_ww, head_dim)
        x_windows_heads = x_windows_heads.permute(0, 2, 1, 3)
        x_windows = x_windows_heads.reshape(nW_B, wh_ww, C)

        # Window reverse
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x_windows, self.window_size, H, W)

        x = x.view(B, H * W, C)
        return x


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
    print("Testing Level 1: Spatial MLP Block")
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

    # Verify the spatial_mlp Conv1d shape
    print(f"[INFO] spatial_mlp weight shape: {model.spatial_mlp.weight.shape}")
    # Expected: (num_heads*wh*ww, wh*ww, 1) = (2*16, 16, 1) = (32, 16, 1)
    expected_channels = model.num_heads * model.window_size ** 2
    assert model.spatial_mlp.weight.shape[0] == expected_channels
    print(f"[INFO] Spatial channels: {expected_channels}")
    print(f"[INFO] Groups: {model.spatial_mlp.groups}")

    # Verify window partition/reverse is lossless (without the MLP)
    x = torch.randn(2, 8, 8, 16)
    windows = window_partition(x, 4)
    recovered = window_reverse(windows, 4, 8, 8)
    assert torch.allclose(x, recovered), "Window partition/reverse not lossless"
    print("[PASS] Window partition/reverse is lossless")

    print("\n[PASS] All Spatial MLP Block tests passed!")
    return True


if __name__ == "__main__":
    run_tests()
