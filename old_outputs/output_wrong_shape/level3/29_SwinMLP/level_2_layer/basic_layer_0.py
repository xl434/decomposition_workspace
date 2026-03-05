"""
Level 2 Layer: BasicLayer 0
First stage BasicLayer with 1 SwinMLPBlock + PatchMerging.

Input: [2, 64, 16] -> Output: [2, 16, 32]

Params for Layer 0:
  - dim=16, input_resolution=(8,8), depth=1, num_heads=2, window_size=4
  - mlp_ratio=2.0, drop=0.0, drop_path=0.0
  - PatchMerging: (8,8) -> (4,4), dim 16 -> 32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinMLPBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]

        self.norm1 = norm_layer(dim)
        self.spatial_mlp = nn.Conv1d(
            self.num_heads * self.window_size ** 2,
            self.num_heads * self.window_size ** 2,
            kernel_size=1,
            groups=self.num_heads
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
        else:
            shifted_x = x

        _, _H, _W, _ = shifted_x.shape

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        nW_B = x_windows.shape[0]
        wh_ww = self.window_size * self.window_size
        head_dim = C // self.num_heads

        x_windows_heads = x_windows.view(nW_B, wh_ww, self.num_heads, head_dim)
        x_windows_heads = x_windows_heads.permute(0, 2, 1, 3)
        x_windows_heads = x_windows_heads.reshape(nW_B, self.num_heads * wh_ww, head_dim)
        x_windows_heads = self.spatial_mlp(x_windows_heads)
        x_windows_heads = x_windows_heads.view(nW_B, self.num_heads, wh_ww, head_dim)
        x_windows_heads = x_windows_heads.permute(0, 2, 1, 3)
        x_windows = x_windows_heads.reshape(nW_B, wh_ww, C)

        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(x_windows, self.window_size, _H, _W)

        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:P_t + H, P_l:P_l + W, :].contiguous()
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class Model(nn.Module):
    """BasicLayer 0: 1 SwinMLPBlock + PatchMerging.

    Input: [B, 64, 16]  (resolution 8x8, dim=16)
    Output: [B, 16, 32]  (resolution 4x4, dim=32)
    """
    def __init__(self):
        super().__init__()
        dim = 16
        input_resolution = (8, 8)
        depth = 1
        num_heads = 2
        window_size = 4
        mlp_ratio = 2.0

        self.blocks = nn.ModuleList([
            SwinMLPBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=0.,
                drop_path=0.,
                norm_layer=nn.LayerNorm)
            for i in range(depth)
        ])

        self.downsample = PatchMerging(input_resolution, dim=dim,
                                        norm_layer=nn.LayerNorm)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x


def get_inputs():
    """Return sample inputs."""
    return [torch.randn(2, 64, 16)]


def get_init_inputs():
    """Return inputs for model initialization."""
    return []


def get_expected_output_shape():
    """Return expected output shapes."""
    return [(2, 16, 32)]


def run_tests():
    """Run validation tests."""
    print("=" * 60)
    print("Testing Level 2: BasicLayer 0")
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

    # Verify intermediate: after block but before downsample
    x = get_inputs()[0]
    for blk in model.blocks:
        x = blk(x)
    print(f"[INFO] After SwinMLPBlock: {x.shape}")  # [2, 64, 16]
    x = model.downsample(x)
    print(f"[INFO] After PatchMerging: {x.shape}")   # [2, 16, 32]

    print("\n[PASS] All BasicLayer 0 tests passed!")
    return True


if __name__ == "__main__":
    run_tests()
