"""
Level 3 Full Model: SwinMLP
Complete self-contained implementation of SwinMLP with all helper classes.
Input: [2, 3, 32, 32] -> Output: [2, 10]

Test dimensions:
  img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=16,
  depths=[1,1,1,1], num_heads=[2,4,4,8], window_size=4, mlp_ratio=2.0,
  drop_rate=0.0, drop_path_rate=0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc
from itertools import repeat
import numpy as np


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
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
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""
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
    """
    Partition input into non-overlapping windows.
    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition.
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: int
        H, W: int
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinMLPBlock(nn.Module):
    """Swin MLP Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads (used for spatial grouping).
        window_size (int): Window size.
        shift_size (int): Shift size for shifted window.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float): Dropout rate.
        drop_path (float): Stochastic depth rate.
        act_layer (nn.Module): Activation layer.
        norm_layer (nn.Module): Normalization layer.
    """
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
            # If window size is larger than input resolution, no partition
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]

        self.norm1 = norm_layer(dim)

        # Spatial MLP: operates on spatial tokens within each window
        # Input: (num_windows*B, window_size*window_size, C)
        # Conv1d with groups=num_heads mixes spatial positions per head
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
        assert L == H * W, f"input feature has wrong size: {L} vs {H * W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
        else:
            shifted_x = x

        _, _H, _W, _ = shifted_x.shape

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        # x_windows: (nW*B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        # x_windows: (nW*B, window_size*window_size, C)

        # Spatial MLP
        # Reshape for grouped conv: (nW*B, C/num_heads, num_heads*window_size^2)
        # Actually the SwinMLP paper does:
        # (nW*B, wh*ww, C) -> reshape to (nW*B, wh*ww, num_heads, C//num_heads)
        # -> permute to (nW*B, num_heads, wh*ww, C//num_heads)
        # -> reshape to (nW*B, num_heads*wh*ww, C//num_heads)
        # -> conv1d(groups=num_heads) -> reverse reshape
        nW_B = x_windows.shape[0]
        wh_ww = self.window_size * self.window_size
        head_dim = C // self.num_heads

        x_windows_heads = x_windows.view(nW_B, wh_ww, self.num_heads, head_dim)
        x_windows_heads = x_windows_heads.permute(0, 2, 1, 3)  # (nW*B, num_heads, wh*ww, head_dim)
        x_windows_heads = x_windows_heads.reshape(nW_B, self.num_heads * wh_ww, head_dim)

        x_windows_heads = self.spatial_mlp(x_windows_heads)  # Conv1d: (nW*B, num_heads*wh*ww, head_dim)

        x_windows_heads = x_windows_heads.view(nW_B, self.num_heads, wh_ww, head_dim)
        x_windows_heads = x_windows_heads.permute(0, 2, 1, 3)  # (nW*B, wh*ww, num_heads, head_dim)
        x_windows = x_windows_heads.reshape(nW_B, wh_ww, C)

        # Merge windows
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(x_windows, self.window_size, _H, _W)

        # Reverse shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:P_t + H, P_l:P_l + W, :].contiguous()
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Residual + FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """A basic Swin MLP layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float): Dropout rate.
        drop_path (float or tuple[float]): Stochastic depth rate.
        norm_layer (nn.Module): Normalization layer.
        downsample (nn.Module): Downsample layer at the end of the layer.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinMLPBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)
        ])

        # Downsample
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.
    Args:
        img_size (int): Image size. Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module): Normalization layer.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class Model(nn.Module):
    """SwinMLP model with test dimensions.

    Test params:
        img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=16,
        depths=[1,1,1,1], num_heads=[2,4,4,8], window_size=4, mlp_ratio=2.0,
        drop_rate=0.0, drop_path_rate=0.0

    Input: [2, 3, 32, 32]
    Output: [2, 10]
    """
    def __init__(self):
        super().__init__()

        img_size = 32
        patch_size = 4
        in_chans = 3
        num_classes = 10
        embed_dim = 16
        depths = [1, 1, 1, 1]
        num_heads = [2, 4, 4, 8]
        window_size = 4
        mlp_ratio = 2.0
        drop_rate = 0.0
        drop_path_rate = 0.0
        norm_layer = nn.LayerNorm

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def get_inputs():
    """Return sample inputs for the model."""
    return [torch.randn(2, 3, 32, 32)]


def get_init_inputs():
    """Return inputs for model initialization."""
    return []


def get_expected_output_shape():
    """Return expected output shapes."""
    return [(2, 10)]


def run_tests():
    """Run validation tests."""
    print("=" * 60)
    print("Testing Level 3: SwinMLP Full Model")
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

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total parameters: {total_params}")

    # Verify intermediate shapes
    x = inputs[0]
    x = model.patch_embed(x)
    print(f"[INFO] After PatchEmbed: {x.shape}")  # [2, 64, 16]
    x = model.pos_drop(x)
    for i, layer in enumerate(model.layers):
        x = layer(x)
        print(f"[INFO] After Layer {i}: {x.shape}")
    x = model.norm(x)
    print(f"[INFO] After Norm: {x.shape}")
    x = model.avgpool(x.transpose(1, 2))
    print(f"[INFO] After AvgPool: {x.shape}")
    x = torch.flatten(x, 1)
    print(f"[INFO] After Flatten: {x.shape}")
    x = model.head(x)
    print(f"[INFO] After Head: {x.shape}")

    print("\n[PASS] All Level 3 tests passed!")
    return True


if __name__ == "__main__":
    run_tests()
