"""
Composition Test for SwinMLP Decomposition
Verifies that:
1. Full model runs and produces correct output shape
2. Decomposed forward pass (step-by-step through sub-modules) matches full forward
3. Key intermediate shapes are correct
4. Individual components produce expected shapes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc
from itertools import repeat
import sys


# ==============================================================================
# Full model definition (self-contained copy for testing)
# ==============================================================================

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
        return x / keep_prob * random_tensor


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


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinMLPBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)
        ])

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
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinMLP(nn.Module):
    """Full SwinMLP model with test dimensions."""
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

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        self.patches_resolution = self.patch_embed.patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                  self.patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
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
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


# ==============================================================================
# Tests
# ==============================================================================

def test_full_model_forward():
    """Test 1: Full model forward pass produces correct shape."""
    print("Test 1: Full model forward pass")
    torch.manual_seed(42)
    model = SwinMLP()
    model.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 10), f"Expected (2,10), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print(f"  Output shape: {output.shape} -- PASS")
    return output, model


def test_decomposed_forward(model):
    """Test 2: Decomposed forward matches full forward."""
    print("Test 2: Decomposed forward vs full forward")
    torch.manual_seed(42)
    model.eval()

    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        # Full forward
        full_out = model(x)

        # Decomposed forward (step by step)
        emb = model.patch_embed(x)
        emb = model.pos_drop(emb)
        for layer in model.layers:
            emb = layer(emb)
        emb = model.norm(emb)
        emb = model.avgpool(emb.transpose(1, 2))
        emb = torch.flatten(emb, 1)
        decomposed_out = model.head(emb)

        max_diff = (full_out - decomposed_out).abs().max().item()
        print(f"  Max difference: {max_diff:.2e}")
        assert torch.allclose(full_out, decomposed_out, rtol=1e-4, atol=1e-5), \
            f"Decomposed output doesn't match. Max diff: {max_diff}"
    print("  PASS")


def test_intermediate_shapes(model):
    """Test 3: Verify all intermediate tensor shapes."""
    print("Test 3: Intermediate shape verification")
    model.eval()
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        # PatchEmbed
        emb = model.patch_embed(x)
        assert emb.shape == (2, 64, 16), f"PatchEmbed: {emb.shape}"
        print(f"  PatchEmbed: {emb.shape} == (2,64,16) -- PASS")

        emb = model.pos_drop(emb)

        # Layer 0: dim=16, res=(8,8) -> PatchMerging -> dim=32, res=(4,4)
        emb = model.layers[0](emb)
        assert emb.shape == (2, 16, 32), f"Layer 0: {emb.shape}"
        print(f"  Layer 0: {emb.shape} == (2,16,32) -- PASS")

        # Layer 1: dim=32, res=(4,4) -> PatchMerging -> dim=64, res=(2,2)
        emb = model.layers[1](emb)
        assert emb.shape == (2, 4, 64), f"Layer 1: {emb.shape}"
        print(f"  Layer 1: {emb.shape} == (2,4,64) -- PASS")

        # Layer 2: dim=64, res=(2,2) -> PatchMerging -> dim=128, res=(1,1)
        emb = model.layers[2](emb)
        assert emb.shape == (2, 1, 128), f"Layer 2: {emb.shape}"
        print(f"  Layer 2: {emb.shape} == (2,1,128) -- PASS")

        # Layer 3: dim=128, res=(1,1) -> no downsample
        emb = model.layers[3](emb)
        assert emb.shape == (2, 1, 128), f"Layer 3: {emb.shape}"
        print(f"  Layer 3: {emb.shape} == (2,1,128) -- PASS")

        # Norm
        emb = model.norm(emb)
        assert emb.shape == (2, 1, 128), f"Norm: {emb.shape}"
        print(f"  Norm: {emb.shape} == (2,1,128) -- PASS")

        # AvgPool
        pooled = model.avgpool(emb.transpose(1, 2))
        assert pooled.shape == (2, 128, 1), f"AvgPool: {pooled.shape}"
        print(f"  AvgPool: {pooled.shape} == (2,128,1) -- PASS")

        # Flatten
        flat = torch.flatten(pooled, 1)
        assert flat.shape == (2, 128), f"Flatten: {flat.shape}"
        print(f"  Flatten: {flat.shape} == (2,128) -- PASS")

        # Head
        out = model.head(flat)
        assert out.shape == (2, 10), f"Head: {out.shape}"
        print(f"  Head: {out.shape} == (2,10) -- PASS")


def test_window_partition_reverse():
    """Test 4: Window partition and reverse are lossless."""
    print("Test 4: Window partition/reverse lossless")
    x = torch.randn(2, 8, 8, 16)
    windows = window_partition(x, 4)
    assert windows.shape == (8, 4, 4, 16), f"Windows shape: {windows.shape}"
    print(f"  Windows shape: {windows.shape} == (8,4,4,16) -- PASS")

    recovered = window_reverse(windows, 4, 8, 8)
    assert recovered.shape == (2, 8, 8, 16), f"Recovered shape: {recovered.shape}"
    assert torch.allclose(x, recovered), "Window partition/reverse not lossless"
    print(f"  Recovered shape: {recovered.shape} == (2,8,8,16) -- PASS")
    print("  Lossless round-trip -- PASS")


def test_patch_merging_shapes():
    """Test 5: PatchMerging produces correct shapes at each stage."""
    print("Test 5: PatchMerging shapes")

    # Stage 0: (8,8), dim=16 -> (4,4), dim=32
    pm0 = PatchMerging((8, 8), dim=16)
    x0 = torch.randn(2, 64, 16)
    out0 = pm0(x0)
    assert out0.shape == (2, 16, 32), f"PM0: {out0.shape}"
    print(f"  PatchMerging(8,8,16): {out0.shape} == (2,16,32) -- PASS")

    # Stage 1: (4,4), dim=32 -> (2,2), dim=64
    pm1 = PatchMerging((4, 4), dim=32)
    x1 = torch.randn(2, 16, 32)
    out1 = pm1(x1)
    assert out1.shape == (2, 4, 64), f"PM1: {out1.shape}"
    print(f"  PatchMerging(4,4,32): {out1.shape} == (2,4,64) -- PASS")

    # Stage 2: (2,2), dim=64 -> (1,1), dim=128
    pm2 = PatchMerging((2, 2), dim=64)
    x2 = torch.randn(2, 4, 64)
    out2 = pm2(x2)
    assert out2.shape == (2, 1, 128), f"PM2: {out2.shape}"
    print(f"  PatchMerging(2,2,64): {out2.shape} == (2,1,128) -- PASS")


def test_spatial_mlp_block():
    """Test 6: SwinMLPBlock with spatial MLP."""
    print("Test 6: SwinMLPBlock shapes")

    # Block at layer 0: dim=16, res=(8,8), num_heads=2, window_size=4
    block = SwinMLPBlock(dim=16, input_resolution=(8, 8), num_heads=2,
                         window_size=4, shift_size=0, mlp_ratio=2.0)
    block.eval()
    x = torch.randn(2, 64, 16)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (2, 64, 16), f"Block: {out.shape}"
    print(f"  SwinMLPBlock(8,8,16): {out.shape} == (2,64,16) -- PASS")

    # Block at layer 3: dim=128, res=(1,1), num_heads=8
    # window_size will be clamped to min(1,1)=1
    block3 = SwinMLPBlock(dim=128, input_resolution=(1, 1), num_heads=8,
                          window_size=4, shift_size=0, mlp_ratio=2.0)
    block3.eval()
    x3 = torch.randn(2, 1, 128)
    with torch.no_grad():
        out3 = block3(x3)
    assert out3.shape == (2, 1, 128), f"Block3: {out3.shape}"
    print(f"  SwinMLPBlock(1,1,128): {out3.shape} == (2,1,128) -- PASS")


def test_gradient_flow():
    """Test 7: Verify gradients flow through the full model."""
    print("Test 7: Gradient flow")
    torch.manual_seed(42)
    model = SwinMLP()
    model.train()

    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "No gradient on input"
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
    print(f"  Input gradient shape: {x.grad.shape} -- PASS")

    # Check all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    print("  All parameters have valid gradients -- PASS")


def run_tests():
    """Run all composition tests."""
    print("=" * 70)
    print("SwinMLP Decomposition - Composition Test")
    print("=" * 70)
    print()

    passed = 0
    failed = 0

    try:
        output, model = test_full_model_forward()
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1
        model = None

    print()

    if model is not None:
        try:
            test_decomposed_forward(model)
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

        print()

        try:
            test_intermediate_shapes(model)
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print()

    try:
        test_window_partition_reverse()
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    print()

    try:
        test_patch_merging_shapes()
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    print()

    try:
        test_spatial_mlp_block()
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    print()

    try:
        test_gradient_flow()
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
    return True


if __name__ == "__main__":
    run_tests()
