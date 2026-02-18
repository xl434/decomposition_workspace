"""
Component: SwinMLPBlock Spatial MLP Branch (no shift)
Source: data/kernelbench/level3/29_SwinMLP.py
Abstraction Level: fusion
Parent: basic_layer_0
Operations: [LayerNorm, view, window_partition, reshape_heads, Conv1d_spatial, reshape_back, window_reverse, view, residual_add]
Input Shapes: [10, 3136, 96]
Output Shapes: [10, 3136, 96]
Description: The spatial MLP branch of SwinMLPBlock with shift_size=0 (even block).
  Applies norm1, partitions into windows, runs spatial MLP via Conv1d,
  reverses windows, and adds residual.
  Stage 0 params: dim=96, resolution=(56,56), num_heads=3, window_size=7
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 96
        self.num_heads = 3
        self.window_size = 7
        self.shift_size = 0
        self.input_resolution = (56, 56)

        self.norm1 = nn.LayerNorm(self.dim)
        self.spatial_mlp = nn.Conv1d(
            self.num_heads * self.window_size ** 2,
            self.num_heads * self.window_size ** 2,
            kernel_size=1,
            groups=self.num_heads
        )
        self.drop_path = nn.Identity()

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # No shift for shift_size=0
        shifted_x = x
        _, _H, _W, _ = shifted_x.shape

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, ws, ws, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Reshape for multi-head spatial MLP
        x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
        x_windows_heads = x_windows_heads.transpose(1, 2)
        x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size, C // self.num_heads)

        # Spatial MLP
        spatial_mlp_windows = self.spatial_mlp(x_windows_heads)

        # Reshape back
        spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size, C // self.num_heads).transpose(1, 2)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size, C)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)

        # Window reverse
        shifted_x = window_reverse(spatial_mlp_windows, self.window_size, _H, _W)

        # No reverse shift for shift_size=0
        x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        return x


def get_inputs():
    return [torch.randn(10, 3136, 96)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(10, 3136, 96)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            expected_shapes = get_expected_output_shape()
            if isinstance(output, tuple):
                actual_shapes = [o.shape for o in output]
            else:
                actual_shapes = [output.shape]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), f"Shape mismatch: {actual} vs {expected}"
            print(f"Input shape(s): {[x.shape for x in inputs]}")
            print(f"Output shape(s): {actual_shapes}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
