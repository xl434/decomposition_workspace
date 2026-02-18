"""
Component: Spatial MLP Conv1d (stage 0, dim=96)
Source: data/kernelbench/level3/29_SwinMLP.py
Abstraction Level: kernel
Parent: swin_mlp_block_spatial
Operations: [Conv1d(num_heads*window_size^2, num_heads*window_size^2, kernel_size=1, groups=num_heads)]
  For stage 0: num_heads=3, window_size=7 => channels = 3*49 = 147, groups=3
Input Shapes: [640, 147, 32] (num_windows*B, num_heads*window_size^2, C//num_heads)
  where C//num_heads = 96//3 = 32
Output Shapes: [640, 147, 32]
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        num_heads = 3
        window_size = 7
        channels = num_heads * window_size ** 2  # 3 * 49 = 147
        self.spatial_mlp = nn.Conv1d(channels, channels, kernel_size=1, groups=num_heads)

    def forward(self, x):
        return self.spatial_mlp(x)


def get_inputs():
    # 640 windows, 147 channels (3*49), 32 features (96//3)
    return [torch.randn(640, 147, 32)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(640, 147, 32)]


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
