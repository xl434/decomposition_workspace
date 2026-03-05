"""
Component: window_partition function
Source: data/kernelbench/level3/29_SwinMLP.py
Abstraction Level: kernel
Parent: swin_mlp_block_spatial
Operations: [view, permute, contiguous, view]
Input Shapes: [10, 56, 56, 96] (B, H, W, C) with window_size=7
Output Shapes: [640, 7, 7, 96] (num_windows*B, window_size, window_size, C)
Description: Partitions input tensor into non-overlapping windows.
  num_windows = (56/7) * (56/7) = 64, total = 10 * 64 = 640
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.window_size = 7

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows


def get_inputs():
    return [torch.randn(10, 56, 56, 96)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    # 10 * (56/7)*(56/7) = 10 * 64 = 640
    return [(640, 7, 7, 96)]


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
