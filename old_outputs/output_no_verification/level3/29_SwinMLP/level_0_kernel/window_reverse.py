"""
Component: window_reverse function
Source: data/kernelbench/level3/29_SwinMLP.py
Abstraction Level: kernel
Parent: swin_mlp_block_spatial
Operations: [view, permute, contiguous, view]
Input Shapes: [640, 7, 7, 96] (num_windows*B, window_size, window_size, C) with H=56, W=56
Output Shapes: [10, 56, 56, 96]
Description: Reverses window_partition, reassembling windows back into full feature map.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.window_size = 7
        self.H = 56
        self.W = 56

    def forward(self, windows):
        window_size = self.window_size
        H, W = self.H, self.W
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


def get_inputs():
    return [torch.randn(640, 7, 7, 96)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(10, 56, 56, 96)]


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
