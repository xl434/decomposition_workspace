"""
Component: PatchMerging Downsample (reshape + concatenate)
Source: data/kernelbench/level3/29_SwinMLP.py
Abstraction Level: kernel
Parent: patch_merging
Operations: [view, slice(0::2, 0::2), slice(1::2, 0::2), slice(0::2, 1::2), slice(1::2, 1::2), cat]
Input Shapes: [10, 3136, 96]
Output Shapes: [10, 784, 384]
Description: Spatial downsampling by 2x via interleaved sampling and concatenation.
  H=56, W=56 => H/2=28, W/2=28 => L=784, C=4*96=384
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.H = 56
        self.W = 56

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]
        return x


def get_inputs():
    return [torch.randn(10, 3136, 96)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(10, 784, 384)]


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
