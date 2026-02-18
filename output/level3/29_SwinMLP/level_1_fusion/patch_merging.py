"""
Component: PatchMerging (Stage 0 -> Stage 1 transition)
Source: data/kernelbench/level3/29_SwinMLP.py
Abstraction Level: fusion
Parent: basic_layer_0
Operations: [view, slice_interleave, cat, view, LayerNorm(384), Linear(384,192,bias=False)]
Input Shapes: [10, 3136, 96]
Output Shapes: [10, 784, 192]
Description: PatchMerging for stage 0 output. Downsamples spatial resolution by 2x
  and doubles channel dimension. H=56,W=56 -> H=28,W=28, C=96 -> 192.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_resolution = (56, 56)
        self.dim = 96
        self.reduction = nn.Linear(4 * self.dim, 2 * self.dim, bias=False)
        self.norm = nn.LayerNorm(4 * self.dim)

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
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)              # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)
        return x


def get_inputs():
    return [torch.randn(10, 3136, 96)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(10, 784, 192)]


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
