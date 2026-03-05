"""
Component: Patch Projection + Norm
Source: data/kernelbench/level3/29_SwinMLP.py
Abstraction Level: fusion
Parent: patch_embed
Operations: [Conv2d(3,96,k=4,s=4), flatten(2), transpose(1,2), LayerNorm(96)]
Input Shapes: [10, 3, 224, 224]
Output Shapes: [10, 3136, 96]
Description: Conv2d projection converts image patches to embeddings, then flattens
  spatial dims and applies LayerNorm. This is the core of PatchEmbed.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(96)

    def forward(self, x):
        x = self.proj(x)           # [10, 96, 56, 56]
        x = x.flatten(2)           # [10, 96, 3136]
        x = x.transpose(1, 2)     # [10, 3136, 96]
        x = self.norm(x)           # [10, 3136, 96]
        return x


def get_inputs():
    return [torch.randn(10, 3, 224, 224)]


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
