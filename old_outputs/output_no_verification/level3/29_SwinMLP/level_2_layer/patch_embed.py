"""
Component: PatchEmbed
Source: data/kernelbench/level3/29_SwinMLP.py
Abstraction Level: layer
Parent: swin_mlp
Operations: [Conv2d(3,96,k=4,s=4), flatten(2), transpose(1,2), LayerNorm(96)]
Input Shapes: [10, 3, 224, 224]
Output Shapes: [10, 3136, 96]
Description: Patch embedding layer. Projects image into patch tokens via convolution,
  then flattens and normalizes. img_size=224, patch_size=4, embed_dim=96.
"""
import torch
import torch.nn as nn
from itertools import repeat
import collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        img_size = to_2tuple(224)
        patch_size = to_2tuple(4)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = 3
        self.embed_dim = 96
        self.proj = nn.Conv2d(3, 96, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(96)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.norm(x)
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
