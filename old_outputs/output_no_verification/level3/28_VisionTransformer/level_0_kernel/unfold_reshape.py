"""
Component: unfold_reshape
Source: data/kernelbench/level3/28_VisionTransformer.py
Abstraction Level: kernel
Parent: patch_extract_embed
Operations: [img.unfold(2, 16, 16).unfold(3, 16, 16).reshape(batch, -1, patch_dim)]
Input Shapes: [2, 3, 224, 224]
Output Shapes: [2, 196, 768]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, patch_size=16, channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels

    def forward(self, img):
        p = self.patch_size
        # unfold along H dimension, then W dimension, then reshape to (batch, num_patches, patch_dim)
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p * p * img.shape[1])
        return x


def get_inputs():
    return [torch.randn(2, 3, 224, 224)]


def get_init_inputs():
    return [16, 3]


def get_expected_output_shape():
    return [(2, 196, 768)]


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
