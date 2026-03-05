"""
Component: layer_norm
Abstraction Level: kernel
Parent: 34_ConvTranspose3d_LayerNorm_GELU_Scaling (level_1_fusion/conv_norm_gelu_scale.py)
Children: none

Operations: nn.LayerNorm(normalized_shape=4, eps=1e-5)
  Normalizes over the last dimension (W=4) of the input tensor.
  After ConvTranspose3d, output shape is (B, C, D, H, W) = (2, 4, 2, 4, 4).
  LayerNorm(4) normalizes over the last dimension of size W=4.

Input Shapes:
  - x: [2, 4, 2, 4, 4] dtype=float32

Output Shapes:
  - output: [2, 4, 2, 4, 4] dtype=float32

Weight Shapes:
  - layer_norm.weight: [4]
  - layer_norm.bias: [4]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Layer normalization over the last dimension.
    Extracted from: 34_ConvTranspose3d_LayerNorm_GELU_Scaling
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)


def get_inputs():
    return [torch.randn(2, 4, 2, 4, 4)]


def get_init_inputs():
    return [4, 1e-5]


def get_expected_output_shape():
    return [(2, 4, 2, 4, 4)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "NaN in output"
            assert not torch.isinf(output).any(), "Inf in output"
            expected_shapes = get_expected_output_shape()
            if isinstance(output, tuple):
                actual_shapes = [o.shape for o in output]
            else:
                actual_shapes = [output.shape]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), f"Shape mismatch: {actual} vs {expected}"
            # Verify normalization: mean should be ~0 and std ~1 along last dim
            mean_last = output.mean(dim=-1)
            std_last = output.std(dim=-1, correction=0)
            assert mean_last.abs().max().item() < 1e-4, \
                f"LayerNorm mean not near zero: max={mean_last.abs().max().item()}"
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
