"""
Component: softmax
Abstraction Level: kernel
Parent: softmax_tanh_scale (level_1_fusion)
Children: none

Operations: torch.softmax(x, dim=1)  -- softmax over the channel dimension

Input Shapes:
  - x: (2, 4, 1, 4, 4) dtype=float32

Output Shapes:
  - output: (2, 4, 1, 4, 4) dtype=float32

Weight Shapes:
  - none
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Softmax activation over the channel dimension (dim=1).
    Extracted from: softmax_tanh_scale fusion (level 1)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.softmax(x, dim=1)


def get_inputs():
    return [torch.randn(2, 4, 1, 4, 4)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(2, 4, 1, 4, 4)]


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
            # Verify softmax properties: values in [0,1] and sum to 1 along dim=1
            assert (output >= 0).all(), "Softmax output has negative values"
            assert (output <= 1).all(), "Softmax output has values > 1"
            channel_sum = output.sum(dim=1)
            assert torch.allclose(channel_sum, torch.ones_like(channel_sum), atol=1e-5), \
                "Softmax output does not sum to 1 along channel dim"
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
