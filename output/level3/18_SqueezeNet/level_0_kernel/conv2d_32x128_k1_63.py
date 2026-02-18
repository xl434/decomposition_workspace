"""
Component: Conv2d(32, 128, kernel_size=1) - expand1x1 for Fire Module 4 (63x63)
Source: data/kernelbench/level3/18_SqueezeNet.py
Abstraction Level: kernel
Parent: expand_concat_32x128x128_2.py
Operations: [Conv2d]
Input Shapes: [64, 32, 63, 63]
Output Shapes: [64, 128, 63, 63]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(32, 128, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    return [torch.randn(64, 32, 63, 63)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(64, 128, 63, 63)]


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
