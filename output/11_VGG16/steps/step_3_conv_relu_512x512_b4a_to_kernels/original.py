"""
Component: Conv2d(512,512,3,padding=1) + ReLU (Block 4, Layer 2)
Source: data/kernelbench/level3/11_VGG16.py
Abstraction Level: fusion
Parent: block4_fp32

Operations: [Conv2d(512,512,kernel_size=3,padding=1), ReLU(inplace=True)]

Input Shapes:
  - x: [10, 512, 28, 28] dtype=float32

Output Shapes:
  - output: [10, 512, 28, 28] dtype=float32

Weight Shapes:
  - conv.weight: [512, 512, 3, 3]
  - conv.bias: [512]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Conv2d(512,512,3,padding=1) + ReLU fusion (block4, second). Extracted from: block4_fp32"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


batch_size = 10


def get_inputs():
    return [torch.randn(batch_size, 512, 28, 28)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(batch_size, 512, 28, 28)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"
            expected_shapes = get_expected_output_shape()
            actual_shapes = [output.shape] if isinstance(output, torch.Tensor) else [o.shape for o in output]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), \
                    f"Output {i} shape mismatch: got {actual}, expected {expected}"
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
