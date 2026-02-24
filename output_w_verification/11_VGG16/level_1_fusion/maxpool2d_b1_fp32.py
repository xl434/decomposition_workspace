"""
Component: MaxPool2d (Block 1 pooling)
Source: data/kernelbench/level3/11_VGG16.py
Abstraction Level: fusion
Parent: block1_fp32

Operations: [MaxPool2d(kernel_size=2, stride=2)]

Input Shapes:
  - x: [10, 64, 224, 224] dtype=float32

Output Shapes:
  - output: [10, 64, 112, 112] dtype=float32

Weight Shapes:
  - (none)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """MaxPool2d(kernel_size=2, stride=2). Extracted from: block1_fp32"""

    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.maxpool(x)


batch_size = 10


def get_inputs():
    return [torch.randn(batch_size, 64, 224, 224)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(batch_size, 64, 112, 112)]


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
