"""
Component: MaxPool2d(kernel_size=2, stride=2) on [10, 256, 56, 56]
Source: data/kernelbench/level3/11_VGG16.py
Abstraction Level: kernel
Parent: Features Block 3 (level_2_layer/features_block_3.py)
Operations: [MaxPool2d(kernel_size=2, stride=2)]
Input Shapes: - x: [10, 256, 56, 56] dtype=float32
Output Shapes: - output: [10, 256, 28, 28] dtype=float32
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.maxpool(x)


def get_inputs():
    return [torch.randn(10, 256, 56, 56)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(10, 256, 28, 28)]


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
