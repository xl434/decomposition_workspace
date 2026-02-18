"""
Component: relu_16x10x10
Source: data/kernelbench/level3/4_LeNet5.py
Abstraction Level: kernel
Parent: conv_relu_pool_6x16x5
Operations: [ReLU]
Input Shapes: - x: [4096, 16, 10, 10] dtype=float32
Output Shapes: - output: [4096, 16, 10, 10] dtype=float32
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x)


batch_size = 4096


def get_inputs():
    return [torch.randn(batch_size, 16, 10, 10)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(batch_size, 16, 10, 10)]


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
