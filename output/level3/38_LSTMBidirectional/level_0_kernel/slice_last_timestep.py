"""
Component: Slice Last Timestep Kernel
Source: data/kernelbench/level3/38_LSTMBidirectional.py
Abstraction Level: kernel
Parent: level_1_fusion/slice_linear.py
Operations: [out[:, -1, :] extraction]
Input Shapes: x=[10, 512, 512]
Output Shapes: [10, 512]
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, -1, :]


def get_inputs():
    return [torch.randn(10, 512, 512)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(10, 512)]


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
