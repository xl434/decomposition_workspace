"""
Component: ReLU_32768
Source: data/kernelbench/level3/2_ShallowWideMLP.py
Abstraction Level: kernel
Parent: LinearReLU_16384x32768 (level_1_fusion/linear_relu_16384x32768.py),
        LinearReLU_32768x32768 (level_1_fusion/linear_relu_32768x32768.py)

Operations: [ReLU]
Input Shapes: - x: [128, 32768] dtype=float32
Output Shapes: - output: [128, 32768] dtype=float32

Standalone ReLU kernel: element-wise max(0, x) activation applied to
tensors of shape [128, 32768]. Used after each hidden Linear layer.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)


batch_size = 128
features = 32768


def get_inputs():
    return [torch.randn(batch_size, features)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(batch_size, features)]


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
            # Additional ReLU-specific tests
            # Test that all outputs are non-negative
            assert (output >= 0).all(), "ReLU output contains negative values"
            # Test that positive inputs pass through unchanged
            pos_input = torch.abs(torch.randn(batch_size, features)) + 1e-6
            pos_output = model(pos_input)
            assert torch.allclose(pos_input, pos_output), "ReLU should pass positive values unchanged"
            # Test that negative inputs become zero
            neg_input = -torch.abs(torch.randn(batch_size, features)) - 1e-6
            neg_output = model(neg_input)
            assert (neg_output == 0).all(), "ReLU should zero out negative values"
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
