"""
Component: Linear_32768x16384
Source: data/kernelbench/level3/2_ShallowWideMLP.py
Abstraction Level: kernel
Parent: OutputLayer (level_2_layer/output_layer.py)

Operations: [Linear(32768, 16384)]
Input Shapes: - x: [128, 32768] dtype=float32
Output Shapes: - output: [128, 16384] dtype=float32

Standalone Linear kernel: fully-connected layer projecting from
in_features=32768 to out_features=16384. Computes x @ W^T + b.
This is the final linear operation in the network (output projection).
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


batch_size = 128
in_features = 32768
out_features = 16384


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]


def get_expected_output_shape():
    return [(batch_size, out_features)]


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
