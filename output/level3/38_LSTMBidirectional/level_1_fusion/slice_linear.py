"""
Component: Slice Last Timestep + Linear Fusion
Source: data/kernelbench/level3/38_LSTMBidirectional.py
Abstraction Level: fusion
Parent: level_2_layer/output_layer.py
Operations: [SliceLastTimestep, Linear(512, 10)]
Input Shapes: x=[10, 512, 512]
Output Shapes: [10, 10]
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, hidden_size_times_2, output_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size_times_2, output_size)

    def forward(self, x):
        out = x[:, -1, :]
        out = self.fc(out)
        return out


def get_inputs():
    return [torch.randn(10, 512, 512)]


def get_init_inputs():
    return [512, 10]


def get_expected_output_shape():
    return [(10, 10)]


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
