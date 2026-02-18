"""
Component: ShallowWideMLP
Source: data/kernelbench/level3/2_ShallowWideMLP.py
Abstraction Level: model
Parent: None

Operations: [Linear(16384,32768), ReLU, Linear(32768,32768), ReLU, Linear(32768,16384)]
Input Shapes: - x: [128, 16384] dtype=float32
Output Shapes: - output: [128, 16384] dtype=float32

Full shallow wide MLP model with 2 hidden layers of size 32768
and input/output sizes of 16384. Uses ReLU activation between hidden layers.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        layers = []
        current_input_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size
        layers.append(nn.Linear(current_input_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


batch_size = 128
input_size = 16384
hidden_layer_sizes = [32768, 32768]
output_size = 16384


def get_inputs():
    return [torch.randn(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]


def get_expected_output_shape():
    return [(batch_size, output_size)]


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
