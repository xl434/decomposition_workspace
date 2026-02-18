"""
Component: Bidirectional LSTM Kernel
Source: data/kernelbench/level3/38_LSTMBidirectional.py
Abstraction Level: kernel
Parent: level_1_fusion/bilstm_fusion.py
Operations: [nn.LSTM(input_size=128, hidden_size=256, num_layers=6, batch_first=True, bidirectional=True)]
Input Shapes: x=[10, 512, 128], h0=[12, 10, 256], c0=[12, 10, 256]
Output Shapes: [10, 512, 512]
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )

    def forward(self, x, h0, c0):
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        return out


def get_inputs():
    return [
        torch.randn(10, 512, 128),
        torch.randn(12, 10, 256),
        torch.randn(12, 10, 256),
    ]


def get_init_inputs():
    return [128, 256, 6, 0.0]


def get_expected_output_shape():
    return [(10, 512, 512)]


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
