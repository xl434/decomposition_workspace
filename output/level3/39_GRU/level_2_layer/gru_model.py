"""
Component: GRU Model (Complete)
Source: data/kernelbench/level3/39_GRU.py
Abstraction Level: layer
Parent: None (top-level)
Children: gru_fusion (level_1_fusion)
Operations: [GRU forward pass, output selection from (output, h_n) tuple]
Input Shapes: x=[512, 10, 128], h0=[6, 10, 256]
Output Shapes: output=[512, 10, 256]

Top-level decomposition of the 39_GRU model. This is the complete model that
takes input sequence x and initial hidden state h0, passes them through a
6-layer GRU with input_size=128 and hidden_size=256 (batch_first=False,
bias=True, bidirectional=False), and returns only the output sequence
(discarding the final hidden state h_n).
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        Complete GRU model matching the original 39_GRU.py specification.

        Args:
            input_size: Number of expected features in the input (128)
            hidden_size: Number of features in the hidden state (256)
            num_layers: Number of recurrent layers (6)
            bias: If True, use bias weights (True)
            batch_first: If True, input/output tensors are (batch, seq, feature) (False)
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout=0,
            bidirectional=False
        )

    def forward(self, x, h0):
        """
        Forward pass of the complete GRU model.

        Args:
            x: Input tensor of shape [seq_len=512, batch_size=10, input_size=128]
            h0: Initial hidden state of shape [num_layers=6, batch_size=10, hidden_size=256]

        Returns:
            output: Output tensor of shape [seq_len=512, batch_size=10, hidden_size=256]
        """
        output, h_n = self.gru(x, h0)
        return output


def get_inputs():
    return [torch.randn(512, 10, 128), torch.randn(6, 10, 256)]


def get_init_inputs():
    return [128, 256, 6]


def get_expected_output_shape():
    return [(512, 10, 256)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any(), "NaN in output"
            assert not torch.isinf(output).any(), "Inf in output"

            expected_shapes = get_expected_output_shape()
            if isinstance(output, tuple):
                actual_shapes = [o.shape for o in output]
            else:
                actual_shapes = [output.shape]

            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), \
                    f"Shape mismatch at index {i}: {actual} vs {expected}"

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
