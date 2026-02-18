"""
Component: Bidirectional GRU Model (Complete)
Source: data/kernelbench/level3/41_GRUBidirectional.py
Abstraction Level: layer
Parent: None (top-level)
Operations: [nn.GRU(input_size=128, hidden_size=256, num_layers=6, bias=True, batch_first=False, bidirectional=True)]
Input Shapes: x=[512, 10, 128], h0=[12, 10, 256]
Output Shapes: output=[512, 10, 512]

Top-level model implementing a 6-layer bidirectional GRU. The model takes a sequence
input and initial hidden state, processes through the bidirectional GRU, and returns
the output tensor. The original model stores a self.h0 in __init__ but does not use
it in forward() -- forward() takes h0 as a parameter instead.

Note: bidirectional=True doubles the number of hidden state layers (num_layers*2=12)
and the output feature dimension (hidden_size*2=512).

Children:
  - level_1_fusion/bigru_fusion.py: The fused bidirectional GRU operation
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        """
        Complete bidirectional GRU model.

        Args:
            input_size: Number of input features (128)
            hidden_size: Number of hidden features per direction (256)
            num_layers: Number of stacked GRU layers (6)
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=True,
        )

    def forward(self, x, h0):
        """
        Args:
            x: Input tensor [seq_len=512, batch=10, input_size=128]
            h0: Initial hidden state [num_layers*2=12, batch=10, hidden_size=256]

        Returns:
            output: [seq_len=512, batch=10, hidden_size*2=512]
        """
        output, h_n = self.gru(x, h0)
        return output


def get_inputs():
    return [torch.randn(512, 10, 128), torch.randn(12, 10, 256)]


def get_init_inputs():
    return [128, 256, 6]


def get_expected_output_shape():
    return [(512, 10, 512)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)

            # Basic validity checks
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"

            # Shape checks
            expected_shapes = get_expected_output_shape()
            if isinstance(output, tuple):
                actual_shapes = [o.shape for o in output]
            else:
                actual_shapes = [output.shape]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), f"Shape mismatch at output {i}: {actual} vs {expected}"

            # Verify full model configuration
            assert model.gru.input_size == 128
            assert model.gru.hidden_size == 256
            assert model.gru.num_layers == 6
            assert model.gru.bidirectional is True
            assert model.gru.bias is True
            assert model.gru.batch_first is False

            # Verify that output has correct bidirectional dimensions
            assert output.shape == (512, 10, 512), f"Expected output shape (512, 10, 512), got {output.shape}"

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
