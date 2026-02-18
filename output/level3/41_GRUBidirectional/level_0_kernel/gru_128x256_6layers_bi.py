"""
Component: Bidirectional GRU Kernel (128->256, 6 layers)
Source: data/kernelbench/level3/41_GRUBidirectional.py
Abstraction Level: kernel
Parent: bigru_fusion
Operations: [nn.GRU with input_size=128, hidden_size=256, num_layers=6, bias=True, batch_first=False, bidirectional=True]
Input Shapes: x=[512, 10, 128], h0=[12, 10, 256]
Output Shapes: output=[512, 10, 512]

The fundamental bidirectional GRU recurrent operation. This kernel processes a sequence
of length 512 through 6 stacked bidirectional GRU layers. Each layer has a forward and
backward pass, so hidden states have num_layers*2=12 entries. The output concatenates
forward and backward hidden states, producing hidden_size*2=512 features per timestep.

GRU gates per direction per layer:
  - Reset gate:  r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
  - Update gate: z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
  - New gate:    n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))
  - Hidden:      h_t = (1 - z_t) * n_t + z_t * h_{t-1}

For bidirectional, the backward direction processes the sequence in reverse order.
Outputs from both directions are concatenated along the feature dimension.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        """
        Bidirectional GRU kernel.

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

            # Verify GRU configuration
            assert model.gru.input_size == 128, f"Wrong input_size: {model.gru.input_size}"
            assert model.gru.hidden_size == 256, f"Wrong hidden_size: {model.gru.hidden_size}"
            assert model.gru.num_layers == 6, f"Wrong num_layers: {model.gru.num_layers}"
            assert model.gru.bidirectional is True, "GRU should be bidirectional"
            assert model.gru.bias is True, "GRU should have bias"
            assert model.gru.batch_first is False, "batch_first should be False"

            # Verify output dimension is hidden_size * 2 (bidirectional)
            assert output.shape[-1] == 256 * 2, f"Output last dim should be 512 (256*2), got {output.shape[-1]}"

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
