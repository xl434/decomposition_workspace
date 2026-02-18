"""
Component: GRU Fusion
Source: data/kernelbench/level3/39_GRU.py
Abstraction Level: fusion
Parent: gru_model (level_2_layer)
Children: gru_128x256_6layers (level_0_kernel)
Operations: [nn.GRU(input_size=128, hidden_size=256, num_layers=6, bias=True, batch_first=False, bidirectional=False)]
Input Shapes: x=[512, 10, 128], h0=[6, 10, 256]
Output Shapes: output=[512, 10, 256]

The GRU fusion represents nn.GRU as a fused operation. It wraps the GRU kernel and
exposes the complete GRU computation (all 6 layers, all 512 time steps) as a single
fused unit. Internally, nn.GRU fuses the following per-timestep gate computations:
  - r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)  (reset gate)
  - z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)  (update gate)
  - n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))  (new gate)
  - h_t = (1 - z_t) * n_t + z_t * h_{t-1}
These are repeated across 6 stacked layers and 512 time steps.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        """
        GRU fusion: nn.GRU(input_size=128, hidden_size=256, num_layers=6,
                           bias=True, batch_first=False, dropout=0, bidirectional=False)

        This fusion-level component wraps the GRU kernel and returns only the
        sequence output (not the final hidden state), matching the parent layer's
        expected interface.

        Args:
            input_size: Number of expected features in the input (128)
            hidden_size: Number of features in the hidden state (256)
            num_layers: Number of recurrent layers (6)
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=False
        )

    def forward(self, x, h0):
        """
        Execute the GRU fusion.

        Runs the fused GRU operation and returns only the output sequence,
        discarding the final hidden state (matching the parent model's behavior).

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
