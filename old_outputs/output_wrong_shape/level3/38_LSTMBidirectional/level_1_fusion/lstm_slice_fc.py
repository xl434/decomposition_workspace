"""
Level 1 Fusion: LSTM -> Slice Last -> FC

Fuses three Level 0 kernels into the complete 38_LSTMBidirectional model:
  1. Bidirectional LSTM: [2,4,8] + h0=[4,2,16] + c0=[4,2,16] -> [2,4,32]
  2. Slice last timestep: [2,4,32] -> [2,32]
  3. Fully connected: [2,32] -> [2,4]

Input:  x of shape [batch_size=2, seq_len=4, input_size=8]
        h0 of shape [num_layers*2=4, batch_size=2, hidden_size=16]
        c0 of shape [num_layers*2=4, batch_size=2, hidden_size=16]
Output: shape [batch_size=2, output_size=4]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, h0, c0):
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # slice last timestep
        out = self.fc(out)
        return out


def get_inputs():
    """Return sample inputs: [x, h0, c0]."""
    batch_size = 2
    seq_len = 4
    input_size = 8
    hidden_size = 16
    num_layers = 2
    num_directions = 2  # bidirectional
    x = torch.randn(batch_size, seq_len, input_size)
    h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
    c0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
    return [x, h0, c0]


def get_init_inputs():
    """Return constructor arguments: (input_size, hidden_size, num_layers, output_size)."""
    return [8, 16, 2, 4]


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 4)


def run_tests():
    """Verify the fused model produces correct output shapes and runs without error."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == expected_shape, (
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    )
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    print(f"[PASS] lstm_slice_fc fusion: output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
