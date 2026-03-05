"""
Level 0 Kernel: Bidirectional LSTM

nn.LSTM with bidirectional=True and batch_first=True.

Input:  x of shape [batch_size=2, seq_len=4, input_size=8]
        h0 of shape [num_layers*2=4, batch_size=2, hidden_size=16]
        c0 of shape [num_layers*2=4, batch_size=2, hidden_size=16]
Output: shape [batch_size=2, seq_len=4, hidden_size*2=32]

Operation: nn.LSTM(input_size=8, hidden_size=16, num_layers=2,
                   batch_first=True, dropout=0, bidirectional=True)

Note: forward() accepts (x, h0, c0) as three separate arguments and
internally packs them as self.lstm(x, (h0, c0)).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0, bidirectional=True
        )

    def forward(self, x, h0, c0):
        output, _ = self.lstm(x, (h0, c0))
        return output


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
    """Return constructor arguments: (input_size, hidden_size, num_layers)."""
    return [8, 16, 2]


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 4, 32)


def run_tests():
    """Verify the kernel produces correct output shapes and runs without error."""
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
    print(f"[PASS] lstm kernel: output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
