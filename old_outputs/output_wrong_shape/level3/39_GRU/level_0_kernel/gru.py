"""
Level 0 Kernel: GRU (Gated Recurrent Unit)

Single nn.GRU operation. This IS the entire model -- no decomposition needed.

Input:  x of shape [seq_len=4, batch_size=2, input_size=8]
        h0 of shape [num_layers=2, batch_size=2, hidden_size=16]
Output: shape [seq_len=4, batch_size=2, hidden_size=16]

Operation: nn.GRU(input_size=8, hidden_size=16, num_layers=2, bias=True,
                  batch_first=False, dropout=0, bidirectional=False)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, bias=True, batch_first=False):
        super(Model, self).__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            bias=bias, batch_first=batch_first,
            dropout=0, bidirectional=False
        )

    def forward(self, x, h0):
        output, h_n = self.gru(x, h0)
        return output


def get_inputs():
    """Return sample inputs for the model: [x, h0]."""
    seq_len = 4
    batch_size = 2
    input_size = 8
    hidden_size = 16
    num_layers = 2
    x = torch.randn(seq_len, batch_size, input_size)
    h0 = torch.randn(num_layers, batch_size, hidden_size)
    return [x, h0]


def get_init_inputs():
    """Return constructor arguments: (input_size, hidden_size, num_layers)."""
    return [8, 16, 2]


def get_expected_output_shape():
    """Return the expected output shape."""
    return (4, 2, 16)


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
    print(f"[PASS] gru kernel: output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
