"""
Level 0 Kernel: Dropout
Applies dropout to the embedding sequence. With dropout=0.0, this is an identity op.
Input: [batch_size, seq_len, dim] = [2, 17, 32]
Output: [batch_size, seq_len, dim] = [2, 17, 32]
No learnable weights.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(x)


def get_inputs():
    return [torch.randn(2, 17, 32)]


def get_init_inputs():
    return [0.0]  # dropout_rate


def get_expected_output_shape():
    return (2, 17, 32)


def run_tests():
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"
    # With dropout=0.0 and eval mode, output should equal input
    assert torch.allclose(output, inputs[0], atol=1e-6), "Dropout with rate 0.0 should be identity"
    print(f"dropout: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
