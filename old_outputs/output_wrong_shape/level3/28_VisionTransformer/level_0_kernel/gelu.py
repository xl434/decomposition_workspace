"""
Level 0 Kernel: GELU Activation
Applies Gaussian Error Linear Unit activation.
Input: [batch_size, seq_len, mlp_dim] = [2, 17, 64]
Output: [batch_size, seq_len, mlp_dim] = [2, 17, 64]
No learnable weights.

Note: nn.TransformerEncoderLayer uses F.relu by default (activation='relu'),
but this kernel provides GELU for standalone use. The transformer_layer fusion
and transformer_encoder layer use the default ReLU activation as in the original model.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x)


def get_inputs():
    return [torch.randn(2, 17, 64)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return (2, 17, 64)


def run_tests():
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"
    # Verify GELU: positive inputs should be slightly less than input, negative inputs should be near zero
    test_in = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    test_out = model(test_in)
    assert test_out[2].item() == 0.0, "GELU(0) should be 0"
    assert test_out[3].item() > 0.0, "GELU(1) should be positive"
    assert abs(test_out[0].item()) < 0.1, "GELU(-2) should be near 0"
    print(f"gelu: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
