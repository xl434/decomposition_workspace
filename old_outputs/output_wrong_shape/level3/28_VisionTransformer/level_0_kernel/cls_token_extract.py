"""
Level 0 Kernel: CLS Token Extraction
Extracts the CLS token (first token) from the sequence.
Input: [batch_size, seq_len, dim] = [2, 17, 32]
Output: [batch_size, dim] = [2, 32]
No learnable weights.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x[:, 0]


def get_inputs():
    return [torch.randn(2, 17, 32)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return (2, 32)


def run_tests():
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"
    # Verify extraction: output should be the first token of input
    assert torch.allclose(output, inputs[0][:, 0], atol=1e-6), "CLS extraction mismatch"
    print(f"cls_token_extract: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
