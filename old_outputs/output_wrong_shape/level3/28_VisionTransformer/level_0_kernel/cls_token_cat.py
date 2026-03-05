"""
Level 0 Kernel: CLS Token Concatenation
Prepends a learnable CLS token to the sequence of patch embeddings.
Input: [batch_size, num_patches, dim] = [2, 16, 32]
Output: [batch_size, num_patches+1, dim] = [2, 17, 32]
Weights: nn.Parameter cls_token [1, 1, 32]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=32):
        super(Model, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat((cls_tokens, x), dim=1)


def get_inputs():
    return [torch.randn(2, 16, 32)]


def get_init_inputs():
    return [32]  # dim


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
    # Verify first token is the cls_token (repeated for each batch element)
    cls_expanded = model.cls_token.expand(2, -1, -1)
    assert torch.allclose(output[:, :1, :], cls_expanded, atol=1e-6), "CLS token mismatch"
    # Verify remaining tokens are the input
    assert torch.allclose(output[:, 1:, :], inputs[0], atol=1e-6), "Patch tokens mismatch"
    print(f"cls_token_cat: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
