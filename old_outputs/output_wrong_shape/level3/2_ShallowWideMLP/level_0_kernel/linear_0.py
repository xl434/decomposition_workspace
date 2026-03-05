"""
Level 0 Kernel: Linear Layer 0 (first layer of MLP)

First fully connected layer in the ShallowWideMLP pipeline.

Input:  shape [batch_size=2, in_features=16]
Output: shape [batch_size=2, out_features=32]

Operation: nn.Linear(16, 32)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


def get_inputs():
    """Return sample inputs: [x]."""
    batch_size = 2
    in_features = 16
    x = torch.randn(batch_size, in_features)
    return [x]


def get_init_inputs():
    """Return constructor arguments: (in_features, out_features)."""
    return [16, 32]


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 32)


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
    print(f"[PASS] linear_0 kernel: output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
