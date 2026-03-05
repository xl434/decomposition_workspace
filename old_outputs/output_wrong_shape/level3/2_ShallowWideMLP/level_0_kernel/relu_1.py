"""
Level 0 Kernel: ReLU Activation 1 (second activation in MLP)

Second ReLU activation in the ShallowWideMLP pipeline.
Stateless operation with no learnable parameters.

Input:  shape [batch_size=2, features=32]
Output: shape [batch_size=2, features=32]

Operation: torch.nn.ReLU()
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)


def get_inputs():
    """Return sample inputs: [x]."""
    batch_size = 2
    features = 32
    x = torch.randn(batch_size, features)
    return [x]


def get_init_inputs():
    """Return constructor arguments (none needed)."""
    return []


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
    # Verify ReLU behavior: all outputs >= 0
    assert (output >= 0).all(), "ReLU output contains negative values"
    # Verify against manual computation
    expected = torch.clamp(inputs[0], min=0)
    assert torch.equal(output, expected), "ReLU output does not match expected"
    print(f"[PASS] relu_1 kernel: output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
