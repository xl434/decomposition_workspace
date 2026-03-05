"""
Level 0 Kernel: Slice Last Timestep

Takes the last timestep from a sequence tensor: out[:, -1, :].
This is a stateless operation with no learnable parameters.

Input:  shape [batch_size=2, seq_len=4, features=32]
Output: shape [batch_size=2, features=32]

Operation: tensor[:, -1, :]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x[:, -1, :]


def get_inputs():
    """Return sample inputs: [x]."""
    batch_size = 2
    seq_len = 4
    features = 32
    x = torch.randn(batch_size, seq_len, features)
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
    # Verify it actually takes the last timestep
    x = inputs[0]
    expected = x[:, -1, :]
    assert torch.equal(output, expected), "Slice operation did not select last timestep correctly"
    print(f"[PASS] slice_last kernel: output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
