"""
Level 1 Fusion: Linear -> ReLU -> Linear -> ReLU -> Linear (ShallowWideMLP)

Fuses five Level 0 kernels into the complete 2_ShallowWideMLP model:
  1. Linear(16, 32):  [2,16] -> [2,32]
  2. ReLU:            [2,32] -> [2,32]
  3. Linear(32, 32):  [2,32] -> [2,32]
  4. ReLU:            [2,32] -> [2,32]
  5. Linear(32, 8):   [2,32] -> [2,8]

Input:  x of shape [batch_size=2, input_size=16]
Output: shape [batch_size=2, output_size=8]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(Model, self).__init__()
        layers = []
        current_input_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size
        layers.append(nn.Linear(current_input_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def get_inputs():
    """Return sample inputs: [x]."""
    batch_size = 2
    input_size = 16
    x = torch.randn(batch_size, input_size)
    return [x]


def get_init_inputs():
    """Return constructor arguments: (input_size, hidden_layer_sizes, output_size)."""
    return [16, [32, 32], 8]


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 8)


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
    print(f"[PASS] mlp_pipeline fusion: output shape {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
