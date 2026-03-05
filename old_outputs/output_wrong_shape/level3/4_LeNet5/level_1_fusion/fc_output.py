"""
Level 1 Fusion: fc_output
Operation: Linear(84, num_classes) - single operation (also L0 kernel)
Input: [2, 84] -> Output: [2, 10]

Part of LeNet-5 (4_LeNet5) hierarchical decomposition.
Final output layer producing class logits. Single operation, so this is
identical to the L0 kernel linear_3.

Composed of Level 0 kernels:
  - linear_3: Linear(84, 10)    [2,84] -> [2,10]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Output linear layer (single-op fusion, same as L0 linear_3)."""

    def __init__(self, in_features=84, out_features=10):
        super(Model, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, 84]
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        return self.fc(x)


def get_inputs():
    """Return list of input tensors for this fusion."""
    batch_size = 2
    return [torch.randn(batch_size, 84)]


def get_init_inputs():
    """Return list of constructor arguments."""
    return [84, 10]  # in_features, out_features (num_classes)


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 10)


def run_tests():
    """Validate the fusion produces correct output shapes and runs without error."""
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)

    expected_shape = get_expected_output_shape()
    assert output.shape == expected_shape, (
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    )

    # Test that gradients flow
    x = inputs[0].clone().requires_grad_(True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should flow back to input"
    assert x.grad.shape == x.shape, "Gradient shape should match input shape"

    # Output can have negative values (no activation)
    print("fc_output: All tests passed!")
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
