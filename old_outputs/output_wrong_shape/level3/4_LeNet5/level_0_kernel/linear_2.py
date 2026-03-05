"""
Level 0 Kernel: Linear_2
Operation: Linear(in_features=120, out_features=84)
Input: [2, 120] -> Output: [2, 84]

Part of LeNet-5 (4_LeNet5) hierarchical decomposition.
Second fully connected layer.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Linear kernel: 120 -> 84 features."""

    def __init__(self, in_features=120, out_features=84):
        super(Model, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, 120]
        Returns:
            Output tensor of shape [batch_size, 84]
        """
        return self.fc(x)


def get_inputs():
    """Return list of input tensors for this kernel."""
    batch_size = 2
    return [torch.randn(batch_size, 120)]


def get_init_inputs():
    """Return list of constructor arguments."""
    return [120, 84]  # in_features, out_features


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 84)


def run_tests():
    """Validate the kernel produces correct output shapes and runs without error."""
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

    print("linear_2: All tests passed!")
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
