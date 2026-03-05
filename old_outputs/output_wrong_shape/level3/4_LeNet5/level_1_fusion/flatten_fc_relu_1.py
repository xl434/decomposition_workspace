"""
Level 1 Fusion: flatten_fc_relu_1
Fused operations: Flatten -> Linear(400,120) -> ReLU
Input: [2, 16, 5, 5] -> Output: [2, 120]

Part of LeNet-5 (4_LeNet5) hierarchical decomposition.
Transition from convolutional to fully connected layers with activation.

Composed of Level 0 kernels:
  - flatten:  View/Reshape       [2,16,5,5] -> [2,400]
  - linear_1: Linear(400, 120)  [2,400]    -> [2,120]
  - relu_3:   ReLU              [2,120]    -> [2,120]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Fused Flatten + Linear + ReLU block."""

    def __init__(self, in_features=400, out_features=120):
        super(Model, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, 16, 5, 5]
        Returns:
            Output tensor of shape [batch_size, 120]
        """
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.relu(x)
        return x


def get_inputs():
    """Return list of input tensors for this fusion."""
    batch_size = 2
    return [torch.randn(batch_size, 16, 5, 5)]


def get_init_inputs():
    """Return list of constructor arguments."""
    return [400, 120]  # in_features (16*5*5), out_features


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 120)


def run_tests():
    """Validate the fusion produces correct output shapes and behavior."""
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)

    expected_shape = get_expected_output_shape()
    assert output.shape == expected_shape, (
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    )

    # Test ReLU behavior: output should be non-negative
    assert (output >= 0).all(), "Output should be non-negative after ReLU"

    # Test that gradients flow
    x = inputs[0].clone().requires_grad_(True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should flow back to input"

    # Test equivalence with sequential application of L0 kernels
    model.eval()
    with torch.no_grad():
        x_test = torch.randn(2, 16, 5, 5)
        fused_out = model(x_test)

        # Manual sequential
        h = x_test.view(x_test.size(0), -1)
        h = model.fc(h)
        h = F.relu(h)

        assert torch.allclose(fused_out, h), (
            "Fused output should match sequential kernel application"
        )

    print("flatten_fc_relu_1: All tests passed!")
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
