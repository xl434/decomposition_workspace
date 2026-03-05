"""
Level 1 Fusion: fc_relu_2
Fused operations: Linear(120,84) -> ReLU
Input: [2, 120] -> Output: [2, 84]

Part of LeNet-5 (4_LeNet5) hierarchical decomposition.
Second fully connected block with activation.

Composed of Level 0 kernels:
  - linear_2: Linear(120, 84)   [2,120] -> [2,84]
  - relu_4:   ReLU              [2,84]  -> [2,84]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Fused Linear + ReLU block."""

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
        x = self.fc(x)
        x = F.relu(x)
        return x


def get_inputs():
    """Return list of input tensors for this fusion."""
    batch_size = 2
    return [torch.randn(batch_size, 120)]


def get_init_inputs():
    """Return list of constructor arguments."""
    return [120, 84]  # in_features, out_features


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 84)


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
        x_test = torch.randn(2, 120)
        fused_out = model(x_test)

        # Manual sequential
        h = model.fc(x_test)
        h = F.relu(h)

        assert torch.allclose(fused_out, h), (
            "Fused output should match sequential kernel application"
        )

    print("fc_relu_2: All tests passed!")
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    return True


if __name__ == "__main__":
    run_tests()
