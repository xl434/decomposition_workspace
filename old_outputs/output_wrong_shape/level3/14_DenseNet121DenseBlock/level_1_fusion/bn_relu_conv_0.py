"""
Level 1 Fusion: BN(4) -> ReLU -> Conv2d(4, 4, 3, padding=1).

Fused batch normalization, ReLU activation, and convolution for dense layer 0.
This is the core compute pattern in each DenseNet layer (excluding dropout).

Input:  [2, 4, 8, 8]
Output: [2, 4, 8, 8]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """BN(4) -> ReLU -> Conv2d(4, 4, 3, padding=1, bias=False)."""

    def __init__(self, in_features: int = 4, growth_rate: int = 4):
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        return out


def get_inputs():
    """Return a list of input tensors for the model."""
    batch_size = 2
    in_features = 4
    height = 8
    width = 8
    return [torch.randn(batch_size, in_features, height, width)]


def get_init_inputs():
    """Return a list of arguments to initialize the model."""
    in_features = 4
    growth_rate = 4
    return [in_features, growth_rate]


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 4, 8, 8)


def run_tests():
    """Validate the model produces correct output shapes and values."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    output = model(*inputs)

    # Check output shape
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Expected shape {expected_shape}, got {output.shape}"

    # Check output is finite
    assert torch.isfinite(output).all(), "Output contains non-finite values"

    # Check spatial dimensions preserved
    assert output.shape[2] == inputs[0].shape[2], "Height should be preserved"
    assert output.shape[3] == inputs[0].shape[3], "Width should be preserved"

    # Check output channels equals growth_rate
    assert output.shape[1] == 4, "Output channels should equal growth_rate=4"

    # Verify the fusion matches sequential execution
    model2 = Model(*get_init_inputs())
    model2.eval()
    x = torch.randn(2, 4, 8, 8)
    fused_out = model2(x)

    bn_out = model2.bn(x)
    relu_out = model2.relu(bn_out.clone())
    conv_out = model2.conv(relu_out)
    assert torch.allclose(fused_out, conv_out, atol=1e-6), \
        "Fused output should match sequential BN->ReLU->Conv"

    print("bn_relu_conv_0: All tests passed.")


if __name__ == "__main__":
    run_tests()
