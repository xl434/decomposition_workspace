"""
Level 2 Layer: Dense Layer 1.

BN(8) -> ReLU -> Conv2d(8, 4, 3, padding=1) -> Dropout(0.0).
This is the second layer in the DenseBlock. It processes the concatenated
features from the original input and layer 0 output.

Input:  [2, 8, 8, 8]   (concatenated: original 4ch + layer 0 output 4ch)
Output: [2, 4, 8, 8]   (new feature, before concatenation)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Dense layer 1: BN(8)->ReLU->Conv(8,4,3,p=1)->Dropout(0.0)."""

    def __init__(self, in_features: int = 8, growth_rate: int = 4):
        super(Model, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def get_inputs():
    """Return a list of input tensors for the model."""
    batch_size = 2
    in_features = 8
    height = 8
    width = 8
    return [torch.randn(batch_size, in_features, height, width)]


def get_init_inputs():
    """Return a list of arguments to initialize the model."""
    in_features = 8
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

    # Check channel reduction: 8 -> 4
    assert output.shape[1] == 4, "Output channels should equal growth_rate=4"

    # Verify layer structure
    assert len(model.layer) == 4, "Layer should have 4 components: BN, ReLU, Conv, Dropout"
    assert isinstance(model.layer[0], nn.BatchNorm2d), "First should be BatchNorm2d"
    assert model.layer[0].num_features == 8, "BN should have 8 features"
    assert isinstance(model.layer[2], nn.Conv2d), "Third should be Conv2d"
    assert model.layer[2].in_channels == 8, "Conv should have 8 in_channels"
    assert model.layer[2].out_channels == 4, "Conv should have 4 out_channels"

    print("dense_layer_1: All tests passed.")


if __name__ == "__main__":
    run_tests()
