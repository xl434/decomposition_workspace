"""
Level 2 Layer: Dense Layer 0.

BN(4) -> ReLU -> Conv2d(4, 4, 3, padding=1) -> Dropout(0.0).
This is the first layer in the DenseBlock. It processes the original input
and produces a new feature map.

Input:  [2, 4, 8, 8]   (original input features)
Output: [2, 4, 8, 8]   (new feature, before concatenation)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Dense layer 0: BN(4)->ReLU->Conv(4,4,3,p=1)->Dropout(0.0)."""

    def __init__(self, in_features: int = 4, growth_rate: int = 4):
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

    # Verify layer structure
    assert len(model.layer) == 4, "Layer should have 4 components: BN, ReLU, Conv, Dropout"
    assert isinstance(model.layer[0], nn.BatchNorm2d), "First should be BatchNorm2d"
    assert isinstance(model.layer[1], nn.ReLU), "Second should be ReLU"
    assert isinstance(model.layer[2], nn.Conv2d), "Third should be Conv2d"
    assert isinstance(model.layer[3], nn.Dropout), "Fourth should be Dropout"

    print("dense_layer_0: All tests passed.")


if __name__ == "__main__":
    run_tests()
