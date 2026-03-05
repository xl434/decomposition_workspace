"""
Level 3 Model: DenseNet121 DenseBlock.

Full dense block with 3 layers and dense connectivity (concatenation).
Each layer produces growth_rate=4 new feature channels, which are concatenated
with all previous features before being fed to the next layer.

Data flow (num_layers=3, num_input_features=4, growth_rate=4):
  Input: [2, 4, 8, 8]
  Layer 0: BN(4)->ReLU->Conv(4,4)->Dropout -> new_feat [2,4,8,8]
           cat -> [2, 8, 8, 8]
  Layer 1: BN(8)->ReLU->Conv(8,4)->Dropout -> new_feat [2,4,8,8]
           cat -> [2, 12, 8, 8]
  Layer 2: BN(12)->ReLU->Conv(12,4)->Dropout -> new_feat [2,4,8,8]
           cat -> [2, 16, 8, 8]
  Output: [2, 16, 8, 8]  (num_input_features + num_layers * growth_rate = 4+3*4 = 16)

Input:  [2, 4, 8, 8]
Output: [2, 16, 8, 8]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """DenseNet121 DenseBlock with dense connectivity."""

    def __init__(self, num_layers: int = 3, num_input_features: int = 4, growth_rate: int = 4):
        super(Model, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x


def get_inputs():
    """Return a list of input tensors for the model."""
    batch_size = 2
    num_input_features = 4
    height = 8
    width = 8
    return [torch.randn(batch_size, num_input_features, height, width)]


def get_init_inputs():
    """Return a list of arguments to initialize the model."""
    num_layers = 3
    num_input_features = 4
    growth_rate = 4
    return [num_layers, num_input_features, growth_rate]


def get_expected_output_shape():
    """Return the expected output shape."""
    # num_input_features + num_layers * growth_rate = 4 + 3 * 4 = 16
    return (2, 16, 8, 8)


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

    # Check output channels = num_input_features + num_layers * growth_rate
    num_layers = 3
    num_input_features = 4
    growth_rate = 4
    expected_channels = num_input_features + num_layers * growth_rate
    assert output.shape[1] == expected_channels, \
        f"Expected {expected_channels} output channels, got {output.shape[1]}"

    # Verify the first num_input_features channels are the original input
    # (Due to dense connectivity, the original input is preserved as the first channels)
    x = inputs[0]
    assert torch.allclose(output[:, :num_input_features, :, :], x), \
        "First channels of output should be the original input"

    # Verify model structure
    assert len(model.layers) == num_layers, \
        f"Expected {num_layers} layers, got {len(model.layers)}"
    for i, layer in enumerate(model.layers):
        in_ch = num_input_features + i * growth_rate
        assert layer[0].num_features == in_ch, \
            f"Layer {i} BN should have {in_ch} features"
        assert layer[2].in_channels == in_ch, \
            f"Layer {i} Conv should have {in_ch} in_channels"
        assert layer[2].out_channels == growth_rate, \
            f"Layer {i} Conv should have {growth_rate} out_channels"

    print("dense_block: All tests passed.")


if __name__ == "__main__":
    run_tests()
