"""
Verification: Composition test for 14_DenseNet121DenseBlock.

Builds the original DenseBlock model and a composed model from decomposed parts.
Shares weights between them and verifies that they produce identical outputs.

The composed model manually executes:
  1. Each layer's BN -> ReLU -> Conv -> Dropout
  2. Concatenates features after each layer
  3. Compares the final output with the original model's output.
"""

import torch
import torch.nn as nn


class OriginalModel(nn.Module):
    """Original DenseNet121 DenseBlock (3 layers, input_features=4, growth_rate=4)."""

    def __init__(self, num_layers: int = 3, num_input_features: int = 4, growth_rate: int = 4):
        super(OriginalModel, self).__init__()
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


class ComposedModel(nn.Module):
    """Composed DenseBlock from decomposed parts.

    Manually builds each layer (BN -> ReLU -> Conv -> Dropout) and
    performs the dense concatenation, matching the original model's logic.
    """

    def __init__(self, num_layers: int = 3, num_input_features: int = 4, growth_rate: int = 4):
        super(ComposedModel, self).__init__()
        # Layer 0: BN(4) -> ReLU -> Conv(4,4) -> Dropout
        self.bn_0 = nn.BatchNorm2d(num_input_features)
        self.relu_0 = nn.ReLU(inplace=True)
        self.conv_0 = nn.Conv2d(num_input_features, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout_0 = nn.Dropout(0.0)

        # Layer 1: BN(8) -> ReLU -> Conv(8,4) -> Dropout
        in_features_1 = num_input_features + growth_rate  # 4 + 4 = 8
        self.bn_1 = nn.BatchNorm2d(in_features_1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv2d(in_features_1, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout_1 = nn.Dropout(0.0)

        # Layer 2: BN(12) -> ReLU -> Conv(12,4) -> Dropout
        in_features_2 = num_input_features + 2 * growth_rate  # 4 + 8 = 12
        self.bn_2 = nn.BatchNorm2d(in_features_2)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(in_features_2, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout_2 = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]

        # Layer 0: process original input
        out = self.bn_0(x)
        out = self.relu_0(out)
        out = self.conv_0(out)
        out = self.dropout_0(out)
        features.append(out)
        x = torch.cat(features, 1)  # [2, 8, 8, 8]

        # Layer 1: process concatenated features
        out = self.bn_1(x)
        out = self.relu_1(out)
        out = self.conv_1(out)
        out = self.dropout_1(out)
        features.append(out)
        x = torch.cat(features, 1)  # [2, 12, 8, 8]

        # Layer 2: process concatenated features
        out = self.bn_2(x)
        out = self.relu_2(out)
        out = self.conv_2(out)
        out = self.dropout_2(out)
        features.append(out)
        x = torch.cat(features, 1)  # [2, 16, 8, 8]

        return x


class Model(nn.Module):
    """Wrapper for composition test. Uses OriginalModel internally."""

    def __init__(self, num_layers: int = 3, num_input_features: int = 4, growth_rate: int = 4):
        super(Model, self).__init__()
        self.model = OriginalModel(num_layers, num_input_features, growth_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


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
    return (2, 16, 8, 8)


def _share_weights(original: OriginalModel, composed: ComposedModel):
    """Copy weights from original model to composed model."""
    # Layer 0: original.layers[0] -> composed.bn_0, conv_0
    composed.bn_0.load_state_dict(original.layers[0][0].state_dict())
    composed.conv_0.load_state_dict(original.layers[0][2].state_dict())

    # Layer 1: original.layers[1] -> composed.bn_1, conv_1
    composed.bn_1.load_state_dict(original.layers[1][0].state_dict())
    composed.conv_1.load_state_dict(original.layers[1][2].state_dict())

    # Layer 2: original.layers[2] -> composed.bn_2, conv_2
    composed.bn_2.load_state_dict(original.layers[2][0].state_dict())
    composed.conv_2.load_state_dict(original.layers[2][2].state_dict())


def run_tests():
    """Validate that composed model matches original model output."""
    torch.manual_seed(42)

    num_layers = 3
    num_input_features = 4
    growth_rate = 4

    # Build both models
    original = OriginalModel(num_layers, num_input_features, growth_rate)
    composed = ComposedModel(num_layers, num_input_features, growth_rate)

    # Share weights from original to composed
    _share_weights(original, composed)

    # Set both to eval mode
    original.eval()
    composed.eval()

    # Create deterministic input
    torch.manual_seed(123)
    x = torch.randn(2, num_input_features, 8, 8)

    # Run both models
    with torch.no_grad():
        original_output = original(x)
        composed_output = composed(x)

    # --- Test 1: Shape match ---
    expected_shape = get_expected_output_shape()
    assert original_output.shape == torch.Size(expected_shape), \
        f"Original shape {original_output.shape} != expected {expected_shape}"
    assert composed_output.shape == torch.Size(expected_shape), \
        f"Composed shape {composed_output.shape} != expected {expected_shape}"
    print("  [PASS] Both models produce correct output shape:", expected_shape)

    # --- Test 2: Exact numerical match ---
    assert torch.allclose(original_output, composed_output, atol=1e-6), \
        f"Outputs differ! Max diff: {(original_output - composed_output).abs().max().item()}"
    print("  [PASS] Composed output matches original output (atol=1e-6)")

    # --- Test 3: Verify dense connectivity (original input preserved) ---
    assert torch.allclose(original_output[:, :num_input_features, :, :], x), \
        "Original input should be preserved as first channels"
    assert torch.allclose(composed_output[:, :num_input_features, :, :], x), \
        "Composed: original input should be preserved as first channels"
    print("  [PASS] Dense connectivity verified: original input preserved in output")

    # --- Test 4: Intermediate shape verification ---
    # Manually check intermediate shapes by running layer by layer
    original.eval()
    features = [x]
    curr = x
    for i, layer in enumerate(original.layers):
        new_feat = layer(curr)
        expected_feat_shape = (2, growth_rate, 8, 8)
        assert new_feat.shape == torch.Size(expected_feat_shape), \
            f"Layer {i} output shape {new_feat.shape} != expected {expected_feat_shape}"
        features.append(new_feat)
        curr = torch.cat(features, 1)
        expected_cat_channels = num_input_features + (i + 1) * growth_rate
        assert curr.shape[1] == expected_cat_channels, \
            f"After layer {i} cat: {curr.shape[1]} channels != expected {expected_cat_channels}"
    print("  [PASS] Intermediate shapes verified for all 3 layers")

    # --- Test 5: Multiple random inputs ---
    for trial in range(5):
        torch.manual_seed(trial * 100 + 7)
        x_trial = torch.randn(2, num_input_features, 8, 8)
        with torch.no_grad():
            out_orig = original(x_trial)
            out_comp = composed(x_trial)
        assert torch.allclose(out_orig, out_comp, atol=1e-6), \
            f"Trial {trial}: outputs differ! Max diff: {(out_orig - out_comp).abs().max().item()}"
    print("  [PASS] Outputs match across 5 random inputs")

    # --- Test 6: Output is finite ---
    assert torch.isfinite(original_output).all(), "Original output has non-finite values"
    assert torch.isfinite(composed_output).all(), "Composed output has non-finite values"
    print("  [PASS] All outputs are finite")

    print("\ncomposition_test: All tests passed.")


if __name__ == "__main__":
    run_tests()
