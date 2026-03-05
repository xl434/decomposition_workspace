"""
Composition test for VGG16 hierarchical decomposition.

Builds the full original VGG16 (test-sized) and a composed model from
individual kernels chained in order. Shares weights between them and
verifies that outputs match exactly.
"""

import torch
import torch.nn as nn


class OriginalVGG16(nn.Module):
    """Original VGG16 model (test-sized)."""

    def __init__(self, num_classes=10):
        super(OriginalVGG16, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 1 * 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ComposedVGG16(nn.Module):
    """
    VGG16 composed from individual kernel operations.

    Each operation is a separate named module to mirror the kernel decomposition.
    """

    def __init__(self, num_classes=10):
        super(ComposedVGG16, self).__init__()
        # Block 1 kernels
        self.conv2d_1_1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu_1_1 = nn.ReLU(inplace=False)
        self.conv2d_1_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.relu_1_2 = nn.ReLU(inplace=False)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2 kernels
        self.conv2d_2_1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu_2_1 = nn.ReLU(inplace=False)
        self.conv2d_2_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.relu_2_2 = nn.ReLU(inplace=False)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3 kernels
        self.conv2d_3_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu_3_1 = nn.ReLU(inplace=False)
        self.conv2d_3_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu_3_2 = nn.ReLU(inplace=False)
        self.conv2d_3_3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu_3_3 = nn.ReLU(inplace=False)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4 kernels
        self.conv2d_4_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu_4_1 = nn.ReLU(inplace=False)
        self.conv2d_4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu_4_2 = nn.ReLU(inplace=False)
        self.conv2d_4_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu_4_3 = nn.ReLU(inplace=False)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5 kernels
        self.conv2d_5_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu_5_1 = nn.ReLU(inplace=False)
        self.conv2d_5_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu_5_2 = nn.ReLU(inplace=False)
        self.conv2d_5_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu_5_3 = nn.ReLU(inplace=False)
        self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier kernels
        self.linear_1 = nn.Linear(64, 256)
        self.relu_fc_1 = nn.ReLU(inplace=False)
        self.dropout_1 = nn.Dropout(p=0.0)
        self.linear_2 = nn.Linear(256, 256)
        self.relu_fc_2 = nn.ReLU(inplace=False)
        self.dropout_2 = nn.Dropout(p=0.0)
        self.linear_3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = self.relu_1_1(self.conv2d_1_1(x))
        x = self.relu_1_2(self.conv2d_1_2(x))
        x = self.pool_1(x)
        # Block 2
        x = self.relu_2_1(self.conv2d_2_1(x))
        x = self.relu_2_2(self.conv2d_2_2(x))
        x = self.pool_2(x)
        # Block 3
        x = self.relu_3_1(self.conv2d_3_1(x))
        x = self.relu_3_2(self.conv2d_3_2(x))
        x = self.relu_3_3(self.conv2d_3_3(x))
        x = self.pool_3(x)
        # Block 4
        x = self.relu_4_1(self.conv2d_4_1(x))
        x = self.relu_4_2(self.conv2d_4_2(x))
        x = self.relu_4_3(self.conv2d_4_3(x))
        x = self.pool_4(x)
        # Block 5
        x = self.relu_5_1(self.conv2d_5_1(x))
        x = self.relu_5_2(self.conv2d_5_2(x))
        x = self.relu_5_3(self.conv2d_5_3(x))
        x = self.pool_5(x)
        # Flatten + Classifier
        x = torch.flatten(x, 1)
        x = self.dropout_1(self.relu_fc_1(self.linear_1(x)))
        x = self.dropout_2(self.relu_fc_2(self.linear_2(x)))
        x = self.linear_3(x)
        return x


def share_weights(original: OriginalVGG16, composed: ComposedVGG16):
    """Copy weights from the original model to the composed model."""
    # Features: conv layers are at indices 0,2,5,7,10,12,14,17,19,21,24,26,28
    # in the Sequential. Map them to composed conv layers.
    conv_mapping = [
        (0, "conv2d_1_1"),
        (2, "conv2d_1_2"),
        (5, "conv2d_2_1"),
        (7, "conv2d_2_2"),
        (10, "conv2d_3_1"),
        (12, "conv2d_3_2"),
        (14, "conv2d_3_3"),
        (17, "conv2d_4_1"),
        (19, "conv2d_4_2"),
        (21, "conv2d_4_3"),
        (24, "conv2d_5_1"),
        (26, "conv2d_5_2"),
        (28, "conv2d_5_3"),
    ]

    for seq_idx, attr_name in conv_mapping:
        orig_conv = original.features[seq_idx]
        comp_conv = getattr(composed, attr_name)
        comp_conv.weight.data.copy_(orig_conv.weight.data)
        comp_conv.bias.data.copy_(orig_conv.bias.data)

    # Classifier: linear layers are at indices 0, 3, 6
    linear_mapping = [
        (0, "linear_1"),
        (3, "linear_2"),
        (6, "linear_3"),
    ]

    for seq_idx, attr_name in linear_mapping:
        orig_linear = original.classifier[seq_idx]
        comp_linear = getattr(composed, attr_name)
        comp_linear.weight.data.copy_(orig_linear.weight.data)
        comp_linear.bias.data.copy_(orig_linear.bias.data)


class Model(nn.Module):
    """Wrapper for composition test conforming to the standard interface."""

    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.original = OriginalVGG16(num_classes)
        self.composed = ComposedVGG16(num_classes)
        share_weights(self.original, self.composed)

    def forward(self, x):
        return self.original(x)


def get_inputs():
    batch_size = 2
    return [torch.randn(batch_size, 3, 32, 32)]


def get_init_inputs():
    return [10]


def get_expected_output_shape():
    return (2, 10)


def run_tests():
    """Run composition tests comparing original vs decomposed model."""
    print("=" * 60)
    print("Composition Test: VGG16 Original vs Decomposed")
    print("=" * 60)

    num_classes = 10
    batch_size = 2

    # Build both models
    original = OriginalVGG16(num_classes)
    composed = ComposedVGG16(num_classes)

    # Share weights
    share_weights(original, composed)

    # Set both to eval mode
    original.eval()
    composed.eval()

    # Create test input
    torch.manual_seed(42)
    x = torch.randn(batch_size, 3, 32, 32)

    # Forward pass
    with torch.no_grad():
        out_original = original(x)
        out_composed = composed(x)

    # Test 1: Shape check
    print("\nTest 1: Output shape check")
    expected_shape = (batch_size, num_classes)
    assert out_original.shape == expected_shape, (
        f"Original shape mismatch: {out_original.shape} vs {expected_shape}"
    )
    assert out_composed.shape == expected_shape, (
        f"Composed shape mismatch: {out_composed.shape} vs {expected_shape}"
    )
    print(f"  Original output shape: {out_original.shape}")
    print(f"  Composed output shape: {out_composed.shape}")
    print("  PASSED")

    # Test 2: Numerical equivalence
    print("\nTest 2: Numerical equivalence")
    max_diff = (out_original - out_composed).abs().max().item()
    print(f"  Max absolute difference: {max_diff:.2e}")
    assert torch.allclose(out_original, out_composed, atol=1e-5), (
        f"Outputs differ! Max diff: {max_diff}"
    )
    print("  PASSED")

    # Test 3: Intermediate shape verification
    print("\nTest 3: Intermediate shape verification (data flow)")
    with torch.no_grad():
        # Trace through original features
        feat = x
        shape_checkpoints = []
        for i, layer in enumerate(original.features):
            feat = layer(feat)
            if isinstance(layer, nn.MaxPool2d):
                shape_checkpoints.append(feat.shape)

        expected_shapes = [
            (2, 8, 16, 16),   # After pool 1
            (2, 16, 8, 8),    # After pool 2
            (2, 32, 4, 4),    # After pool 3
            (2, 64, 2, 2),    # After pool 4
            (2, 64, 1, 1),    # After pool 5
        ]

        for i, (actual, expected) in enumerate(zip(shape_checkpoints, expected_shapes)):
            assert actual == expected, (
                f"Block {i+1} pool output: got {actual}, expected {expected}"
            )
            print(f"  After Block {i+1} pool: {actual}")

        # Flatten
        flat = torch.flatten(feat, 1)
        assert flat.shape == (2, 64), f"Flatten shape: got {flat.shape}, expected (2, 64)"
        print(f"  After flatten: {flat.shape}")

        # Classifier
        clf_out = original.classifier(flat)
        assert clf_out.shape == (2, 10), f"Classifier output: got {clf_out.shape}"
        print(f"  After classifier: {clf_out.shape}")
    print("  PASSED")

    # Test 4: Weight sharing verification
    print("\nTest 4: Weight sharing verification")
    conv_pairs = [
        (original.features[0], composed.conv2d_1_1, "conv2d_1_1"),
        (original.features[2], composed.conv2d_1_2, "conv2d_1_2"),
        (original.features[5], composed.conv2d_2_1, "conv2d_2_1"),
        (original.features[7], composed.conv2d_2_2, "conv2d_2_2"),
        (original.features[10], composed.conv2d_3_1, "conv2d_3_1"),
        (original.features[12], composed.conv2d_3_2, "conv2d_3_2"),
        (original.features[14], composed.conv2d_3_3, "conv2d_3_3"),
        (original.features[17], composed.conv2d_4_1, "conv2d_4_1"),
        (original.features[19], composed.conv2d_4_2, "conv2d_4_2"),
        (original.features[21], composed.conv2d_4_3, "conv2d_4_3"),
        (original.features[24], composed.conv2d_5_1, "conv2d_5_1"),
        (original.features[26], composed.conv2d_5_2, "conv2d_5_2"),
        (original.features[28], composed.conv2d_5_3, "conv2d_5_3"),
    ]
    for orig_layer, comp_layer, name in conv_pairs:
        assert torch.equal(orig_layer.weight.data, comp_layer.weight.data), (
            f"Weight mismatch in {name}"
        )
        assert torch.equal(orig_layer.bias.data, comp_layer.bias.data), (
            f"Bias mismatch in {name}"
        )
    linear_pairs = [
        (original.classifier[0], composed.linear_1, "linear_1"),
        (original.classifier[3], composed.linear_2, "linear_2"),
        (original.classifier[6], composed.linear_3, "linear_3"),
    ]
    for orig_layer, comp_layer, name in linear_pairs:
        assert torch.equal(orig_layer.weight.data, comp_layer.weight.data), (
            f"Weight mismatch in {name}"
        )
        assert torch.equal(orig_layer.bias.data, comp_layer.bias.data), (
            f"Bias mismatch in {name}"
        )
    print(f"  Verified {len(conv_pairs)} conv layer pairs")
    print(f"  Verified {len(linear_pairs)} linear layer pairs")
    print("  PASSED")

    # Test 5: Multiple random inputs
    print("\nTest 5: Multiple random inputs")
    for trial in range(5):
        torch.manual_seed(trial * 100)
        test_x = torch.randn(batch_size, 3, 32, 32)
        with torch.no_grad():
            o1 = original(test_x)
            o2 = composed(test_x)
        diff = (o1 - o2).abs().max().item()
        assert torch.allclose(o1, o2, atol=1e-5), f"Trial {trial}: max diff {diff}"
    print(f"  All 5 random input trials passed")
    print("  PASSED")

    print("\n" + "=" * 60)
    print("ALL COMPOSITION TESTS PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_tests()
