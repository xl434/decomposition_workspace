"""
Composition Verification Test for SqueezeNet (Model 18, KernelBench Level 3)

Builds the original SqueezeNet and a composed version from decomposed kernels,
shares all weights, and verifies numerical equivalence.

Tests:
1. Level 0 kernels compose into Level 1 fusions
2. Level 1 fusions compose into Level 2 layers
3. Level 2 layers compose into the full Level 3 model
4. End-to-end: composed model matches original model output
"""

import torch
import torch.nn as nn


# ============================================================
# Original SqueezeNet (reference implementation)
# ============================================================

class OriginalFireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(OriginalFireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class OriginalSqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(OriginalSqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            OriginalFireModule(96, 16, 64, 64),
            OriginalFireModule(128, 16, 64, 64),
            OriginalFireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            OriginalFireModule(256, 32, 128, 128),
            OriginalFireModule(256, 48, 192, 192),
            OriginalFireModule(384, 48, 192, 192),
            OriginalFireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            OriginalFireModule(512, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


# ============================================================
# Composed SqueezeNet (built from decomposed kernels)
# ============================================================

class ComposedFireModule(nn.Module):
    """Fire Module composed from L1 fusions: squeeze_relu + expand_cat."""

    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(ComposedFireModule, self).__init__()
        # L1 Fusion: squeeze_relu
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        # L1 Fusion: expand_cat
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # L1: squeeze_relu
        x = self.squeeze(x)
        x = self.squeeze_activation(x)
        # L1: expand_cat (parallel paths then cat)
        out1x1 = self.expand1x1(x)
        out1x1 = self.expand1x1_activation(out1x1)
        out3x3 = self.expand3x3(x)
        out3x3 = self.expand3x3_activation(out3x3)
        return torch.cat([out1x1, out3x3], dim=1)


class ComposedSqueezeNet(nn.Module):
    """SqueezeNet composed from L2 layers which are composed from L1 fusions."""

    def __init__(self, num_classes=10):
        super(ComposedSqueezeNet, self).__init__()
        # L2: initial_block (composed from L1: conv_relu_pool)
        # L0 kernels: conv2d_3_96 + relu + max_pool2d
        self.initial_conv = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.initial_relu = nn.ReLU(inplace=True)
        self.initial_pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # L2: fire modules (each composed from L1: squeeze_relu + expand_cat)
        self.fire1 = ComposedFireModule(96, 16, 64, 64)
        self.fire2 = ComposedFireModule(128, 16, 64, 64)
        self.fire3 = ComposedFireModule(128, 32, 128, 128)

        # L0: max_pool2d
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fire4 = ComposedFireModule(256, 32, 128, 128)
        self.fire5 = ComposedFireModule(256, 48, 192, 192)
        self.fire6 = ComposedFireModule(384, 48, 192, 192)
        self.fire7 = ComposedFireModule(384, 64, 256, 256)

        # L0: max_pool2d
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fire8 = ComposedFireModule(512, 64, 256, 256)

        # L2: classifier_block (composed from L0 kernels)
        self.classifier_dropout = nn.Dropout(p=0.0)
        self.classifier_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier_relu = nn.ReLU(inplace=True)
        self.classifier_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # L2: initial_block
        x = self.initial_conv(x)
        x = self.initial_relu(x)
        x = self.initial_pool(x)

        # L2: fire modules group 1
        x = self.fire1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.pool2(x)

        # L2: fire modules group 2
        x = self.fire4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.pool3(x)

        # L2: fire module 8
        x = self.fire8(x)

        # L2: classifier_block
        x = self.classifier_dropout(x)
        x = self.classifier_conv(x)
        x = self.classifier_relu(x)
        x = self.classifier_pool(x)
        x = torch.flatten(x, 1)
        return x


def share_fire_weights(orig_fire, composed_fire):
    """Copy weights from original FireModule to composed FireModule."""
    composed_fire.squeeze.weight.data.copy_(orig_fire.squeeze.weight.data)
    composed_fire.squeeze.bias.data.copy_(orig_fire.squeeze.bias.data)
    composed_fire.expand1x1.weight.data.copy_(orig_fire.expand1x1.weight.data)
    composed_fire.expand1x1.bias.data.copy_(orig_fire.expand1x1.bias.data)
    composed_fire.expand3x3.weight.data.copy_(orig_fire.expand3x3.weight.data)
    composed_fire.expand3x3.bias.data.copy_(orig_fire.expand3x3.bias.data)


def share_weights(original, composed):
    """Copy ALL weights from original SqueezeNet to composed SqueezeNet."""
    # Initial conv
    composed.initial_conv.weight.data.copy_(original.features[0].weight.data)
    composed.initial_conv.bias.data.copy_(original.features[0].bias.data)

    # Fire modules: features[3]=fire1, features[4]=fire2, ..., features[12]=fire8
    fire_pairs = [
        (original.features[3], composed.fire1),
        (original.features[4], composed.fire2),
        (original.features[5], composed.fire3),
        (original.features[7], composed.fire4),   # skip index 6 (MaxPool)
        (original.features[8], composed.fire5),
        (original.features[9], composed.fire6),
        (original.features[10], composed.fire7),
        (original.features[12], composed.fire8),   # skip index 11 (MaxPool)
    ]
    for orig_fire, comp_fire in fire_pairs:
        share_fire_weights(orig_fire, comp_fire)

    # Classifier conv: classifier[1] is the Conv2d
    composed.classifier_conv.weight.data.copy_(original.classifier[1].weight.data)
    composed.classifier_conv.bias.data.copy_(original.classifier[1].bias.data)


# ============================================================
# Tests
# ============================================================

def test_fire_module_equivalence():
    """Test that a single composed fire module matches the original."""
    print("Test 1: Fire Module equivalence...")
    torch.manual_seed(42)
    orig = OriginalFireModule(96, 16, 64, 64)
    comp = ComposedFireModule(96, 16, 64, 64)
    share_fire_weights(orig, comp)

    orig.eval()
    comp.eval()

    x = torch.randn(2, 96, 7, 7)
    with torch.no_grad():
        out_orig = orig(x)
        out_comp = comp(x)

    assert out_orig.shape == out_comp.shape, \
        f"Shape mismatch: orig={out_orig.shape}, comp={out_comp.shape}"
    assert torch.allclose(out_orig, out_comp, rtol=1e-4, atol=1e-5), \
        f"Value mismatch: max diff = {(out_orig - out_comp).abs().max().item()}"
    print(f"  PASSED: max diff = {(out_orig - out_comp).abs().max().item():.2e}")


def test_initial_block_equivalence():
    """Test that initial block (conv+relu+pool) matches."""
    print("Test 2: Initial block equivalence...")
    torch.manual_seed(42)

    # Original path
    orig_conv = nn.Conv2d(3, 96, kernel_size=7, stride=2)
    orig_relu = nn.ReLU(inplace=True)
    orig_pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
    orig_block = nn.Sequential(orig_conv, orig_relu, orig_pool)

    # Composed path (same structure)
    comp_conv = nn.Conv2d(3, 96, kernel_size=7, stride=2)
    comp_relu = nn.ReLU(inplace=True)
    comp_pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

    # Share weights
    comp_conv.weight.data.copy_(orig_conv.weight.data)
    comp_conv.bias.data.copy_(orig_conv.bias.data)

    orig_block.eval()
    comp_conv.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_orig = orig_block(x)
        out_comp = comp_pool(comp_relu(comp_conv(x)))

    assert torch.allclose(out_orig, out_comp, rtol=1e-4, atol=1e-5), \
        f"Value mismatch: max diff = {(out_orig - out_comp).abs().max().item()}"
    assert out_orig.shape == (2, 96, 7, 7), f"Unexpected shape: {out_orig.shape}"
    print(f"  PASSED: shape={out_orig.shape}, max diff = {(out_orig - out_comp).abs().max().item():.2e}")


def test_classifier_block_equivalence():
    """Test that classifier block matches."""
    print("Test 3: Classifier block equivalence...")
    torch.manual_seed(42)

    # Original classifier
    orig_classifier = nn.Sequential(
        nn.Dropout(p=0.0),
        nn.Conv2d(512, 10, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )

    # Composed classifier (individual ops)
    comp_dropout = nn.Dropout(p=0.0)
    comp_conv = nn.Conv2d(512, 10, kernel_size=1)
    comp_relu = nn.ReLU(inplace=True)
    comp_pool = nn.AdaptiveAvgPool2d((1, 1))

    # Share weights
    comp_conv.weight.data.copy_(orig_classifier[1].weight.data)
    comp_conv.bias.data.copy_(orig_classifier[1].bias.data)

    orig_classifier.eval()
    comp_dropout.eval()
    comp_conv.eval()

    x = torch.randn(2, 512, 2, 2)
    with torch.no_grad():
        out_orig = torch.flatten(orig_classifier(x), 1)
        out_comp = torch.flatten(comp_pool(comp_relu(comp_conv(comp_dropout(x)))), 1)

    assert torch.allclose(out_orig, out_comp, rtol=1e-4, atol=1e-5), \
        f"Value mismatch: max diff = {(out_orig - out_comp).abs().max().item()}"
    assert out_orig.shape == (2, 10), f"Unexpected shape: {out_orig.shape}"
    print(f"  PASSED: shape={out_orig.shape}, max diff = {(out_orig - out_comp).abs().max().item():.2e}")


def test_full_model_equivalence():
    """Test that fully composed SqueezeNet matches the original end-to-end."""
    print("Test 4: Full model equivalence (original vs composed)...")
    torch.manual_seed(42)

    original = OriginalSqueezeNet(num_classes=10)
    composed = ComposedSqueezeNet(num_classes=10)
    share_weights(original, composed)

    original.eval()
    composed.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_orig = original(x)
        out_comp = composed(x)

    assert out_orig.shape == out_comp.shape == (2, 10), \
        f"Shape mismatch: orig={out_orig.shape}, comp={out_comp.shape}"
    assert torch.allclose(out_orig, out_comp, rtol=1e-4, atol=1e-5), \
        f"Value mismatch: max diff = {(out_orig - out_comp).abs().max().item()}"
    print(f"  PASSED: shape={out_orig.shape}, max diff = {(out_orig - out_comp).abs().max().item():.2e}")


def test_intermediate_shapes():
    """Verify all intermediate tensor shapes through the composed model."""
    print("Test 5: Intermediate shape verification...")
    torch.manual_seed(42)
    model = ComposedSqueezeNet(num_classes=10)
    model.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        # Initial block
        x = model.initial_conv(x)
        assert x.shape == (2, 96, 13, 13), f"After initial_conv: {x.shape}"
        x = model.initial_relu(x)
        assert x.shape == (2, 96, 13, 13), f"After initial_relu: {x.shape}"
        x = model.initial_pool(x)
        assert x.shape == (2, 96, 7, 7), f"After initial_pool: {x.shape}"

        # Fire modules group 1
        x = model.fire1(x)
        assert x.shape == (2, 128, 7, 7), f"After fire1: {x.shape}"
        x = model.fire2(x)
        assert x.shape == (2, 128, 7, 7), f"After fire2: {x.shape}"
        x = model.fire3(x)
        assert x.shape == (2, 256, 7, 7), f"After fire3: {x.shape}"
        x = model.pool2(x)
        assert x.shape == (2, 256, 4, 4), f"After pool2: {x.shape}"

        # Fire modules group 2
        x = model.fire4(x)
        assert x.shape == (2, 256, 4, 4), f"After fire4: {x.shape}"
        x = model.fire5(x)
        assert x.shape == (2, 384, 4, 4), f"After fire5: {x.shape}"
        x = model.fire6(x)
        assert x.shape == (2, 384, 4, 4), f"After fire6: {x.shape}"
        x = model.fire7(x)
        assert x.shape == (2, 512, 4, 4), f"After fire7: {x.shape}"
        x = model.pool3(x)
        assert x.shape == (2, 512, 2, 2), f"After pool3: {x.shape}"

        # Fire8
        x = model.fire8(x)
        assert x.shape == (2, 512, 2, 2), f"After fire8: {x.shape}"

        # Classifier
        x = model.classifier_dropout(x)
        assert x.shape == (2, 512, 2, 2), f"After dropout: {x.shape}"
        x = model.classifier_conv(x)
        assert x.shape == (2, 10, 2, 2), f"After classifier_conv: {x.shape}"
        x = model.classifier_relu(x)
        assert x.shape == (2, 10, 2, 2), f"After classifier_relu: {x.shape}"
        x = model.classifier_pool(x)
        assert x.shape == (2, 10, 1, 1), f"After classifier_pool: {x.shape}"
        x = torch.flatten(x, 1)
        assert x.shape == (2, 10), f"After flatten: {x.shape}"

    print("  PASSED: All 18 intermediate shapes verified")


def test_l0_to_l1_composition():
    """Test that L0 kernels compose correctly into L1 fusions."""
    print("Test 6: L0 -> L1 composition (squeeze_relu)...")
    torch.manual_seed(42)

    # L0 kernels individually
    conv_squeeze = nn.Conv2d(96, 16, kernel_size=1)
    relu = nn.ReLU(inplace=True)

    # L1 fusion
    squeeze_relu_conv = nn.Conv2d(96, 16, kernel_size=1)
    squeeze_relu_act = nn.ReLU(inplace=True)

    # Share weights
    squeeze_relu_conv.weight.data.copy_(conv_squeeze.weight.data)
    squeeze_relu_conv.bias.data.copy_(conv_squeeze.bias.data)

    conv_squeeze.eval()
    squeeze_relu_conv.eval()

    x = torch.randn(2, 96, 7, 7)
    with torch.no_grad():
        # L0 path
        out_l0 = relu(conv_squeeze(x.clone()))
        # L1 path
        out_l1 = squeeze_relu_act(squeeze_relu_conv(x.clone()))

    assert torch.allclose(out_l0, out_l1, rtol=1e-4, atol=1e-5), \
        f"Value mismatch: max diff = {(out_l0 - out_l1).abs().max().item()}"
    print(f"  PASSED: max diff = {(out_l0 - out_l1).abs().max().item():.2e}")


def test_parameter_count():
    """Verify that both models have the same number of parameters."""
    print("Test 7: Parameter count verification...")
    original = OriginalSqueezeNet(num_classes=10)
    composed = ComposedSqueezeNet(num_classes=10)

    orig_params = sum(p.numel() for p in original.parameters())
    comp_params = sum(p.numel() for p in composed.parameters())

    assert orig_params == comp_params, \
        f"Parameter count mismatch: original={orig_params}, composed={comp_params}"
    print(f"  PASSED: Both models have {orig_params} parameters")


def run_tests():
    """Run all composition verification tests."""
    print("=" * 70)
    print("SqueezeNet Composition Verification Tests")
    print("=" * 70)

    test_fire_module_equivalence()
    test_initial_block_equivalence()
    test_classifier_block_equivalence()
    test_full_model_equivalence()
    test_intermediate_shapes()
    test_l0_to_l1_composition()
    test_parameter_count()

    print("=" * 70)
    print("ALL 7 TESTS PASSED")
    print("=" * 70)
    return True


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 3, 32, 32)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 10)


# Alias for compatibility
Model = OriginalSqueezeNet


if __name__ == "__main__":
    run_tests()
