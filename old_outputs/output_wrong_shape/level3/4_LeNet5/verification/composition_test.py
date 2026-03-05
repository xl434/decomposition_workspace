"""
Composition Test for LeNet-5 (4_LeNet5) Hierarchical Decomposition

This test verifies that the decomposed kernels and fusions compose correctly
to reproduce the original LeNet-5 model's behavior. It builds the original
model inline, then composes from Level 0 kernels and Level 1 fusions,
sharing weights to ensure numerical equivalence.

Tests:
1. L0 kernel composition: chain all 12 kernels with shared weights -> match original
2. L1 fusion composition: chain all 5 fusions with shared weights -> match original
3. L2 layer direct: run full model -> match original
4. Cross-level consistency: L0 composed == L1 composed == L2 output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Original LeNet-5 Model (built inline for reference)
# ============================================================================
class OriginalLeNet5(nn.Module):
    """Original LeNet-5 model as specified in KernelBench."""

    def __init__(self, num_classes):
        super(OriginalLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================================
# Import Level 0 kernels
# ============================================================================
from level_0_kernel.conv2d_1 import Model as Conv2d1Model
from level_0_kernel.relu_1 import Model as ReLU1Model
from level_0_kernel.max_pool2d_1 import Model as MaxPool2d1Model
from level_0_kernel.conv2d_2 import Model as Conv2d2Model
from level_0_kernel.relu_2 import Model as ReLU2Model
from level_0_kernel.max_pool2d_2 import Model as MaxPool2d2Model
from level_0_kernel.flatten import Model as FlattenModel
from level_0_kernel.linear_1 import Model as Linear1Model
from level_0_kernel.relu_3 import Model as ReLU3Model
from level_0_kernel.linear_2 import Model as Linear2Model
from level_0_kernel.relu_4 import Model as ReLU4Model
from level_0_kernel.linear_3 import Model as Linear3Model

# ============================================================================
# Import Level 1 fusions
# ============================================================================
from level_1_fusion.conv_relu_pool_1 import Model as ConvReluPool1Model
from level_1_fusion.conv_relu_pool_2 import Model as ConvReluPool2Model
from level_1_fusion.flatten_fc_relu_1 import Model as FlattenFcRelu1Model
from level_1_fusion.fc_relu_2 import Model as FcRelu2Model
from level_1_fusion.fc_output import Model as FcOutputModel

# ============================================================================
# Import Level 2 layer
# ============================================================================
from level_2_layer.lenet5 import Model as LeNet5Model


def share_weights_l0(original, kernels):
    """Share weights from original model to L0 kernel instances.

    Args:
        original: OriginalLeNet5 model
        kernels: dict of kernel name -> kernel model instance
    """
    # conv2d_1 shares conv1 weights
    kernels["conv2d_1"].conv.weight.data.copy_(original.conv1.weight.data)
    kernels["conv2d_1"].conv.bias.data.copy_(original.conv1.bias.data)

    # conv2d_2 shares conv2 weights
    kernels["conv2d_2"].conv.weight.data.copy_(original.conv2.weight.data)
    kernels["conv2d_2"].conv.bias.data.copy_(original.conv2.bias.data)

    # linear_1 shares fc1 weights
    kernels["linear_1"].fc.weight.data.copy_(original.fc1.weight.data)
    kernels["linear_1"].fc.bias.data.copy_(original.fc1.bias.data)

    # linear_2 shares fc2 weights
    kernels["linear_2"].fc.weight.data.copy_(original.fc2.weight.data)
    kernels["linear_2"].fc.bias.data.copy_(original.fc2.bias.data)

    # linear_3 shares fc3 weights
    kernels["linear_3"].fc.weight.data.copy_(original.fc3.weight.data)
    kernels["linear_3"].fc.bias.data.copy_(original.fc3.bias.data)


def share_weights_l1(original, fusions):
    """Share weights from original model to L1 fusion instances.

    Args:
        original: OriginalLeNet5 model
        fusions: dict of fusion name -> fusion model instance
    """
    # conv_relu_pool_1 shares conv1 weights
    fusions["conv_relu_pool_1"].conv.weight.data.copy_(original.conv1.weight.data)
    fusions["conv_relu_pool_1"].conv.bias.data.copy_(original.conv1.bias.data)

    # conv_relu_pool_2 shares conv2 weights
    fusions["conv_relu_pool_2"].conv.weight.data.copy_(original.conv2.weight.data)
    fusions["conv_relu_pool_2"].conv.bias.data.copy_(original.conv2.bias.data)

    # flatten_fc_relu_1 shares fc1 weights
    fusions["flatten_fc_relu_1"].fc.weight.data.copy_(original.fc1.weight.data)
    fusions["flatten_fc_relu_1"].fc.bias.data.copy_(original.fc1.bias.data)

    # fc_relu_2 shares fc2 weights
    fusions["fc_relu_2"].fc.weight.data.copy_(original.fc2.weight.data)
    fusions["fc_relu_2"].fc.bias.data.copy_(original.fc2.bias.data)

    # fc_output shares fc3 weights
    fusions["fc_output"].fc.weight.data.copy_(original.fc3.weight.data)
    fusions["fc_output"].fc.bias.data.copy_(original.fc3.bias.data)


def share_weights_l2(original, l2_model):
    """Share weights from original model to L2 model.

    Args:
        original: OriginalLeNet5 model
        l2_model: LeNet5Model instance
    """
    l2_model.conv1.weight.data.copy_(original.conv1.weight.data)
    l2_model.conv1.bias.data.copy_(original.conv1.bias.data)
    l2_model.conv2.weight.data.copy_(original.conv2.weight.data)
    l2_model.conv2.bias.data.copy_(original.conv2.bias.data)
    l2_model.fc1.weight.data.copy_(original.fc1.weight.data)
    l2_model.fc1.bias.data.copy_(original.fc1.bias.data)
    l2_model.fc2.weight.data.copy_(original.fc2.weight.data)
    l2_model.fc2.bias.data.copy_(original.fc2.bias.data)
    l2_model.fc3.weight.data.copy_(original.fc3.weight.data)
    l2_model.fc3.bias.data.copy_(original.fc3.bias.data)


def compose_l0_forward(kernels, x):
    """Run forward pass by composing all L0 kernels sequentially.

    Args:
        kernels: dict of kernel name -> kernel model instance
        x: Input tensor [batch_size, 1, 32, 32]
    Returns:
        Output tensor [batch_size, 10]
    """
    x = kernels["conv2d_1"](x)        # [B,1,32,32] -> [B,6,28,28]
    x = kernels["relu_1"](x)          # [B,6,28,28] -> [B,6,28,28]
    x = kernels["max_pool2d_1"](x)    # [B,6,28,28] -> [B,6,14,14]
    x = kernels["conv2d_2"](x)        # [B,6,14,14] -> [B,16,10,10]
    x = kernels["relu_2"](x)          # [B,16,10,10] -> [B,16,10,10]
    x = kernels["max_pool2d_2"](x)    # [B,16,10,10] -> [B,16,5,5]
    x = kernels["flatten"](x)         # [B,16,5,5] -> [B,400]
    x = kernels["linear_1"](x)        # [B,400] -> [B,120]
    x = kernels["relu_3"](x)          # [B,120] -> [B,120]
    x = kernels["linear_2"](x)        # [B,120] -> [B,84]
    x = kernels["relu_4"](x)          # [B,84] -> [B,84]
    x = kernels["linear_3"](x)        # [B,84] -> [B,10]
    return x


def compose_l1_forward(fusions, x):
    """Run forward pass by composing all L1 fusions sequentially.

    Args:
        fusions: dict of fusion name -> fusion model instance
        x: Input tensor [batch_size, 1, 32, 32]
    Returns:
        Output tensor [batch_size, 10]
    """
    x = fusions["conv_relu_pool_1"](x)    # [B,1,32,32] -> [B,6,14,14]
    x = fusions["conv_relu_pool_2"](x)    # [B,6,14,14] -> [B,16,5,5]
    x = fusions["flatten_fc_relu_1"](x)   # [B,16,5,5] -> [B,120]
    x = fusions["fc_relu_2"](x)           # [B,120] -> [B,84]
    x = fusions["fc_output"](x)           # [B,84] -> [B,10]
    return x


def run_tests():
    """Run all composition verification tests."""
    print("=" * 70)
    print("LeNet-5 (4_LeNet5) Hierarchical Decomposition - Composition Test")
    print("=" * 70)

    torch.manual_seed(42)

    # ========================================================================
    # Setup: Create original model and test input
    # ========================================================================
    num_classes = 10
    batch_size = 2
    original = OriginalLeNet5(num_classes)
    original.eval()

    x = torch.randn(batch_size, 1, 32, 32)

    with torch.no_grad():
        original_output = original(x)

    print(f"\nOriginal model output shape: {original_output.shape}")
    print(f"Original model output:\n{original_output}")

    # ========================================================================
    # Test 1: L0 Kernel Composition
    # ========================================================================
    print("\n" + "-" * 70)
    print("Test 1: Level 0 Kernel Composition")
    print("-" * 70)

    kernels = {
        "conv2d_1": Conv2d1Model(1, 6, 5, 1),
        "relu_1": ReLU1Model(),
        "max_pool2d_1": MaxPool2d1Model(2, 2),
        "conv2d_2": Conv2d2Model(6, 16, 5, 1),
        "relu_2": ReLU2Model(),
        "max_pool2d_2": MaxPool2d2Model(2, 2),
        "flatten": FlattenModel(),
        "linear_1": Linear1Model(400, 120),
        "relu_3": ReLU3Model(),
        "linear_2": Linear2Model(120, 84),
        "relu_4": ReLU4Model(),
        "linear_3": Linear3Model(84, 10),
    }

    # Share weights from original
    share_weights_l0(original, kernels)

    for k in kernels.values():
        k.eval()

    with torch.no_grad():
        l0_output = compose_l0_forward(kernels, x)

    print(f"L0 composed output shape: {l0_output.shape}")
    l0_match = torch.allclose(original_output, l0_output, atol=1e-6)
    l0_max_diff = (original_output - l0_output).abs().max().item()
    print(f"L0 matches original: {l0_match} (max diff: {l0_max_diff:.2e})")
    assert l0_match, f"L0 composition failed! Max diff: {l0_max_diff}"
    print("PASSED: L0 kernel composition matches original model")

    # ========================================================================
    # Test 2: L1 Fusion Composition
    # ========================================================================
    print("\n" + "-" * 70)
    print("Test 2: Level 1 Fusion Composition")
    print("-" * 70)

    fusions = {
        "conv_relu_pool_1": ConvReluPool1Model(1, 6, 5, 1, 2, 2),
        "conv_relu_pool_2": ConvReluPool2Model(6, 16, 5, 1, 2, 2),
        "flatten_fc_relu_1": FlattenFcRelu1Model(400, 120),
        "fc_relu_2": FcRelu2Model(120, 84),
        "fc_output": FcOutputModel(84, 10),
    }

    # Share weights from original
    share_weights_l1(original, fusions)

    for f in fusions.values():
        f.eval()

    with torch.no_grad():
        l1_output = compose_l1_forward(fusions, x)

    print(f"L1 composed output shape: {l1_output.shape}")
    l1_match = torch.allclose(original_output, l1_output, atol=1e-6)
    l1_max_diff = (original_output - l1_output).abs().max().item()
    print(f"L1 matches original: {l1_match} (max diff: {l1_max_diff:.2e})")
    assert l1_match, f"L1 composition failed! Max diff: {l1_max_diff}"
    print("PASSED: L1 fusion composition matches original model")

    # ========================================================================
    # Test 3: L2 Layer (Full Model from decomposition)
    # ========================================================================
    print("\n" + "-" * 70)
    print("Test 3: Level 2 Layer (Decomposed Full Model)")
    print("-" * 70)

    l2_model = LeNet5Model(num_classes)
    share_weights_l2(original, l2_model)
    l2_model.eval()

    with torch.no_grad():
        l2_output = l2_model(x)

    print(f"L2 model output shape: {l2_output.shape}")
    l2_match = torch.allclose(original_output, l2_output, atol=1e-6)
    l2_max_diff = (original_output - l2_output).abs().max().item()
    print(f"L2 matches original: {l2_match} (max diff: {l2_max_diff:.2e})")
    assert l2_match, f"L2 composition failed! Max diff: {l2_max_diff}"
    print("PASSED: L2 layer matches original model")

    # ========================================================================
    # Test 4: Cross-Level Consistency
    # ========================================================================
    print("\n" + "-" * 70)
    print("Test 4: Cross-Level Consistency")
    print("-" * 70)

    l0_l1_match = torch.allclose(l0_output, l1_output, atol=1e-6)
    l0_l1_diff = (l0_output - l1_output).abs().max().item()
    print(f"L0 == L1: {l0_l1_match} (max diff: {l0_l1_diff:.2e})")

    l1_l2_match = torch.allclose(l1_output, l2_output, atol=1e-6)
    l1_l2_diff = (l1_output - l2_output).abs().max().item()
    print(f"L1 == L2: {l1_l2_match} (max diff: {l1_l2_diff:.2e})")

    l0_l2_match = torch.allclose(l0_output, l2_output, atol=1e-6)
    l0_l2_diff = (l0_output - l2_output).abs().max().item()
    print(f"L0 == L2: {l0_l2_match} (max diff: {l0_l2_diff:.2e})")

    assert l0_l1_match and l1_l2_match and l0_l2_match, (
        "Cross-level consistency check failed!"
    )
    print("PASSED: All levels produce identical outputs")

    # ========================================================================
    # Test 5: Shape Verification Through Data Flow
    # ========================================================================
    print("\n" + "-" * 70)
    print("Test 5: Data Flow Shape Verification")
    print("-" * 70)

    expected_shapes = [
        ("input",        (2, 1, 32, 32)),
        ("conv2d_1",     (2, 6, 28, 28)),
        ("relu_1",       (2, 6, 28, 28)),
        ("max_pool2d_1", (2, 6, 14, 14)),
        ("conv2d_2",     (2, 16, 10, 10)),
        ("relu_2",       (2, 16, 10, 10)),
        ("max_pool2d_2", (2, 16, 5, 5)),
        ("flatten",      (2, 400)),
        ("linear_1",     (2, 120)),
        ("relu_3",       (2, 120)),
        ("linear_2",     (2, 84)),
        ("relu_4",       (2, 84)),
        ("linear_3",     (2, 10)),
    ]

    h = x.clone()
    kernel_order = [
        "conv2d_1", "relu_1", "max_pool2d_1",
        "conv2d_2", "relu_2", "max_pool2d_2",
        "flatten", "linear_1", "relu_3",
        "linear_2", "relu_4", "linear_3",
    ]

    print(f"  {'Stage':<16} {'Expected':<20} {'Actual':<20} {'Match'}")
    print(f"  {'input':<16} {str(expected_shapes[0][1]):<20} {str(tuple(h.shape)):<20} {tuple(h.shape) == expected_shapes[0][1]}")

    with torch.no_grad():
        for i, name in enumerate(kernel_order):
            h = kernels[name](h)
            expected = expected_shapes[i + 1][1]
            actual = tuple(h.shape)
            match = actual == expected
            print(f"  {name:<16} {str(expected):<20} {str(actual):<20} {match}")
            assert match, f"Shape mismatch at {name}: got {actual}, expected {expected}"

    print("PASSED: All intermediate shapes match expected data flow")

    # ========================================================================
    # Test 6: Individual Kernel Standalone Tests
    # ========================================================================
    print("\n" + "-" * 70)
    print("Test 6: Individual Module Standalone Tests")
    print("-" * 70)

    # Import and run each module's run_tests
    from level_0_kernel import conv2d_1, relu_1, max_pool2d_1
    from level_0_kernel import conv2d_2, relu_2, max_pool2d_2
    from level_0_kernel import flatten, linear_1, relu_3
    from level_0_kernel import linear_2, relu_4, linear_3
    from level_1_fusion import conv_relu_pool_1, conv_relu_pool_2
    from level_1_fusion import flatten_fc_relu_1, fc_relu_2, fc_output
    from level_2_layer import lenet5

    l0_modules = [
        conv2d_1, relu_1, max_pool2d_1,
        conv2d_2, relu_2, max_pool2d_2,
        flatten, linear_1, relu_3,
        linear_2, relu_4, linear_3,
    ]
    l1_modules = [
        conv_relu_pool_1, conv_relu_pool_2,
        flatten_fc_relu_1, fc_relu_2, fc_output,
    ]
    l2_modules = [lenet5]

    print("  Running L0 kernel tests...")
    for mod in l0_modules:
        mod.run_tests()

    print("  Running L1 fusion tests...")
    for mod in l1_modules:
        mod.run_tests()

    print("  Running L2 layer tests...")
    for mod in l2_modules:
        mod.run_tests()

    print("PASSED: All individual module tests pass")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("ALL COMPOSITION TESTS PASSED")
    print("=" * 70)
    print(f"  Level 0 kernels:  12 (all verified)")
    print(f"  Level 1 fusions:   5 (all verified)")
    print(f"  Level 2 layer:     1 (verified)")
    print(f"  Cross-level:       consistent (L0 == L1 == L2 == Original)")
    print(f"  Data flow shapes:  all 13 stages verified")
    print("=" * 70)

    return True


if __name__ == "__main__":
    run_tests()
