"""
Composition Verification Test for 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling

Verifies:
1. Level 1 fusions match when composed from Level 0 kernels
2. Level 2 layer matches when composed from Level 1 fusions
3. Full composed model (all Level 0 kernels chained) matches the original model
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Level 0 kernels
from level_0_kernel.conv_transpose3d import Model as ConvTranspose3dModel
from level_0_kernel.mean_pool import Model as MeanPoolModel
from level_0_kernel.bias_add import Model as BiasAddModel
from level_0_kernel.softmax import Model as SoftmaxModel
from level_0_kernel.tanh import Model as TanhModel
from level_0_kernel.scaling import Model as ScalingModel

# Import Level 1 fusions
from level_1_fusion.conv_mean_bias import Model as ConvMeanBiasModel
from level_1_fusion.softmax_tanh_scale import Model as SoftmaxTanhScaleModel

# Import Level 2 layer
from level_2_layer.conv_mean_softmax_tanh_scale import Model as FullLayerModel


class OriginalModel(nn.Module):
    """The original model from the problem specification."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(OriginalModel, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.mean(dim=2, keepdim=True)
        x = x + self.bias
        x = torch.softmax(x, dim=1)
        x = torch.tanh(x)
        x = x * self.scaling_factor
        return x


def test_level1_fusion1_from_kernels():
    """Test: conv_mean_bias fusion matches composition of conv_transpose3d -> mean_pool -> bias_add kernels."""
    print("=" * 70)
    print("TEST 1: Level 1 Fusion 'conv_mean_bias' vs composed Level 0 kernels")
    print("=" * 70)

    torch.manual_seed(42)

    # Create the fusion model
    fusion = ConvMeanBiasModel(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
    fusion.eval()

    # Create individual kernels and share weights
    k_conv = ConvTranspose3dModel(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
    k_mean = MeanPoolModel()
    k_bias = BiasAddModel(out_channels=4)

    # Share weights from fusion to kernels
    k_conv.conv_transpose.weight.data.copy_(fusion.conv_transpose.weight.data)
    k_conv.conv_transpose.bias.data.copy_(fusion.conv_transpose.bias.data)
    k_bias.bias.data.copy_(fusion.bias.data)

    k_conv.eval()
    k_mean.eval()
    k_bias.eval()

    # Run with same input
    torch.manual_seed(123)
    x = torch.randn(2, 2, 4, 4, 4)

    with torch.no_grad():
        # Fusion output
        out_fusion = fusion(x)

        # Composed kernel output
        out_composed = k_conv(x)
        out_composed = k_mean(out_composed)
        out_composed = k_bias(out_composed)

    match = torch.allclose(out_fusion, out_composed, rtol=1e-4, atol=1e-5)
    max_diff = (out_fusion - out_composed).abs().max().item()
    print(f"  Fusion output shape: {out_fusion.shape}")
    print(f"  Composed output shape: {out_composed.shape}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  RESULT: {'PASS' if match else 'FAIL'}")
    print()
    return match


def test_level1_fusion2_from_kernels():
    """Test: softmax_tanh_scale fusion matches composition of softmax -> tanh -> scaling kernels."""
    print("=" * 70)
    print("TEST 2: Level 1 Fusion 'softmax_tanh_scale' vs composed Level 0 kernels")
    print("=" * 70)

    torch.manual_seed(42)

    # Create the fusion model
    fusion = SoftmaxTanhScaleModel(scaling_factor=2.0)
    fusion.eval()

    # Create individual kernels
    k_softmax = SoftmaxModel()
    k_tanh = TanhModel()
    k_scale = ScalingModel(scaling_factor=2.0)

    k_softmax.eval()
    k_tanh.eval()
    k_scale.eval()

    # Run with same input
    torch.manual_seed(123)
    x = torch.randn(2, 4, 1, 4, 4)

    with torch.no_grad():
        # Fusion output
        out_fusion = fusion(x)

        # Composed kernel output
        out_composed = k_softmax(x)
        out_composed = k_tanh(out_composed)
        out_composed = k_scale(out_composed)

    match = torch.allclose(out_fusion, out_composed, rtol=1e-4, atol=1e-5)
    max_diff = (out_fusion - out_composed).abs().max().item()
    print(f"  Fusion output shape: {out_fusion.shape}")
    print(f"  Composed output shape: {out_composed.shape}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  RESULT: {'PASS' if match else 'FAIL'}")
    print()
    return match


def test_level2_from_fusions():
    """Test: Level 2 layer matches composition of Level 1 fusions."""
    print("=" * 70)
    print("TEST 3: Level 2 Layer vs composed Level 1 fusions")
    print("=" * 70)

    torch.manual_seed(42)

    # Create the full layer model
    layer = FullLayerModel(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1, scaling_factor=2.0)
    layer.eval()

    # Create Level 1 fusions and share weights
    f_conv_mean_bias = ConvMeanBiasModel(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
    f_softmax_tanh_scale = SoftmaxTanhScaleModel(scaling_factor=2.0)

    # Share weights from layer to fusion 1
    f_conv_mean_bias.conv_transpose.weight.data.copy_(layer.conv_transpose.weight.data)
    f_conv_mean_bias.conv_transpose.bias.data.copy_(layer.conv_transpose.bias.data)
    f_conv_mean_bias.bias.data.copy_(layer.bias.data)

    f_conv_mean_bias.eval()
    f_softmax_tanh_scale.eval()

    # Run with same input
    torch.manual_seed(123)
    x = torch.randn(2, 2, 4, 4, 4)

    with torch.no_grad():
        # Layer output
        out_layer = layer(x)

        # Composed fusion output
        out_composed = f_conv_mean_bias(x)
        out_composed = f_softmax_tanh_scale(out_composed)

    match = torch.allclose(out_layer, out_composed, rtol=1e-4, atol=1e-5)
    max_diff = (out_layer - out_composed).abs().max().item()
    print(f"  Layer output shape: {out_layer.shape}")
    print(f"  Composed output shape: {out_composed.shape}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  RESULT: {'PASS' if match else 'FAIL'}")
    print()
    return match


def test_full_composition_vs_original():
    """Test: Full composition of all Level 0 kernels matches the original model."""
    print("=" * 70)
    print("TEST 4: Full Level 0 kernel composition vs Original model")
    print("=" * 70)

    torch.manual_seed(42)

    # Create original model
    original = OriginalModel(
        in_channels=2, out_channels=4, kernel_size=3,
        stride=1, padding=1, scaling_factor=2.0
    )
    original.eval()

    # Create all Level 0 kernels
    k_conv = ConvTranspose3dModel(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
    k_mean = MeanPoolModel()
    k_bias = BiasAddModel(out_channels=4)
    k_softmax = SoftmaxModel()
    k_tanh = TanhModel()
    k_scale = ScalingModel(scaling_factor=2.0)

    # Share weights from original to kernels
    k_conv.conv_transpose.weight.data.copy_(original.conv_transpose.weight.data)
    k_conv.conv_transpose.bias.data.copy_(original.conv_transpose.bias.data)
    k_bias.bias.data.copy_(original.bias.data)

    k_conv.eval()
    k_mean.eval()
    k_bias.eval()
    k_softmax.eval()
    k_tanh.eval()
    k_scale.eval()

    # Run with same input
    torch.manual_seed(123)
    x = torch.randn(2, 2, 4, 4, 4)

    with torch.no_grad():
        # Original output
        out_original = original(x)

        # Full composed output: conv -> mean -> bias -> softmax -> tanh -> scale
        out_composed = k_conv(x)
        out_composed = k_mean(out_composed)
        out_composed = k_bias(out_composed)
        out_composed = k_softmax(out_composed)
        out_composed = k_tanh(out_composed)
        out_composed = k_scale(out_composed)

    match = torch.allclose(out_original, out_composed, rtol=1e-4, atol=1e-5)
    max_diff = (out_original - out_composed).abs().max().item()
    print(f"  Original output shape: {out_original.shape}")
    print(f"  Composed output shape: {out_composed.shape}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  RESULT: {'PASS' if match else 'FAIL'}")
    print()
    return match


def test_level2_vs_original():
    """Test: Level 2 layer matches the original model exactly."""
    print("=" * 70)
    print("TEST 5: Level 2 Layer vs Original model")
    print("=" * 70)

    torch.manual_seed(42)

    # Create original model
    original = OriginalModel(
        in_channels=2, out_channels=4, kernel_size=3,
        stride=1, padding=1, scaling_factor=2.0
    )
    original.eval()

    # Create level 2 layer and share weights
    layer = FullLayerModel(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1, scaling_factor=2.0)
    layer.conv_transpose.weight.data.copy_(original.conv_transpose.weight.data)
    layer.conv_transpose.bias.data.copy_(original.conv_transpose.bias.data)
    layer.bias.data.copy_(original.bias.data)
    layer.eval()

    # Run with same input
    torch.manual_seed(123)
    x = torch.randn(2, 2, 4, 4, 4)

    with torch.no_grad():
        out_original = original(x)
        out_layer = layer(x)

    match = torch.allclose(out_original, out_layer, rtol=1e-4, atol=1e-5)
    max_diff = (out_original - out_layer).abs().max().item()
    print(f"  Original output shape: {out_original.shape}")
    print(f"  Layer output shape: {out_layer.shape}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  RESULT: {'PASS' if match else 'FAIL'}")
    print()
    return match


def run_tests():
    print("Composition Verification Tests")
    print("Model: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling")
    print("=" * 70)
    print()

    results = []
    results.append(("Level 1 Fusion 1 (conv_mean_bias) vs Level 0 kernels", test_level1_fusion1_from_kernels()))
    results.append(("Level 1 Fusion 2 (softmax_tanh_scale) vs Level 0 kernels", test_level1_fusion2_from_kernels()))
    results.append(("Level 2 Layer vs Level 1 fusions", test_level2_from_fusions()))
    results.append(("Full Level 0 composition vs Original", test_full_composition_vs_original()))
    results.append(("Level 2 Layer vs Original", test_level2_vs_original()))

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")

    return all_pass


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
