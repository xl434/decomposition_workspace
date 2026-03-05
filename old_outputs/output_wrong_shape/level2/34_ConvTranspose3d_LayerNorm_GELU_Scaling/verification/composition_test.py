"""
Composition test for 34_ConvTranspose3d_LayerNorm_GELU_Scaling.
Verifies that chaining the individual kernels (conv_transpose3d, layer_norm, gelu, scaling)
produces identical results to the full fusion model, given shared weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path so we can import kernel modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from level_0_kernel.conv_transpose3d import Model as ConvTranspose3dModel
from level_0_kernel.layer_norm import Model as LayerNormModel
from level_0_kernel.gelu import Model as GELUModel
from level_0_kernel.scaling import Model as ScalingModel
from level_1_fusion.conv_norm_gelu_scale import Model as FusionModel


def test_composition():
    """Test that composed kernels match the original fused model."""
    torch.manual_seed(42)

    # Original model (inline definition matching the original code)
    class OriginalModel(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                     bias=True, eps=1e-5, scaling_factor=1.0):
            super().__init__()
            self.conv_transpose = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=bias
            )
            self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
            self.scaling_factor = scaling_factor

        def forward(self, x):
            x = self.conv_transpose(x)
            x = self.layer_norm(x)
            x = F.gelu(x)
            x = x * self.scaling_factor
            return x

    # Composed model from individual kernels
    class ComposedModel(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                     bias=True, eps=1e-5, scaling_factor=1.0):
            super().__init__()
            self.conv_kernel = ConvTranspose3dModel(
                in_channels, out_channels, kernel_size, stride, padding, bias
            )
            self.norm_kernel = LayerNormModel(out_channels, eps)
            self.gelu_kernel = GELUModel()
            self.scale_kernel = ScalingModel(scaling_factor)

        def forward(self, x):
            x = self.conv_kernel(x)
            x = self.norm_kernel(x)
            x = self.gelu_kernel(x)
            x = self.scale_kernel(x)
            return x

    # Test parameters
    in_channels = 2
    out_channels = 4
    kernel_size = 1
    stride = 1
    padding = 0
    bias = True
    eps = 1e-5
    scaling_factor = 2.0

    original = OriginalModel(in_channels, out_channels, kernel_size, stride, padding,
                             bias, eps, scaling_factor)
    composed = ComposedModel(in_channels, out_channels, kernel_size, stride, padding,
                             bias, eps, scaling_factor)

    # Transfer weights from original to composed
    # ConvTranspose3d weights
    composed.conv_kernel.conv_transpose.weight.data.copy_(original.conv_transpose.weight.data)
    composed.conv_kernel.conv_transpose.bias.data.copy_(original.conv_transpose.bias.data)
    # LayerNorm weights
    composed.norm_kernel.layer_norm.weight.data.copy_(original.layer_norm.weight.data)
    composed.norm_kernel.layer_norm.bias.data.copy_(original.layer_norm.bias.data)

    original.eval()
    composed.eval()

    # Also test the fusion model
    fusion = FusionModel(in_channels, out_channels, kernel_size, stride, padding,
                         bias, eps, scaling_factor)
    fusion.conv_transpose.weight.data.copy_(original.conv_transpose.weight.data)
    fusion.conv_transpose.bias.data.copy_(original.conv_transpose.bias.data)
    fusion.layer_norm.weight.data.copy_(original.layer_norm.weight.data)
    fusion.layer_norm.bias.data.copy_(original.layer_norm.bias.data)
    fusion.eval()

    with torch.no_grad():
        x = torch.randn(2, in_channels, 2, 4, 4)

        out_original = original(x)
        out_composed = composed(x)
        out_fusion = fusion(x)

        # Check composed vs original
        max_diff_composed = (out_original - out_composed).abs().max().item()
        print(f"Max difference (original vs composed): {max_diff_composed}")
        assert torch.allclose(out_original, out_composed, rtol=1e-4, atol=1e-5), \
            f"Composition test FAILED: max_diff={max_diff_composed}"
        print("PASS - Composed kernels match original model")

        # Check fusion vs original
        max_diff_fusion = (out_original - out_fusion).abs().max().item()
        print(f"Max difference (original vs fusion): {max_diff_fusion}")
        assert torch.allclose(out_original, out_fusion, rtol=1e-4, atol=1e-5), \
            f"Fusion test FAILED: max_diff={max_diff_fusion}"
        print("PASS - Fusion model matches original model")

        # Verify output shape
        expected_shape = (2, out_channels, 2, 4, 4)
        assert out_original.shape == expected_shape, \
            f"Shape mismatch: {out_original.shape} vs {expected_shape}"
        print(f"Output shape: {out_original.shape}")

        print("\nAll composition tests passed.")


if __name__ == "__main__":
    test_composition()
