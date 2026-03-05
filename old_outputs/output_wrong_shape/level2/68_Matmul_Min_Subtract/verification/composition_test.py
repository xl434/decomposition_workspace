"""
Composition test for 68_Matmul_Min_Subtract.
Verifies that chaining the individual kernels (linear, min_clamp, subtract)
produces identical results to the full fusion model, given shared weights.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path so we can import kernel modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from level_0_kernel.linear import Model as LinearModel
from level_0_kernel.min_clamp import Model as MinClampModel
from level_0_kernel.subtract import Model as SubtractModel
from level_1_fusion.linear_min_subtract import Model as FusionModel


def test_composition():
    """Test that composed kernels match the original fused model."""
    torch.manual_seed(42)

    # Original / fused model
    class OriginalModel(nn.Module):
        def __init__(self, in_features, out_features, constant):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.constant = nn.Parameter(torch.tensor(constant))

        def forward(self, x):
            x = self.linear(x)
            x = torch.min(x, self.constant)
            x = x - self.constant
            return x

    # Composed model from individual kernels
    class ComposedModel(nn.Module):
        def __init__(self, in_features, out_features, constant):
            super().__init__()
            self.linear_kernel = LinearModel(in_features, out_features)
            self.min_kernel = MinClampModel(constant)
            self.subtract_kernel = SubtractModel(constant)

        def forward(self, x):
            x = self.linear_kernel(x)
            x = self.min_kernel(x)
            x = self.subtract_kernel(x)
            return x

    in_features = 16
    out_features = 32
    constant = 2.0

    original = OriginalModel(in_features, out_features, constant)
    composed = ComposedModel(in_features, out_features, constant)

    # Transfer weights from original to composed
    composed.linear_kernel.linear.weight.data.copy_(original.linear.weight.data)
    composed.linear_kernel.linear.bias.data.copy_(original.linear.bias.data)
    composed.min_kernel.constant.data.copy_(original.constant.data)
    composed.subtract_kernel.constant.data.copy_(original.constant.data)

    original.eval()
    composed.eval()

    # Also test the fusion model
    fusion = FusionModel(in_features, out_features, constant)
    fusion.linear.weight.data.copy_(original.linear.weight.data)
    fusion.linear.bias.data.copy_(original.linear.bias.data)
    fusion.constant.data.copy_(original.constant.data)
    fusion.eval()

    with torch.no_grad():
        x = torch.randn(2, in_features)

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

        print("\nAll composition tests passed.")


if __name__ == "__main__":
    test_composition()
