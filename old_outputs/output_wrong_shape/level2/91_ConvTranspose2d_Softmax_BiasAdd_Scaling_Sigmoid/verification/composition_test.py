"""
Composition Verification Test for 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid

Verifies that chaining the 5 individual kernel components produces the same
output as the fused (original) model, given shared weights and the same input.

Components chained:
  1. ConvTranspose2d(3, 8, 4, stride=2, padding=1, output_padding=1)
  2. Softmax(dim=1)
  3. BiasAdd(bias_shape=(8, 1, 1))
  4. Scaling(scaling_factor=2.0)
  5. Sigmoid()

Test dimensions:
  - Input: [2, 3, 4, 4]
  - Output: [2, 8, 9, 9]
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from level_0_kernel.conv_transpose2d import Model as ConvTranspose2dModel
from level_0_kernel.softmax import Model as SoftmaxModel
from level_0_kernel.bias_add import Model as BiasAddModel
from level_0_kernel.scaling import Model as ScalingModel
from level_0_kernel.sigmoid import Model as SigmoidModel
from level_1_fusion.conv_softmax_bias_scale_sigmoid import Model as FusedModel


class ComposedModel(nn.Module):
    """
    Composed model built by chaining the 5 individual kernel components.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose2d = ConvTranspose2dModel(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.softmax = SoftmaxModel(1)  # dim=1
        self.bias_add = BiasAddModel(bias_shape)
        self.scaling = ScalingModel(scaling_factor)
        self.sigmoid = SigmoidModel()

    def forward(self, x):
        x = self.conv_transpose2d(x)
        x = self.softmax(x)
        x = self.bias_add(x)
        x = self.scaling(x)
        x = self.sigmoid(x)
        return x


def run_tests():
    try:
        torch.manual_seed(42)

        # Test parameters (small test dimensions)
        in_channels = 3
        out_channels = 8
        kernel_size = 4
        stride = 2
        padding = 1
        output_padding = 1
        bias_shape = (8, 1, 1)
        scaling_factor = 2.0

        # Build fused (original) model
        fused_model = FusedModel(
            in_channels, out_channels, kernel_size, stride,
            padding, output_padding, bias_shape, scaling_factor
        )

        # Build composed model from individual kernels
        composed_model = ComposedModel(
            in_channels, out_channels, kernel_size, stride,
            padding, output_padding, bias_shape, scaling_factor
        )

        # Share weights: copy ConvTranspose2d weights from fused to composed
        composed_model.conv_transpose2d.conv_transpose.weight.data.copy_(
            fused_model.conv_transpose.weight.data
        )
        composed_model.conv_transpose2d.conv_transpose.bias.data.copy_(
            fused_model.conv_transpose.bias.data
        )

        # Share weights: copy bias parameter from fused to composed
        composed_model.bias_add.bias.data.copy_(
            fused_model.bias.data
        )

        # Set both to eval mode
        fused_model.eval()
        composed_model.eval()

        # Create test input
        torch.manual_seed(123)
        x = torch.randn(2, 3, 4, 4)

        with torch.no_grad():
            fused_output = fused_model(x)
            composed_output = composed_model(x)

        # Verify shapes match
        assert fused_output.shape == composed_output.shape, \
            f"Shape mismatch: fused {fused_output.shape} vs composed {composed_output.shape}"

        # Verify values match
        rtol = 1e-4
        atol = 1e-5
        match = torch.allclose(fused_output, composed_output, rtol=rtol, atol=atol)

        if not match:
            max_diff = (fused_output - composed_output).abs().max().item()
            mean_diff = (fused_output - composed_output).abs().mean().item()
            print(f"FAIL: Outputs do not match within tolerance (rtol={rtol}, atol={atol})")
            print(f"  Max absolute difference: {max_diff:.2e}")
            print(f"  Mean absolute difference: {mean_diff:.2e}")
            return False

        print(f"Input shape: {x.shape}")
        print(f"Fused output shape: {fused_output.shape}")
        print(f"Composed output shape: {composed_output.shape}")
        print(f"Max absolute difference: {(fused_output - composed_output).abs().max().item():.2e}")
        print(f"Tolerance: rtol={rtol}, atol={atol}")
        print("PASS")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
