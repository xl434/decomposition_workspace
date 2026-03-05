"""
Composition Test for 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

Verifies that composing individual level-0 kernels produces the same output
as the fused level-1 model, with shared weights.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from level_0_kernel.conv_transpose3d import Model as ConvTranspose3dModel
from level_0_kernel.scale1 import Model as Scale1Model
from level_0_kernel.avg_pool3d import Model as AvgPool3dModel
from level_0_kernel.bias_add import Model as BiasAddModel
from level_0_kernel.scale2 import Model as Scale2Model
from level_1_fusion.conv_scale_pool_bias_scale import Model as FusedModel


def run_composition_test():
    """Test that composed kernels match the fused model output."""
    try:
        torch.manual_seed(42)

        # Create the fused (original) model
        fused_model = FusedModel(
            in_channels=2, out_channels=4, kernel_size=3,
            stride=2, padding=1, scale1=0.5, scale2=1.0,
            bias_shape=(4, 1, 1, 1)
        )
        fused_model.eval()

        # Create individual kernel models
        conv_transpose_model = ConvTranspose3dModel(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1)
        scale1_model = Scale1Model(scale_value=0.5)
        avg_pool_model = AvgPool3dModel(kernel_size=2)
        bias_add_model = BiasAddModel(bias_shape=(4, 1, 1, 1))
        scale2_model = Scale2Model(scale_value=1.0)

        # Share weights from fused model to individual kernels
        conv_transpose_model.conv_transpose.load_state_dict(fused_model.conv_transpose.state_dict())
        scale1_model.scale.data.copy_(fused_model.scale1.data)
        bias_add_model.bias.data.copy_(fused_model.bias.data)
        scale2_model.scale.data.copy_(fused_model.scale2.data)

        # Set all models to eval mode
        conv_transpose_model.eval()
        scale1_model.eval()
        avg_pool_model.eval()
        bias_add_model.eval()
        scale2_model.eval()

        # Create test input
        x = torch.randn(2, 2, 4, 4, 4)

        with torch.no_grad():
            # Fused model output
            fused_output = fused_model(x)

            # Composed kernel output (sequential pipeline)
            out = conv_transpose_model(x)
            out = scale1_model(out)
            out = avg_pool_model(out)
            out = bias_add_model(out)
            composed_output = scale2_model(out)

        # Verify outputs match
        assert fused_output.shape == composed_output.shape, \
            f"Shape mismatch: fused={fused_output.shape} vs composed={composed_output.shape}"

        assert torch.allclose(fused_output, composed_output, rtol=1e-4, atol=1e-5), \
            f"Output mismatch! Max diff: {(fused_output - composed_output).abs().max().item()}"

        print(f"Fused model output shape: {fused_output.shape}")
        print(f"Composed model output shape: {composed_output.shape}")
        print(f"Max absolute difference: {(fused_output - composed_output).abs().max().item():.2e}")
        print("PASS")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(0 if run_composition_test() else 1)
