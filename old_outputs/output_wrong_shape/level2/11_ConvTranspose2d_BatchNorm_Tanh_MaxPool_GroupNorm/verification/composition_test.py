"""
Composition Test for 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

Verifies that composing individual level-0 kernels produces the same output
as the fused level-1 model, with shared weights.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from level_0_kernel.conv_transpose2d import Model as ConvTranspose2dModel
from level_0_kernel.batch_norm2d import Model as BatchNorm2dModel
from level_0_kernel.tanh import Model as TanhModel
from level_0_kernel.max_pool2d import Model as MaxPool2dModel
from level_0_kernel.group_norm import Model as GroupNormModel
from level_1_fusion.conv_bn_tanh_pool_gnorm import Model as FusedModel


def run_composition_test():
    """Test that composed kernels match the fused model output."""
    try:
        torch.manual_seed(42)

        # Create the fused (original) model
        fused_model = FusedModel(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, groups=4, num_groups=4)
        fused_model.eval()

        # Create individual kernel models
        conv_transpose_model = ConvTranspose2dModel(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        batch_norm_model = BatchNorm2dModel(num_features=8)
        tanh_model = TanhModel()
        max_pool_model = MaxPool2dModel(kernel_size=2, stride=2)
        group_norm_model = GroupNormModel(num_groups=4, num_channels=8)

        # Share weights from fused model to individual kernels
        conv_transpose_model.conv_transpose.load_state_dict(fused_model.conv_transpose.state_dict())
        batch_norm_model.batch_norm.load_state_dict(fused_model.batch_norm.state_dict())
        group_norm_model.group_norm.load_state_dict(fused_model.group_norm.state_dict())

        # Set all models to eval mode
        conv_transpose_model.eval()
        batch_norm_model.eval()
        tanh_model.eval()
        max_pool_model.eval()
        group_norm_model.eval()

        # Create test input
        x = torch.randn(2, 3, 8, 8)

        with torch.no_grad():
            # Fused model output
            fused_output = fused_model(x)

            # Composed kernel output (sequential pipeline)
            out = conv_transpose_model(x)
            out = batch_norm_model(out)
            out = tanh_model(out)
            out = max_pool_model(out)
            composed_output = group_norm_model(out)

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
