"""
Composition test for 29_Matmul_Mish_Mish

Verifies that the decomposed kernel components, when composed together,
produce the same output as the original fused model.

Test configuration:
  - batch_size=2, in_features=16, out_features=32
"""

import torch
import sys
import os

# Add parent directory to path so we can import kernel/fusion modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from level_0_kernel.linear import Model as LinearModel
from level_0_kernel.mish_1 import Model as Mish1Model
from level_0_kernel.mish_2 import Model as Mish2Model
from level_1_fusion.linear_mish_mish import Model as FusedModel


def test_composition():
    """Test that composed kernels match the original model output."""
    torch.manual_seed(42)

    in_features = 16
    out_features = 32
    batch_size = 2

    # --- Build the original (fused) model ---
    original = FusedModel(in_features, out_features)
    original.eval()

    # --- Build composed model from individual kernels ---
    linear_kernel = LinearModel(in_features, out_features)
    mish1_kernel = Mish1Model()
    mish2_kernel = Mish2Model()

    # Share weights: copy the fused model's linear weights into the kernel
    linear_kernel.linear.weight.data.copy_(original.linear.weight.data)
    linear_kernel.linear.bias.data.copy_(original.linear.bias.data)

    linear_kernel.eval()
    mish1_kernel.eval()
    mish2_kernel.eval()

    # --- Create test input ---
    x = torch.randn(batch_size, in_features)

    # --- Run original model ---
    with torch.no_grad():
        expected = original(x)

    # --- Run composed pipeline: linear -> mish_1 -> mish_2 ---
    with torch.no_grad():
        h = linear_kernel(x)
        h = mish1_kernel(h)
        composed = mish2_kernel(h)

    # --- Compare outputs ---
    match = torch.allclose(expected, composed, rtol=1e-4, atol=1e-5)
    max_diff = (expected - composed).abs().max().item()

    print(f"Original output shape:  {expected.shape}")
    print(f"Composed output shape:  {composed.shape}")
    print(f"Max absolute diff:      {max_diff:.2e}")
    print(f"Outputs match:          {match}")

    assert match, (
        f"Composition test FAILED: max diff = {max_diff:.2e} "
        f"(rtol=1e-4, atol=1e-5)"
    )
    print("PASS: Composed kernels match the original fused model.")
    return True


if __name__ == "__main__":
    success = test_composition()
    sys.exit(0 if success else 1)
