"""
Composition test for 56_Matmul_Sigmoid_Sum

Verifies that the decomposed kernel components, when composed together,
produce the same output as the original fused model.

Test configuration:
  - batch_size=2, input_size=16, hidden_size=32
"""

import torch
import sys
import os

# Add parent directory to path so we can import kernel/fusion modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from level_0_kernel.linear import Model as LinearModel
from level_0_kernel.sigmoid import Model as SigmoidModel
from level_0_kernel.sum import Model as SumModel
from level_1_fusion.linear_sigmoid_sum import Model as FusedModel


def test_composition():
    """Test that composed kernels match the original model output."""
    torch.manual_seed(42)

    input_size = 16
    hidden_size = 32
    batch_size = 2

    # --- Build the original (fused) model ---
    original = FusedModel(input_size, hidden_size)
    original.eval()

    # --- Build composed model from individual kernels ---
    linear_kernel = LinearModel(input_size, hidden_size)
    sigmoid_kernel = SigmoidModel()
    sum_kernel = SumModel()

    # Share weights: copy the fused model's linear weights into the kernel
    linear_kernel.linear.weight.data.copy_(original.linear.weight.data)
    linear_kernel.linear.bias.data.copy_(original.linear.bias.data)

    linear_kernel.eval()
    sigmoid_kernel.eval()
    sum_kernel.eval()

    # --- Create test input ---
    x = torch.randn(batch_size, input_size)

    # --- Run original model ---
    with torch.no_grad():
        expected = original(x)

    # --- Run composed pipeline: linear -> sigmoid -> sum ---
    with torch.no_grad():
        h = linear_kernel(x)
        h = sigmoid_kernel(h)
        composed = sum_kernel(h)

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
