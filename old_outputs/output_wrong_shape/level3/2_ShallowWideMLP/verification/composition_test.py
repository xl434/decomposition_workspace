"""
Composition test for 2_ShallowWideMLP.

Verifies that composing the 5 Level 0 kernels (linear_0, relu_0, linear_1,
relu_1, linear_2) in sequence produces output identical to the Level 1 fusion
and to the original model.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from level_0_kernel.linear_0 import Model as Linear0Kernel
from level_0_kernel.relu_0 import Model as ReLU0Kernel
from level_0_kernel.linear_1 import Model as Linear1Kernel
from level_0_kernel.relu_1 import Model as ReLU1Kernel
from level_0_kernel.linear_2 import Model as Linear2Kernel
from level_1_fusion.mlp_pipeline import Model as FusionModel, get_inputs, get_init_inputs


class OriginalModel(nn.Module):
    """Original model from KernelBench level3/2_ShallowWideMLP."""

    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(OriginalModel, self).__init__()
        layers = []
        current_input_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size
        layers.append(nn.Linear(current_input_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def run_tests():
    """Verify composed kernels match the original model and the fusion."""
    torch.manual_seed(42)

    # Build original model
    original = OriginalModel(*get_init_inputs())
    original.eval()

    # Build fusion model and share weights with original
    fusion = FusionModel(*get_init_inputs())
    fusion.load_state_dict(original.state_dict())
    fusion.eval()

    # Build individual kernels and share weights
    linear_0 = Linear0Kernel(16, 32)
    relu_0 = ReLU0Kernel()
    linear_1 = Linear1Kernel(32, 32)
    relu_1 = ReLU1Kernel()
    linear_2 = Linear2Kernel(32, 8)

    # Copy weights from original model's nn.Sequential
    # network.0 = Linear(16,32), network.1 = ReLU, network.2 = Linear(32,32),
    # network.3 = ReLU, network.4 = Linear(32,8)
    orig_state = original.state_dict()

    linear_0.fc.weight.data.copy_(orig_state["network.0.weight"])
    linear_0.fc.bias.data.copy_(orig_state["network.0.bias"])
    linear_0.eval()

    linear_1.fc.weight.data.copy_(orig_state["network.2.weight"])
    linear_1.fc.bias.data.copy_(orig_state["network.2.bias"])
    linear_1.eval()

    linear_2.fc.weight.data.copy_(orig_state["network.4.weight"])
    linear_2.fc.bias.data.copy_(orig_state["network.4.bias"])
    linear_2.eval()

    # Generate inputs
    inputs = get_inputs()
    x = inputs[0]

    with torch.no_grad():
        # Run original
        out_original = original(x)

        # Run fusion
        out_fusion = fusion(x)

        # Run composed kernels step by step
        step1 = linear_0(x)        # [2, 32]
        step2 = relu_0(step1)      # [2, 32]
        step3 = linear_1(step2)    # [2, 32]
        step4 = relu_1(step3)      # [2, 32]
        step5 = linear_2(step4)    # [2, 8]
        out_composed = step5

    # Compare fusion vs original
    match_fusion = torch.allclose(out_original, out_fusion, rtol=1e-4, atol=1e-5)
    diff_fusion = (out_original - out_fusion).abs().max().item()
    assert match_fusion, f"Fusion vs original mismatch! max diff = {diff_fusion}"
    print(f"[PASS] fusion vs original: match (max diff = {diff_fusion:.2e})")

    # Compare composed vs original
    match_composed = torch.allclose(out_original, out_composed, rtol=1e-4, atol=1e-5)
    diff_composed = (out_original - out_composed).abs().max().item()
    assert match_composed, f"Composed vs original mismatch! max diff = {diff_composed}"
    print(f"[PASS] composed kernels vs original: match (max diff = {diff_composed:.2e})")

    # Verify intermediate shapes
    assert step1.shape == (2, 32), f"Linear_0 output shape wrong: {step1.shape}"
    assert step2.shape == (2, 32), f"ReLU_0 output shape wrong: {step2.shape}"
    assert step3.shape == (2, 32), f"Linear_1 output shape wrong: {step3.shape}"
    assert step4.shape == (2, 32), f"ReLU_1 output shape wrong: {step4.shape}"
    assert step5.shape == (2, 8), f"Linear_2 output shape wrong: {step5.shape}"
    print(f"[PASS] intermediate shapes: L0={step1.shape}, R0={step2.shape}, "
          f"L1={step3.shape}, R1={step4.shape}, L2={step5.shape}")

    print(f"       final output shape: {out_original.shape}")
    return True


if __name__ == "__main__":
    run_tests()
