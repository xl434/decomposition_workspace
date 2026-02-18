"""
Composition Test: ShallowWideMLP
Source: data/kernelbench/level3/2_ShallowWideMLP.py

Verifies that the decomposed kernel components, when composed together
with shared weights, produce identical outputs to the original model.

Steps:
1. Load the original Model from the source file
2. Build a composed forward pass from individual L0 kernel components
3. Copy weights from the original model into the kernel components
4. Compare outputs with torch.allclose(rtol=1e-4, atol=1e-5)
"""
import sys
import os
import torch
import torch.nn as nn

# Add paths for imports
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DECOMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the workspace root so we can import the original model source
sys.path.insert(0, WORKSPACE_ROOT)

# Import the original model
from data.kernelbench.level3 import __path__ as _level3_path

# We need to import dynamically since the module name starts with a digit
import importlib.util

original_model_path = os.path.join(WORKSPACE_ROOT, "data", "kernelbench", "level3", "2_ShallowWideMLP.py")
spec = importlib.util.spec_from_file_location("original_model", original_model_path)
original_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original_module)

# Import kernel components
kernel_dir = os.path.join(DECOMP_ROOT, "level_0_kernel")
sys.path.insert(0, kernel_dir)

spec_lin1 = importlib.util.spec_from_file_location(
    "linear_16384x32768", os.path.join(kernel_dir, "linear_16384x32768.py"))
linear_16384x32768_mod = importlib.util.module_from_spec(spec_lin1)
spec_lin1.loader.exec_module(linear_16384x32768_mod)

spec_relu = importlib.util.spec_from_file_location(
    "relu_32768", os.path.join(kernel_dir, "relu_32768.py"))
relu_32768_mod = importlib.util.module_from_spec(spec_relu)
spec_relu.loader.exec_module(relu_32768_mod)

spec_lin2 = importlib.util.spec_from_file_location(
    "linear_32768x32768", os.path.join(kernel_dir, "linear_32768x32768.py"))
linear_32768x32768_mod = importlib.util.module_from_spec(spec_lin2)
spec_lin2.loader.exec_module(linear_32768x32768_mod)

spec_lin3 = importlib.util.spec_from_file_location(
    "linear_32768x16384", os.path.join(kernel_dir, "linear_32768x16384.py"))
linear_32768x16384_mod = importlib.util.module_from_spec(spec_lin3)
spec_lin3.loader.exec_module(linear_32768x16384_mod)


class ComposedModel(nn.Module):
    """Composed model built from individual L0 kernel components."""

    def __init__(self):
        super().__init__()
        # Create kernel components
        self.linear1 = linear_16384x32768_mod.Model(
            *linear_16384x32768_mod.get_init_inputs())
        self.relu1 = relu_32768_mod.Model(
            *relu_32768_mod.get_init_inputs())
        self.linear2 = linear_32768x32768_mod.Model(
            *linear_32768x32768_mod.get_init_inputs())
        self.relu2 = relu_32768_mod.Model(
            *relu_32768_mod.get_init_inputs())
        self.linear3 = linear_32768x16384_mod.Model(
            *linear_32768x16384_mod.get_init_inputs())

    def forward(self, x):
        # Compose the kernels in the correct data flow order
        x = self.linear1(x)     # Linear(16384, 32768)
        x = self.relu1(x)       # ReLU
        x = self.linear2(x)     # Linear(32768, 32768)
        x = self.relu2(x)       # ReLU
        x = self.linear3(x)     # Linear(32768, 16384)
        return x


def share_weights(original_model, composed_model):
    """Copy weights from the original model into the composed kernel components."""
    # Original model structure:
    #   network.0 = Linear(16384, 32768)  -> composed_model.linear1.linear
    #   network.1 = ReLU                  -> composed_model.relu1.relu (no params)
    #   network.2 = Linear(32768, 32768)  -> composed_model.linear2.linear
    #   network.3 = ReLU                  -> composed_model.relu2.relu (no params)
    #   network.4 = Linear(32768, 16384)  -> composed_model.linear3.linear

    # Copy Linear layer 0 weights
    composed_model.linear1.linear.weight.data.copy_(
        original_model.network[0].weight.data)
    composed_model.linear1.linear.bias.data.copy_(
        original_model.network[0].bias.data)

    # Copy Linear layer 2 weights
    composed_model.linear2.linear.weight.data.copy_(
        original_model.network[2].weight.data)
    composed_model.linear2.linear.bias.data.copy_(
        original_model.network[2].bias.data)

    # Copy Linear layer 4 weights
    composed_model.linear3.linear.weight.data.copy_(
        original_model.network[4].weight.data)
    composed_model.linear3.linear.bias.data.copy_(
        original_model.network[4].bias.data)


def run_tests():
    try:
        print("=" * 60)
        print("Composition Test: ShallowWideMLP")
        print("=" * 60)

        # Step 1: Create the original model
        print("\n[1] Creating original model...")
        original_model = original_module.Model(*original_module.get_init_inputs())
        original_model.eval()
        print(f"    Original model created with {sum(p.numel() for p in original_model.parameters())} parameters")

        # Step 2: Create the composed model from kernels
        print("[2] Creating composed model from L0 kernels...")
        composed_model = ComposedModel()
        composed_model.eval()
        print(f"    Composed model created with {sum(p.numel() for p in composed_model.parameters())} parameters")

        # Step 3: Share weights
        print("[3] Sharing weights from original -> composed...")
        share_weights(original_model, composed_model)
        print("    Weights shared successfully")

        # Step 4: Generate input
        print("[4] Generating test input...")
        torch.manual_seed(42)
        inputs = original_module.get_inputs()
        print(f"    Input shape: {inputs[0].shape}")

        # Step 5: Run both models
        print("[5] Running forward passes...")
        with torch.no_grad():
            original_output = original_model(*inputs)
            composed_output = composed_model(*inputs)

        print(f"    Original output shape: {original_output.shape}")
        print(f"    Composed output shape: {composed_output.shape}")

        # Step 6: Compare outputs
        print("[6] Comparing outputs...")
        assert original_output.shape == composed_output.shape, \
            f"Shape mismatch: {original_output.shape} vs {composed_output.shape}"

        max_abs_diff = (original_output - composed_output).abs().max().item()
        mean_abs_diff = (original_output - composed_output).abs().mean().item()
        print(f"    Max absolute difference: {max_abs_diff:.2e}")
        print(f"    Mean absolute difference: {mean_abs_diff:.2e}")

        is_close = torch.allclose(original_output, composed_output, rtol=1e-4, atol=1e-5)
        assert is_close, \
            f"Output mismatch! max_abs_diff={max_abs_diff:.2e}, mean_abs_diff={mean_abs_diff:.2e}"

        print("\n" + "=" * 60)
        print("PASS - Composed kernels match original model output")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nFAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
