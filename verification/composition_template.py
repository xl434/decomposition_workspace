"""
Composition Verification Template

This is a TEMPLATE that the agent should fill in for each decomposition.
The agent should create verification/composition_test.py based on this template.

IMPORTANT: This template must be customized for each model!
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# =============================================================================
# STEP 1: Define or import the ORIGINAL model
# =============================================================================

# Option A: Copy the original model class here
class OriginalModel(nn.Module):
    """
    TODO: Paste the original model implementation here.

    This should be an exact copy of the model being decomposed.
    """
    def __init__(self):
        super().__init__()
        # TODO: Copy __init__ from original

    def forward(self, x):
        # TODO: Copy forward from original
        pass


# Option B: Import from file
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples"))
# from simple_transformer import Model as OriginalModel


# =============================================================================
# STEP 2: Import all decomposed kernel components
# =============================================================================

# TODO: Import each kernel component
# Example:
# sys.path.insert(0, str(Path(__file__).parent.parent / "level_0_kernel"))
# from linear_qkv import Model as LinearQKV
# from layer_norm import Model as LayerNorm
# from softmax import Model as Softmax
# ...


# =============================================================================
# STEP 3: Build the composed model from kernels
# =============================================================================

class ComposedModel(nn.Module):
    """
    Model composed from decomposed kernel components.

    This should produce IDENTICAL output to OriginalModel when
    given the same weights and inputs.
    """
    def __init__(self):
        super().__init__()
        # TODO: Initialize using decomposed components
        # Example:
        # self.norm1 = LayerNorm()
        # self.qkv = LinearQKV()
        # ...

    def forward(self, x):
        # TODO: Chain components in same order as original
        # Example:
        # x = self.norm1(x)
        # qkv = self.qkv(x)
        # ...
        pass


# =============================================================================
# STEP 4: Define test inputs (same as original's get_inputs())
# =============================================================================

def get_test_inputs():
    """
    Generate test inputs.

    TODO: Match the original model's get_inputs() function.
    """
    # Example for transformer:
    batch_size = 2
    seq_len = 32
    hidden_dim = 768
    return [torch.randn(batch_size, seq_len, hidden_dim)]


# =============================================================================
# STEP 5: Verification function
# =============================================================================

def verify_composition(rtol=1e-4, atol=1e-5):
    """
    Verify that ComposedModel produces same output as OriginalModel.

    Args:
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if verification passes
    """
    print("=" * 60)
    print("COMPOSITION VERIFICATION")
    print("=" * 60)

    # Create models
    original = OriginalModel()
    composed = ComposedModel()

    # CRITICAL: Copy weights from original to composed
    # This ensures we're comparing computation, not initialization
    try:
        composed.load_state_dict(original.state_dict(), strict=False)
        print("[OK] Weights transferred")
    except Exception as e:
        print(f"[WARNING] Weight transfer issue: {e}")
        print("         Continuing with random weights (may cause mismatch)")

    original.eval()
    composed.eval()

    # Get test inputs
    test_inputs = get_test_inputs()
    print(f"[OK] Test inputs: {[x.shape for x in test_inputs]}")

    # Run both models
    with torch.no_grad():
        try:
            original_output = original(*test_inputs)
            print(f"[OK] Original output: {original_output.shape}")
        except Exception as e:
            print(f"[FAIL] Original model failed: {e}")
            return False

        try:
            composed_output = composed(*test_inputs)
            print(f"[OK] Composed output: {composed_output.shape}")
        except Exception as e:
            print(f"[FAIL] Composed model failed: {e}")
            return False

        # Compare outputs
        if isinstance(original_output, tuple):
            shape_match = all(
                o.shape == c.shape
                for o, c in zip(original_output, composed_output)
            )
            value_match = all(
                torch.allclose(o, c, rtol=rtol, atol=atol)
                for o, c in zip(original_output, composed_output)
            )
            max_diff = max(
                (o - c).abs().max().item()
                for o, c in zip(original_output, composed_output)
            )
        else:
            shape_match = original_output.shape == composed_output.shape
            value_match = torch.allclose(
                original_output, composed_output, rtol=rtol, atol=atol
            )
            max_diff = (original_output - composed_output).abs().max().item()

    # Report results
    print()
    print("-" * 60)
    print(f"Shape match:    {shape_match}")
    print(f"Value match:    {value_match}")
    print(f"Max difference: {max_diff:.2e}")
    print("-" * 60)

    if shape_match and value_match:
        print()
        print("[PASS] Composition verification PASSED!")
        print("       Decomposed components correctly reproduce original.")
        return True
    else:
        print()
        print("[FAIL] Composition verification FAILED!")
        if not shape_match:
            print("       Output shapes do not match.")
        if not value_match:
            print(f"       Values differ by more than tolerance (rtol={rtol}, atol={atol})")
            print("       Debug by comparing intermediate outputs.")
        return False


# =============================================================================
# STEP 6: Debug helper (optional but recommended)
# =============================================================================

def debug_divergence():
    """
    Find where original and composed models diverge.

    Run this if verification fails to identify the problematic component.
    """
    print("=" * 60)
    print("DEBUGGING DIVERGENCE")
    print("=" * 60)

    original = OriginalModel()
    composed = ComposedModel()
    composed.load_state_dict(original.state_dict(), strict=False)

    original.eval()
    composed.eval()

    test_inputs = get_test_inputs()

    with torch.no_grad():
        # TODO: Add intermediate checkpoints here
        # Example:
        #
        # x_orig = test_inputs[0]
        # x_comp = test_inputs[0].clone()
        #
        # # After norm1
        # x_orig = original.norm1(x_orig)
        # x_comp = composed.norm1(x_comp)
        # diff = (x_orig - x_comp).abs().max().item()
        # print(f"After norm1: max_diff = {diff:.2e}")
        # if diff > 1e-5:
        #     print("  ^ DIVERGENCE DETECTED")
        #
        # Continue for each component...
        pass


if __name__ == "__main__":
    success = verify_composition()
    if not success:
        print("\nRunning debug...")
        debug_divergence()
    sys.exit(0 if success else 1)
