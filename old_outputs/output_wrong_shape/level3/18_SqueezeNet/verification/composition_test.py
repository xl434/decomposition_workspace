"""
Composition Test for SqueezeNet Decomposition

This test verifies that composing all decomposed components reproduces
the exact output of the original model.
"""

import sys
import os
import torch
import torch.nn as nn

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import original model
sys.path.insert(0, os.path.join(os.path.dirname(parent_dir), '..', '..', 'data', 'kernelbench', 'level3'))
from importlib import import_module

# Load original model
original_module = import_module('18_SqueezeNet')
OriginalModel = original_module.Model

# Import decomposed components
sys.path.insert(0, os.path.join(parent_dir, 'level_3_model'))
from squeezenet import Model as DecomposedModel

def test_composition():
    """Test that decomposed model matches original model output."""
    print("="*80)
    print("COMPOSITION TEST: SqueezeNet")
    print("="*80)

    # Test parameters
    batch_size = 2
    num_classes = 10
    input_shape = (batch_size, 3, 32, 32)

    # Create models
    print("\n1. Creating models...")
    original = OriginalModel(num_classes=num_classes)
    decomposed = DecomposedModel(num_classes=num_classes)

    # Set to eval mode
    original.eval()
    decomposed.eval()

    # Copy weights from original to decomposed
    print("2. Copying weights...")
    decomposed.load_state_dict(original.state_dict())

    # Generate test input
    print("3. Generating test input...")
    torch.manual_seed(42)
    test_input = torch.randn(input_shape, dtype=torch.float32)

    # Run inference
    print("4. Running inference...")
    with torch.no_grad():
        original_output = original(test_input)
        decomposed_output = decomposed(test_input)

    # Verify shapes match
    print("\n5. Verifying shapes...")
    print(f"   Original output shape:   {original_output.shape}")
    print(f"   Decomposed output shape: {decomposed_output.shape}")
    assert original_output.shape == decomposed_output.shape, \
        f"Shape mismatch: {original_output.shape} vs {decomposed_output.shape}"
    print("   [OK] Shapes match")

    # Verify values match
    print("\n6. Verifying values...")
    max_diff = torch.max(torch.abs(original_output - decomposed_output)).item()
    mean_diff = torch.mean(torch.abs(original_output - decomposed_output)).item()

    print(f"   Max difference:  {max_diff:.2e}")
    print(f"   Mean difference: {mean_diff:.2e}")

    # Check with tolerances
    rtol = 1e-4
    atol = 1e-5
    matches = torch.allclose(original_output, decomposed_output, rtol=rtol, atol=atol)

    if matches:
        print(f"   [OK] Values match within tolerance (rtol={rtol}, atol={atol})")
    else:
        print(f"   [FAIL] Values DO NOT match within tolerance (rtol={rtol}, atol={atol})")
        # Print sample values for debugging
        print(f"\n   Sample original values:   {original_output[0, :5]}")
        print(f"   Sample decomposed values: {decomposed_output[0, :5]}")
        return False

    # Verify no NaN or Inf
    print("\n7. Checking for numerical issues...")
    assert not torch.isnan(original_output).any(), "Original output contains NaN"
    assert not torch.isnan(decomposed_output).any(), "Decomposed output contains NaN"
    assert not torch.isinf(original_output).any(), "Original output contains Inf"
    assert not torch.isinf(decomposed_output).any(), "Decomposed output contains Inf"
    print("   [OK] No NaN or Inf values")

    print("\n" + "="*80)
    print("COMPOSITION TEST: PASS")
    print("="*80)
    return True

def test_level2_composition():
    """Test that Level 2 components (features + classifier + flatten) match."""
    print("\n" + "="*80)
    print("LEVEL 2 COMPOSITION TEST")
    print("="*80)

    batch_size = 2
    num_classes = 10

    # Import level 2 components
    sys.path.insert(0, os.path.join(parent_dir, 'level_2_layer'))
    from features import Model as Features
    from classifier import Model as Classifier

    sys.path.insert(0, os.path.join(parent_dir, 'level_0_kernel'))
    from flatten import Model as Flatten

    # Create composed model from L2 components
    class ComposedL2Model(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = Features()
            self.classifier = Classifier(num_classes)
            self.flatten = Flatten()

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            x = self.flatten(x)
            return x

    # Create models
    original = OriginalModel(num_classes=num_classes)
    composed_l2 = ComposedL2Model(num_classes=num_classes)

    original.eval()
    composed_l2.eval()

    # Copy weights manually to handle nested Sequential
    # Features: original.features -> composed_l2.features.features
    for name, param in original.features.named_parameters():
        composed_l2.features.features.get_parameter(name).data.copy_(param.data)

    # Classifier: original.classifier -> composed_l2.classifier.classifier
    for name, param in original.classifier.named_parameters():
        composed_l2.classifier.classifier.get_parameter(name).data.copy_(param.data)

    # Test
    torch.manual_seed(42)
    test_input = torch.randn(batch_size, 3, 32, 32, dtype=torch.float32)

    with torch.no_grad():
        original_output = original(test_input)
        composed_output = composed_l2(test_input)

    # Verify
    max_diff = torch.max(torch.abs(original_output - composed_output)).item()
    print(f"Max difference: {max_diff:.2e}")

    matches = torch.allclose(original_output, composed_output, rtol=1e-4, atol=1e-5)
    if matches:
        print("[OK] Level 2 composition PASS")
        return True
    else:
        print("[FAIL] Level 2 composition FAIL")
        return False

def test_level1_fire_module():
    """Test that FireModule fusion matches original."""
    print("\n" + "="*80)
    print("LEVEL 1 FIRE MODULE TEST")
    print("="*80)

    sys.path.insert(0, os.path.join(parent_dir, 'level_1_fusion'))
    from fire_module import Model as FireModule

    # Create modules
    original_fire = original_module.FireModule(96, 16, 64, 64)
    decomposed_fire = FireModule(96, 16, 64, 64)

    original_fire.eval()
    decomposed_fire.eval()

    # Copy weights
    decomposed_fire.load_state_dict(original_fire.state_dict())

    # Test
    torch.manual_seed(42)
    test_input = torch.randn(2, 96, 6, 6, dtype=torch.float32)

    with torch.no_grad():
        original_output = original_fire(test_input)
        decomposed_output = decomposed_fire(test_input)

    # Verify
    max_diff = torch.max(torch.abs(original_output - decomposed_output)).item()
    print(f"Max difference: {max_diff:.2e}")

    matches = torch.allclose(original_output, decomposed_output, rtol=1e-4, atol=1e-5)
    if matches:
        print("[OK] FireModule fusion PASS")
        return True
    else:
        print("[FAIL] FireModule fusion FAIL")
        return False

if __name__ == "__main__":
    try:
        # Run all tests
        test1 = test_composition()
        test2 = test_level2_composition()
        test3 = test_level1_fire_module()

        if test1 and test2 and test3:
            print("\n" + "="*80)
            print("ALL TESTS PASSED")
            print("="*80)
            sys.exit(0)
        else:
            print("\n" + "="*80)
            print("SOME TESTS FAILED")
            print("="*80)
            sys.exit(1)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
